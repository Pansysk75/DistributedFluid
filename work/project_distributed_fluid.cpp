
#include <algorithm>
#include <cassert>
#include <cstddef>    // for size_t
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>
#include <random>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tile_component.hpp>
#include <utils/BitmapPlusPlus.hpp>
#include <utils/timer.hpp>

struct Cell
{
    float vx, vy;              // Velocity
    float p;                   // Pressure
    float d;                   // Density
    std::array<float, 3> s;    // Colored smoke
};

using elem_t = Cell;

static ps::timer_registry timers(
    ps::timer_registry::out_mode::median, ps::timer_registry::out_format::json);

HPX_REGISTER_COMPONENT(hpx::components::component<Tile<elem_t>>, Tile_elem_t)

// kernel class:
// Stores the dimensions, and a function that operates on tiles
template <typename F>
class Kernel
{
public:
    Kernel(size_t dim_x, size_t dim_y, F f)
      : dim_x_(dim_x)
      , dim_y_(dim_y)
      , f_(f)
    {
    }
    template <typename T>
    void operator()(T& tile, T const& prev_tile)
    {
        f_(tile, prev_tile);
    }

    size_t dim_x() const
    {
        return dim_x_;
    }
    size_t dim_y() const
    {
        return dim_y_;
    }

private:
    size_t dim_x_, dim_y_;
    F f_;
};

void iter_apply(auto begin, auto end, auto f)
{
    for (; begin != end; ++begin)
    {
        f(begin);
    }
}

void iter2_apply(auto begin, auto end, auto p_begin, auto f)
{
    for (; begin != end; ++begin, ++p_begin)
    {
        f(begin, p_begin);
    }
}

// Kernel will not touch buffer/ghost cells
template <typename elem_t, typename F>
void update_tile_inner_k(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    // Lets avoid the kernel accessing any pad elements
    auto x0 = kernel.dim_x() / 2;
    auto y0 = kernel.dim_y() / 2;
    auto y1 = tile.dim_y() - y0;
    auto x1 = tile.dim_x() - x0;

    auto curr = tile.inner(x0, x1, y0, y1).begin();
    auto p_curr = prev_tile.inner(x0, x1, y0, y1).begin();
    auto end = tile.inner(x0, x1, y0, y1).end();

    iter2_apply(curr, end, p_curr, kernel);
}

// Kernel might touch buffer/ghost cells, but its
// center will be withing the inner bounds of the tile
template <typename elem_t, typename F>
void update_tile_inner(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    auto curr = tile.inner().begin();
    auto p_curr = prev_tile.inner().begin();
    auto end = tile.inner().end();

    // execute_par(curr, end, p_curr, kernel);
    iter2_apply(curr, end, p_curr, kernel);
}

// Runs through whole tile
// dx/dy indicate if some side is a world boundary
// World boundaries are excluded
template <typename elem_t, typename F>
void update_tile(Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel,
    int dx, int dy)
{
    // size_t x0 = (dx == -1) ? tile.pad_x() : 0;
    // size_t x1 = (dx == 1) ? tile.size_x() - tile.pad_x() : tile.size_x();
    // size_t y0 = (dy == -1) ? tile.pad_y() : 0;
    // size_t y1 = (dy == 1) ? tile.size_y() - tile.pad_y() : tile.size_y();
    // auto curr = tile.view(x0, x1, y0, y1).begin();
    // auto p_curr = prev_tile.view(x0, x1, y0, y1).begin();
    // auto end = tile.view(x0, x1, y0, y1).end();

    auto curr = tile.view().begin();
    auto p_curr = prev_tile.view().begin();
    auto end = tile.view().end();

    // execute_par(curr, end, p_curr, kernel);
    iter2_apply(curr, end, p_curr, kernel);
}

// Runs through whole tile, leaving enough space for the kernel to operate
// dx/dy indicate if some side is a world boundary
// World boundaries are excluded
template <typename elem_t, typename F>
void update_tile_k(Tile<elem_t>& tile, Tile<elem_t>& prev_tile,
    Kernel<F>& kernel, int dx, int dy)
{
    // Lets avoid the kernel accessing out of bounds elements
    auto x0 = kernel.dim_x() / 2;
    auto y0 = kernel.dim_y() / 2;
    auto x1 = tile.size_x() - x0;
    auto y1 = tile.size_y() - y0;
    // x0 += (dx == -1 ? tile.pad_x() : 0);
    // x1 -= (dx == 1 ? tile.pad_x() : 0);
    // y0 += (dy == -1 ? tile.pad_y() : 0);
    // y1 -= (dy == 1 ? tile.pad_y() : 0);

    auto curr = tile.view(x0, x1, y0, y1).begin();
    auto p_curr = prev_tile.view(x0, x1, y0, y1).begin();
    auto end = tile.view(x0, x1, y0, y1).end();

    iter2_apply(curr, end, p_curr, kernel);
}

template <typename elem_t, typename F>
void update_view(typename Tile<elem_t>::inner_2d_tile_t& view,
    typename Tile<elem_t>::inner_2d_tile_t& prev_view, Kernel<F>& kernel)
{
    iter2_apply(view.begin(), view.end(), prev_view.begin(), kernel);
}

void save_to_bitmap(Tile<Cell> const& tile, const std::string& filename)
{
    bmp::Bitmap bitmap(tile.dim_x(), tile.dim_y());

    // First go through elements to find min/max values for normalization
    auto minmax_p = std::minmax_element(
        tile.inner().begin(), tile.inner().end(), [](auto a, auto b) {
            return a.p < b.p;
        });

    auto minmax_vx = std::minmax_element(
        tile.inner().begin(), tile.inner().end(), [](auto a, auto b) {
            return a.vx < b.vx;
        });

    auto norm_px = [](float val, float min_val, float max_val) {
        return static_cast<std::uint8_t>(
            std::clamp((val - min_val) / (max_val - min_val) * 255.0, 0.0, 255.0));
    };

    iter_apply(tile.inner().begin(), tile.inner().end(), [&](auto it) {
        auto idx = it - tile.inner().begin();
        auto& c = it.get();
        // Assuming velocity and pressure are in range [0, 255]
        std::uint8_t avg_s = (c.s[0] + c.s[1] + c.s[2]) / 3.0;

        // std::uint8_t r = 0; //norm_px(c.p, minmax_p.first->p, minmax_p.second->p);
        // std::uint8_t g = norm_px(c.vx, minmax_vx.first->vx, minmax_vx.second->vx);
        // auto b = avg_s; // static_cast<std::uint8_t>(std::clamp(avg_s * 255.0, 0.0, 255.0));

        std::uint8_t r = std::clamp(c.s[0] * 255.0, 0.0, 255.0);
        std::uint8_t g = std::clamp(c.s[1] * 255.0, 0.0, 255.0);
        std::uint8_t b = std::clamp(c.s[2] * 255.0, 0.0, 255.0);

            // static_cast<std::uint8_t>(std::clamp(c.s[1] * 255.0, 0.0, 255.0));  

        //lazy hack to draw borders between the remote tiles
        if (c.d == 42.0){
            r = g = b = 255;
        }
        bmp::Pixel pixel(r, g, b);    // RGB pixel

        bitmap.get(idx % tile.dim_x(), idx / tile.dim_x()) = pixel;
        ++idx;
    });
    bitmap.save(filename);
    std::cout << "Bitmap saved to " << filename << std::endl;
}

template <typename TileType>
TileType::inner_2d_tile_t get_ghost_cells_view(TileType& t, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    // this is a bit ugly
    using inner_view_t = typename TileType::inner_2d_tile_t;
    inner_view_t view;
    if (dx == -1)    // Left slice
        view = t.view(0, t.pad_x(), t.pad_y(), t.dim_y() + t.pad_y());
    if (dx == 1)    // Right slice
        view = t.view(t.dim_x() + t.pad_x(), t.dim_x() + 2 * t.pad_x(),
            t.pad_y(), t.dim_y() + t.pad_y());
    if (dy == -1)    // Upper slice
        view = t.view(t.pad_x(), t.dim_x() + t.pad_x(), 0, t.pad_y());
    if (dy == 1)    // Lower slice
        view = t.view(t.pad_x(), t.dim_x() + t.pad_x(), t.dim_y() + t.pad_y(),
            t.dim_y() + 2 * t.pad_y());
    
    assert(dx == 0 || view.size() == t.pad_x() * t.dim_y());
    assert(dy == 0 || view.size() == t.pad_y() * t.dim_x());

    return view;
}

// Gets side cells, which is a strip the width/height of the tile padding
// This is used to send the ghost elements to the neighbor tiles
template <typename TileType>
TileType::inner_2d_tile_t get_side_cells_view(TileType& t, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    using inner_view_t = typename TileType::inner_2d_tile_t;
    inner_view_t view;
    if (dx == -1)    // Left slice
        view = t.inner(0, t.pad_x(), 0, t.dim_y());
    if (dx == 1)    // Right slice
        view = t.inner(t.dim_x() - t.pad_x(), t.dim_x(), 0, t.dim_y());
    if (dy == -1)    // Upper slice
        view = t.inner(0, t.dim_x(), 0, t.pad_y());
    if (dy == 1)    // Lower slice
        view = t.inner(0, t.dim_x(), t.dim_y() - t.pad_y(), t.dim_y());

    assert(dx == 0 || view.size() == t.pad_x() * t.dim_y());
    assert(dy == 0 || view.size() == t.pad_y() * t.dim_x());

    return view;
}

template <typename TileType>
void receive_ghost_elements(hpx::id_type tile_id,
    std::vector<typename TileType::value_type> to_receive, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();

    // For now, make a new vector to send the data
    using elem_t = typename TileType::value_type;
    using inner_view_t = typename TileType::inner_2d_tile_t;

    inner_view_t inner_view = get_ghost_cells_view(*tile_ptr, dx, dy);

    assert(inner_view.size() == to_receive.size());

    // Now copy the elements from the vector to the tile
    auto it = inner_view.begin();
    for (const auto& elem : to_receive)
    {
        it.get() = elem;
        ++it;
    }
}

HPX_PLAIN_ACTION(
    receive_ghost_elements<Tile<elem_t>>, receive_ghost_elements_action);

template <typename TileType>
void copy_ghost_elements(
    hpx::id_type curr_tile_id, hpx::id_type neighbor_tile_id, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    auto curr_tile_ptr = hpx::get_ptr<TileType>(curr_tile_id).get();

    // For now, make a new vector to send the data
    using elem_t = typename TileType::value_type;
    std::vector<elem_t> to_send;
    using inner_view_t = typename TileType::inner_2d_tile_t;
    inner_view_t inner_view = get_side_cells_view(*curr_tile_ptr, dx, dy);

    // Now copy the elements to the vector
    // TODO: reserve
    for (auto it = inner_view.begin(); it != inner_view.end(); ++it)
    {
        to_send.push_back(it.get());
    }

    assert(to_send.size() == inner_view.size());

    // Send the data to the neighbor tile
    // Negate dx/dy, as the data should be received from the opposite side
    // (e.g. if we are sending from left, we need to receive on the right)
    auto f = hpx::async(receive_ghost_elements_action{},
        hpx::colocated(neighbor_tile_id), neighbor_tile_id, to_send, -dx, -dy);

    f.get();
}

HPX_PLAIN_ACTION(copy_ghost_elements<Tile<elem_t>>, copy_ghost_elements_action);

template <typename TileType>
hpx::future<void> emit_ghost_elements(
    TileType& world, typename TileType::iterator_2d_t it)
{
    auto update_direction = [](auto it, int dx, int dy) {
        hpx::id_type curr_id = it->t_id;
        hpx::id_type neighbor_id = it.get(dx, dy).t_id;
        auto f1 = hpx::async(copy_ghost_elements_action{},
            hpx::colocated(curr_id), curr_id, neighbor_id, dx, dy);
        // // Let's copy buffers too, not sure which is needed
        // hpx::id_type curr_buf_id = it->buf_t_id;
        // hpx::id_type neighbor_buf_id = it.get(dx, dy).buf_t_id;
        // auto f2 = hpx::async(copy_ghost_elements_action{},
        //     hpx::colocated(curr_buf_id), curr_buf_id, neighbor_buf_id, dx, dy);
        return hpx::when_all(f1/*, f2*/);
    };

    std::vector<hpx::future<void>> comm;

    if (it.x() > 0)
        comm.push_back(update_direction(it, -1, 0));

    if (it.x() < world.dim_x() - 1)
        comm.push_back(update_direction(it, 1, 0));

    if (it.y() > 0)
        comm.push_back(update_direction(it, 0, -1));

    if (it.y() < world.dim_y() - 1)
        comm.push_back(update_direction(it, 0, 1));

    // Create new future that waits for all communication to finish
    return hpx::when_all(comm.begin(), comm.end());
}

template <typename WorldTile>
hpx::future<void> neighbors_ready(
    WorldTile& world, typename WorldTile::iterator_2d_t it)
{
    std::vector<hpx::shared_future<void>> comm_futures;
    if (it.x() > 0)
        comm_futures.push_back(it.get(-1, 0).fut);
    if (it.x() < world.dim_x() - 1)
        comm_futures.push_back(it.get(1, 0).fut);
    if (it.y() > 0)
        comm_futures.push_back(it.get(0, -1).fut);
    if (it.y() < world.dim_y() - 1)
        comm_futures.push_back(it.get(0, 1).fut);

    return hpx::when_all(comm_futures.begin(), comm_futures.end());
}

struct SimulationParams
{
    static constexpr size_t sim_steps = 10000;
    static constexpr size_t jacobi_iterations = 5;
    static constexpr double dt = 0.01;             // Time step
    static constexpr double h = 0.1;               // Grid spacing
    static constexpr double viscosity = 0.0005;    // Viscosity of the fluid
    static constexpr double gravity = 9.81;        // Gravity acceleration
    static constexpr double time = 0.0;            // Current simulation time
};
template <typename Tile, typename iter_2d_t = typename Tile::iterator_2d_t>
void k_update_padding(iter_2d_t& curr, iter_2d_t const& prev, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    auto t_ptr = curr.tile();

    // Take care of corner sections
    if (!(curr.x() >= t_ptr->pad_x() && curr.x() < t_ptr->dim_x()) ||
        !(curr.y() >= t_ptr->pad_y() && curr.y() < t_ptr->dim_y()))
    {
        curr.get().vx = 0.0;
        curr.get().vy = 0.0;
        curr.get().d = 0.0;      // Density is set to 1.0
        curr.get().s = {0.0, 0.0, 0.0};    // Smoke density is also zero
        return;    // Not in the inner section
    }

    // No pressure differential at boundaries
    // Get nearest inner cell, and copy its pressure
    using elem_t = typename Tile::value_type;
    elem_t* nearest = nullptr;
    if (dx == -1)
        nearest = &t_ptr->inner().get(0, curr.y());
    else if (dx == 1)
        nearest = &t_ptr->inner().get(t_ptr->dim_x() - 1, curr.y());
    else if (dy == -1)
        nearest = &t_ptr->inner().get(curr.x(), 0);
    else if (dy == 1)
        nearest = &t_ptr->inner().get(curr.x(), t_ptr->dim_y() - 1);

    if (nearest)
    {
        curr.get().p = nearest->p;    // Set pressure to the nearest inner cell
        curr.get().vx = -nearest->vx;    // No-slip
        curr.get().vy = -nearest->vy;    // No-slip
    }
};

template <typename Tile, typename iter_2d_t = typename Tile::iterator_2d_t>
void k_advection(iter_2d_t& curr, iter_2d_t const& prev)
{
    using elem_t = typename iter_2d_t::value_type;
    auto vx = prev->vx;
    auto vy = prev->vy;
    // Trace where fluid might have come from

    // Calculate the 4 possible source cells
    auto true_x = curr.x() - vx * SimulationParams::dt;
    auto true_y = curr.y() - vy * SimulationParams::dt;
    auto x0 = static_cast<int>(true_x);
    auto y0 = static_cast<int>(true_y);
    auto x1 = x0 + 1;    // Next cell in x
    auto y1 = y0 + 1;    // Next cell in y

    // Calculate the fractional part of the coordinates
    double dx = true_x - x0;    // Fractional part in x
    double dy = true_y - y0;    // Fractional part in y

    auto tp = prev.tile();
    if (!tp->is_valid(x0, y0) || !tp->is_valid(x1, y1))
    {
        return;
    }
    // Interpolate the source cell values
    auto c00 = prev.tile()->get(x0, y0);
    auto c10 = prev.tile()->get(x1, y0);
    auto c01 = prev.tile()->get(x0, y1);
    auto c11 = prev.tile()->get(x1, y1);
    // Calculate the interpolation factors
    dx = true_x - x0;    // Fractional part in x
    dy = true_y - y0;    // Fractional part in y

// Do I deserve C++ jail
#define INTERP_PROPERTY(prop)                                                  \
    (c00.prop * (1 - dx) * (1 - dy) + c10.prop * dx * (1 - dy) +               \
        c01.prop * (1 - dx) * dy + c11.prop * dx * dy)

    // Interpolate the properties
    auto& c = curr.get();
    c.vx = INTERP_PROPERTY(vx);
    c.vy = INTERP_PROPERTY(vy);
    c.d = INTERP_PROPERTY(d);
    float s0 = INTERP_PROPERTY(s[0]);
    float s1 = INTERP_PROPERTY(s[1]);
    float s2 = INTERP_PROPERTY(s[2]);
    c.s = {s0, s1, s2};

#undef INTERP_PROPERTY
};

// 3x3 kernel
// This reads and write
template <typename Tile, typename iter_2d_t = typename Tile::iterator_2d_t>
void k_pressure_solver(iter_2d_t& curr, iter_2d_t const& prev)
{
    using elem_t = typename iter_2d_t::value_type;
    // Divergence of velocity field
    auto div = (prev.get(1, 0).vx - prev.get(-1, 0).vx) / 2.0 +
        (prev.get(0, 1).vy - prev.get(0, -1).vy) / 2.0;

    // Update pressure based on previous values
    auto p_sum = prev.get(1, 0).p + prev.get(-1, 0).p + prev.get(0, 1).p +
        prev.get(0, -1).p;
    auto h = SimulationParams::h;
    auto new_p = (p_sum - div * h * h) / 4.0;
    // Update the current cell's pressure
    curr->p = new_p;
}

template <typename Tile, typename iter_2d_t = typename Tile::iterator_2d_t>
void k_apply_pressure(iter_2d_t& curr, iter_2d_t const& prev)
{
    using elem_t = typename iter_2d_t::value_type;
    // Apply pressure forces to velocity
    auto h = SimulationParams::h;
    auto dt = SimulationParams::dt;
    auto pressure_force_x = (prev.get(1, 0).p - prev.get(-1, 0).p) / (2.0 * h);
    auto pressure_force_y = (prev.get(0, 1).p - prev.get(0, -1).p) / (2.0 * h);

    curr->vx -= pressure_force_x * dt / h;
    curr->vy -= pressure_force_y * dt / h;

    // Let's also do diffusion while we're here
    auto viscosity = SimulationParams::viscosity;
    float diffusion = viscosity * dt / (h * h);
    curr->vx += diffusion *
        (prev.get(1, 0).vx + prev.get(-1, 0).vx + prev.get(0, 1).vx +
            prev.get(0, -1).vx - 4 * prev.get().vx);
    curr->vy += diffusion *
        (prev.get(1, 0).vy + prev.get(-1, 0).vy + prev.get(0, 1).vy +
            prev.get(0, -1).vy - 4 * prev.get().vy);
};

// I made this a callable struct bc i need to put the current time somewhere
// (and I couldn't pass it as a parameter)
template <typename Tile, typename iter_2d_t = typename Tile::iterator_2d_t>
struct k_make_some_fun
{
    double time_;
    int dx_, dy_;
    std::mt19937 rng_;

    void operator()(iter_2d_t& curr, iter_2d_t const& prev)
    {
        // If this is the first iteration, give some initial conditions
        if (time_ == 0.0)
        {
            // Now use sin to create a bell-like wave
            size_t x = curr.x() - curr.tile()->pad_x();
            size_t y = curr.y() - curr.tile()->pad_y();
            size_t s_x = curr.tile()->dim_x();
            size_t s_y = curr.tile()->dim_y();

            double pi = 3.1415;
            float bell = std::sin(pi * x / s_x) * std::sin(pi * y / s_y);

            // Each tile gets a random color
            rng_ = std::mt19937((size_t)curr.tile());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng_);    // Random red component
            float g = dist(rng_);    // Random green component
            float b = dist(rng_);    // Random blue component

            // And a random direction
            int dice = std::uniform_int_distribution<int>(0, 3)(rng_);
            float A = std::uniform_real_distribution<float>(0.5f, 1.0f)(rng_);
            float x_vel = (dice == 0) ? 1.0f : ((dice == 2) ? -1.0f : 0.0f);
            float y_vel = (dice == 1) ? 1.0f : ((dice == 3) ? -1.0f : 0.0f);

            curr->s = {r * bell, g * bell, b * bell};    // Initial smoke color
            curr->vx = A * bell * x_vel / SimulationParams::dt;    // Initial horizontal velocity
            curr->vy = A * bell * y_vel / SimulationParams::dt;    // Initial vertical velocity
            curr->p = 0.0;                     // Initial pressure
            // curr->vx = (float)std::sin(2 * pi * curr.x() / 68.0) / SimulationParams::dt;    // Initial horizontal velocity
            // curr->vy = (float)std::sin(2 * pi * curr.x() / 42.0) / SimulationParams::dt;    // Initial vertical velocity
            // curr->s = {(float)std::sin(2 * pi * curr.x() / 50.0)*0.5f + 0.5f,
            //     (float)std::sin(2 * pi * curr.y() / 40.0)*0.5f + 0.5f,
            //     (float)std::sin(2 * pi * (curr.x() + curr.y())*0.5f / 60.0) + 0.5f};    // Initial smoke color
        }

        // Only apply to a thin strip at the left of the simulated area
        if (dx_ != -1)
            return;
        if (curr.x() >= curr.tile()->pad_x() + 10 ||
            curr.y() < size_t(0.45 * curr.tile()->dim_y()) ||
            curr.y() >= size_t(0.55 * curr.tile()->dim_y()))
        {
            return;    // No fun here
        }
        // Create a pressure gradient to create a flow
        // curr.get().p = curr.x() * 5.0 * params.h * params.h / params.dt;
        curr.get().vx = 1.0 / SimulationParams::dt;

        float speed = 0.5;
        float pi = 3.1415;
        float red = 0.3 + 0.7 * sin(2 * pi * time_ * speed * 0.93 + 2*pi/3.0);
        float green = 0.3 + 0.7 * sin(2 * pi * time_ * speed + 4*pi/3.0);
        float blue = 0.3 + 0.7 * sin(2 * pi * time_ * speed * 1.07 + 6*pi/3.0);
        curr.get().s[0] = red;
        curr.get().s[1] = green;
        curr.get().s[2] = blue;
    }
};

template <typename TileType>
void pre_solver_remote(
    hpx::id_type tile_id, hpx::id_type prev_tile_id, int dx, int dy)
{
    using iter_2d_t = typename TileType::iterator_2d_t;
    using elem_t = typename TileType::value_type;
    // Get local tile instances from hpx::id_type
    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();
    auto prev_tile_ptr = hpx::get_ptr<TileType>(prev_tile_id).get();

    // Copy to buffer
    *prev_tile_ptr = *tile_ptr;

    // Update padding cells, if we are on the edge of the world
    if (dx != 0)
    {
        auto k = Kernel(1, 1, [dx](auto curr, auto prev) {
            return k_update_padding<TileType>(curr, prev, dx, 0);
        });
        auto view = get_ghost_cells_view(*tile_ptr, dx, 0);
        auto prev_view = get_ghost_cells_view(*prev_tile_ptr, dx, 0);
        update_view<elem_t>(view, prev_view, k);
    }
    if (dy != 0)
    {
        auto k = Kernel(1, 1, [dy](auto curr, auto prev) {
            return k_update_padding<TileType>(curr, prev, 0, dy);
        });
        auto view = get_ghost_cells_view(*tile_ptr, 0, dy);
        auto prev_view = get_ghost_cells_view(*prev_tile_ptr, 0, dy);
        update_view<elem_t>(view, prev_view, k);
    }

    auto k2 = Kernel(1, 1, k_advection<TileType>);
    update_tile(*tile_ptr, *prev_tile_ptr, k2, dx, dy);
}

HPX_PLAIN_ACTION(pre_solver_remote<Tile<elem_t>>, pre_solver_remote_action);

template <typename TileType>
void solver_remote(
    hpx::id_type tile_id, hpx::id_type prev_tile_id, int dx, int dy)
{
    using iter_2d_t = typename TileType::iterator_2d_t;
    // Get local tile instances from hpx::id_type
    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();
    auto prev_tile_ptr = hpx::get_ptr<TileType>(prev_tile_id).get();

    // Each iteration of the solver has to use a buffer, but we want to avoid 
    // modifying the time-step buffer of "prev_tile" on every iteration.
    // So, we use "tile_ptr" to read from, and we write to a temporary buffer
    // Only after one whole iteration finishes, we copy the result to "tile_ptr"

    TileType dest_buffer_tile(*tile_ptr);
    TileType &source_tile = *tile_ptr;

    auto k1 = Kernel(3, 3, k_pressure_solver<TileType>);
    update_tile_inner(dest_buffer_tile, source_tile, k1);

    // Now we can put the result of this iteration back to the active tile
    std::swap(*tile_ptr, dest_buffer_tile);
}

HPX_PLAIN_ACTION(solver_remote<Tile<elem_t>>, solver_remote_action);

template <typename TileType>
void post_solver_remote(hpx::id_type tile_id, hpx::id_type prev_tile_id,
    double time, int dx, int dy)
{
    using iter_2d_t = typename TileType::iterator_2d_t;
    // Get local tile instances from hpx::id_type
    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();
    auto prev_tile_ptr = hpx::get_ptr<TileType>(prev_tile_id).get();

    auto k1 = Kernel(3, 3, &k_apply_pressure<TileType>);
    update_tile_k(*tile_ptr, *prev_tile_ptr, k1, dx, dy);

    // Make some fun
    auto k_obj = k_make_some_fun<TileType>{time, dx, dy};
    auto k2 = Kernel(1, 1, k_obj);
    update_tile_inner(*tile_ptr, *prev_tile_ptr, k2);

}

HPX_PLAIN_ACTION(post_solver_remote<Tile<elem_t>>, post_solver_remote_action);

struct RemoteTile
{
    hpx::id_type t_id;
    hpx::id_type buf_t_id;
    hpx::shared_future<void> fut;
};

template <typename TileType>
void sim_iteration(TileType& world)
{
    auto world_edge_flags = [&world](auto it) {
        // Are we on the edge of the world?
        int dx = it.x() == 0 ? -1 : (it.x() == world.dim_x() - 1 ? 1 : 0);
        int dy = it.y() == 0 ? -1 : (it.y() == world.dim_y() - 1 ? 1 : 0);
        return std::make_pair(dx, dy);
    };

    std::vector<hpx::future<void>> futures;
    {

        futures.clear();
        for (auto it = world.begin(); it != world.end(); ++it)
        {
            futures.push_back(emit_ghost_elements(world, it));
        }
        hpx::wait_all(futures);

        // Communicate ghost elements between tiles
        auto _ = timers.time_scope("Pre-solver");
        std::cout << "Communicating ghost elements..." << std::endl;
        for (auto it = world.begin(); it != world.end(); ++it)
        {
            it->fut = emit_ghost_elements(world, it);
        }

        std::cout << "Running pre-solver kernel..." << std::endl;
        for (auto it = world.begin(); it != world.end(); ++it)
        {
            // All neighbors must have finished communicating for this tile
            // to proceed
            auto f1 = neighbors_ready(world, it);

            auto [dx, dy] = world_edge_flags(it);

            auto f2 = f1.then([it, dx, dy](hpx::future<void> f) {
                // This will run after all communication is done
                // Runs on the locality where the tile is located
                // TODO: Any better way to do this?
                f.get();
                hpx::async(pre_solver_remote_action{}, hpx::colocated(it->t_id),
                    it->t_id, it->buf_t_id, dx, dy).get();
            });

            futures.push_back(std::move(f2));
        }
        // Wait for all kernels to finish
        hpx::wait_all(futures);
    }

    // sync
    futures.clear();
    for (auto it = world.begin(); it != world.end(); ++it)
    {
        futures.push_back(emit_ghost_elements(world, it));
    }
    hpx::wait_all(futures);

    {
        auto _ = timers.time_scope("Solver(" +
            std::to_string(SimulationParams::jacobi_iterations) +
            " iterations)");
        for (int i = 0; i < SimulationParams::jacobi_iterations; ++i)
        {
            std::cout << "Running solver kernel iteration: " << i << std::endl;
            futures.clear();
            for (auto it = world.begin(); it != world.end(); ++it)
            {
                auto [dx, dy] = world_edge_flags(it);
                futures.push_back(hpx::async(solver_remote_action{},
                    hpx::colocated(it->t_id), it->t_id, it->buf_t_id, dx, dy));
            }
            // Wait for all solver kernels to finish
            hpx::wait_all(futures);

            futures.clear();
            for (auto it = world.begin(); it != world.end(); ++it)
            {
                futures.push_back(emit_ghost_elements(world, it));
            }
            // Wait for all ghost elements to be communicated
            hpx::wait_all(futures);
        }
    }

    {
        // Now we can run the post-solver kernel on each tile
        auto _ = timers.time_scope("Post-solver");
        std::cout << "Running post-solver kernel..." << std::endl;
        futures.clear();
        static double time = 0.0;
        for (auto it = world.begin(); it != world.end(); ++it)
        {
            auto [dx, dy] = world_edge_flags(it);
            futures.push_back(hpx::async(post_solver_remote_action{},
                hpx::colocated(it->t_id), it->t_id, it->buf_t_id, time, dx,
                dy));
        }
        time += SimulationParams::dt;

        hpx::wait_all(futures);
    }
}

template <typename TileType>
std::vector<typename TileType::value_type> sample_remote(
    hpx::id_type tile_id, size_t s_x, size_t s_y)
{
    // Get local tile instance from hpx::id_type
    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();
    std::vector<typename TileType::value_type> data;
    data.reserve(s_x * s_y);

    // Sample the tile data
    // auto view = tile_ptr->inner();
    auto view = tile_ptr->view();
    double x_step = static_cast<double>(view.size_x()) / s_x;
    double y_step = static_cast<double>(view.size_y()) / s_y;

    // double x_step = static_cast<double>(tile_ptr->dim_x()) / s_x;
    // double y_step = static_cast<double>(tile_ptr->dim_y()) / s_y;

    for (size_t y = 0; y < s_y; ++y)
    {
        for (size_t x = 0; x < s_x; ++x)
        {
            // Calculate the coordinates in the tile
            size_t tile_x = static_cast<size_t>(x * x_step);
            size_t tile_y = static_cast<size_t>(y * y_step);
              
            //lazy hack to draw borders between the remote tiles
            if (y == 0 || y == s_y - 1 || x == 0 || x == s_x - 1)
            {
                typename TileType::value_type elem{};
                elem.d = 42.0;    
                data.push_back(elem);
            }else{
                // Get the element at the calculated coordinates
                auto& elem = view.get(tile_x, tile_y);
                data.push_back(elem);
            }
            
        }
    }
    assert(data.size() == s_x * s_y);
    return data;
}

HPX_PLAIN_ACTION(sample_remote<Tile<elem_t>>, sample_remote_action);

template <typename TileType, typename SamplesTile>
void sample_world(TileType& world, SamplesTile& samples)
{
    std::vector<hpx::future<void>> futures;
    // Iterate over the world tiles
    for (auto it = world.begin(); it != world.end(); ++it)
    {
        auto x0 = it.x() * samples.dim_x() / world.dim_x();
        auto x1 = (it.x() + 1) * samples.dim_x() / world.dim_x();
        auto y0 = it.y() * samples.dim_y() / world.dim_y();
        auto y1 = (it.y() + 1) * samples.dim_y() / world.dim_y();

        auto s_x = x1 - x0;    // Sample width
        auto s_y = y1 - y0;    // Sample height

        // Request the tile to sample its data
        auto f1 = hpx::async(sample_remote_action{}, hpx::colocated(it->t_id),
            it->t_id, s_x, s_y);

        // When ready, copy into the respective view
        auto view = samples.view(x0, x1, y0, y1);

        
        auto f2 = f1.then([&samples, view](auto f) mutable {
            // This will run after the sample is ready
            auto data = f.get();
            assert(view.size() == data.size());
            auto it = view.begin();
            for (const auto& elem : data)
            {
                it.get() = elem;    // Copy the sampled data
                ++it;
            }
        });

        futures.push_back(std::move(f2));
    }

    // Wait for all sampling to finish
    hpx::wait_all(futures);
}

int hpx_main()
{
    // A 2d grid. Each grid element consists of two tiles (main and buffer)
    // A future is also used to synchronize the computation
    size_t world_dim_x = 8;
    size_t world_dim_y = 8;
    Tile<RemoteTile> world(world_dim_x, world_dim_y, 0, 0);

    size_t dim_x = 100;    // Tile width
    size_t dim_y = 100;    // Tile height
    size_t ghst_x = 4;    // Ghost cell padding in x
    size_t ghst_y = 4;    // Ghost cell padding in y
    // Initialize the tiles, distributing them across localities
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    size_t n_tiles = world_dim_x * world_dim_y;
    size_t idx = 0;
    for (auto& [id, buff_id, _] : world)
    {
        // Assign each tile to a locality
        size_t loc_idx = idx * localities.size() / n_tiles;
        auto loc = localities[loc_idx];
        id = hpx::new_<Tile<elem_t>>(loc, dim_x, dim_y, ghst_x, ghst_y).get();
        buff_id = hpx::new_<Tile<elem_t>>(loc, dim_x, dim_y, ghst_x, ghst_y).get();
        ++idx;
    }

    Tile<elem_t> samples(800, 800, 0, 0);
    for (int i = 0; i < SimulationParams::sim_steps; ++i)
    {
        std::cout << "Simulation step: " << i << std::endl;
        // Run a simulation iteration
        sim_iteration(world);
        // Sample the distributed world and copy locally
        sample_world(world, samples);
        // Save the sampled data to a bitmap
        save_to_bitmap(samples, "output/" + std::to_string(i) + ".bmp");
    }

    return hpx::finalize();    // Shutdown HPX runtime
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}