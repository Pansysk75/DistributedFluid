
#include <algorithm>
#include <cassert>
#include <cstddef>    // for size_t
#include <future>
#include <iostream>
#include <vector>
#include <array>

#include <tile.hpp>
#include <utils/BitmapPlusPlus.hpp>
#include <utils/timer.hpp>


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
    void operator()(T const& prev_tile, T& tile) const
    {
        f_(prev_tile, tile);
    }

    template <typename T>
    void operator()(T& tile) const
    {
        f_(tile);
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

void execute(auto begin, auto end, auto f){
    for (; begin != end; ++begin)
    {
        f(begin);
    }
}

void execute_par(auto begin, auto end, auto f)
{
    std::vector<std::future<void>> futures;
    size_t n_threads = std::thread::hardware_concurrency();
    size_t size = end - begin;

    for (size_t tid = 0; tid < n_threads; ++tid)
    {
        size_t iters = (tid + 1) * size / n_threads - tid * size / n_threads;
        auto sub_end = begin;
        sub_end += iters;

        futures.push_back(std::async(
            std::launch::async,
            [begin, sub_end, &f]() {
                execute(begin, sub_end, f);
            }));

        begin = sub_end;
    }

    for (auto& fut : futures)
    {
        fut.get(); // Wait for all threads to finish
    }
}

void execute(auto begin, auto end, auto p_begin, auto kernel)
{
    for (; begin != end; ++begin, ++p_begin)
    {
        kernel(p_begin, begin);
    }
}

void execute_par(
    auto begin, auto end, auto p_begin, auto kernel)
{
    std::vector<std::future<void>> futures;
    size_t n_threads = std::thread::hardware_concurrency();
    size_t size = end - begin;
    
    for (size_t tid = 0; tid < n_threads; ++tid)
    {
        size_t iters = (tid + 1) * size / n_threads - tid * size / n_threads;
        auto sub_end = begin;
        sub_end += iters;
        
        futures.push_back(std::async(
            std::launch::async,
            [begin, sub_end, p_begin, &kernel]() {
                execute(begin, sub_end, p_begin, kernel);
            }));
        
        begin = sub_end;
        p_begin += iters;
    }
    
    for (auto& fut : futures)
    {
        fut.get(); // Wait for all threads to finish
    }
}

template <typename elem_t, typename F>
void update_tile_inner(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    // Lets avoid the kernel accessing any pad elements
    auto x0 = kernel.dim_x() / 2;
    auto y0 = kernel.dim_y() / 2;
    auto y1 = tile.dim_y() - y0;
    auto x1 = tile.dim_x() - x0;

    auto curr = tile.inner(x0, x1, y0, y1).begin();
    auto p_curr = prev_tile.inner(x0, x1, y0, y1).begin();
    auto size = tile.inner(x0, x1, y0, y1).size();
    auto end = tile.inner(x0, x1, y0, y1).end();

    execute_par(curr, end, p_curr, kernel);
}

template <typename elem_t, typename F>
void update_tile_a(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    auto curr = tile.inner().begin();
    auto p_curr = prev_tile.inner().begin();
    auto size = tile.inner().size();
    auto end = tile.inner().end();

    // execute_par(curr, end, p_curr, kernel);
    execute(curr, end, p_curr, kernel);
}

template <typename elem_t>
// Takes a view and breaks it up into smaller blocks
auto gen_blocked_views(
    typename Tile<elem_t>::inner_2d_tile_t & in_view){
    std::vector<typename Tile<elem_t>::inner_2d_tile_t> views;

    size_t size_x = in_view.size_x();
    size_t size_y = in_view.size_y();

    size_t const block_size_x = 64;
    size_t const block_size_y = 64;

    size_t num_blocks_x = (size_x + block_size_x - 1) / block_size_x;
    size_t num_blocks_y = (size_y + block_size_y - 1) / block_size_y;
    for (size_t i = 0; i < num_blocks_x; ++i)
    {
        for (size_t j = 0; j < num_blocks_y; ++j)
        {
            size_t x_min = i * block_size_x;
            size_t x_max = std::min((i + 1) * block_size_x, size_x);
            size_t y_min = j * block_size_y;
            size_t y_max = std::min((j + 1) * block_size_y, size_y);

            views.emplace_back(in_view.subview(x_min, x_max, y_min, y_max));
        }

    }
    return views;
}

// Blocked, cache friendly
template <typename elem_t, typename F>
void update_tile_b(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    using view_t = typename Tile<elem_t>::inner_2d_tile_t;
    using iter_t = typename Tile<elem_t>::iterator_2d_t;

    view_t inner = tile.inner();
    view_t p_inner = prev_tile.inner();
    std::vector<view_t> views = gen_blocked_views<elem_t>(inner);
    std::vector<view_t> p_views = gen_blocked_views<elem_t>(p_inner);

    execute_par(
        views.begin(), views.end(), p_views.begin(),
        [&](auto& it_p_views, auto& it_views) {
            view_t view = *it_views;
            view_t p_view = *it_p_views;

            iter_t curr = view.begin();
            iter_t p_curr = p_view.begin();
            iter_t end = view.end();

            execute(curr, end, p_curr, kernel);
        });

}

template <typename elem_t, typename F>
void update_tile(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    // update_tile_a(tile, prev_tile, kernel);
    update_tile_b(tile, prev_tile, kernel);
}

template <typename elem_t, typename F>
void update_padding(
    Tile<elem_t>& tile, Kernel<F>& kernel){
        auto l = tile.view(
            0, tile.dim_x(), 0, tile.pad_y());
        auto r = tile.view(
            0, tile.dim_x(), tile.dim_y() - tile.pad_y(), tile.dim_y());
        auto t = tile.view(
            0, tile.pad_x(), 0, tile.dim_y());
        auto b = tile.view(
            tile.dim_x() - tile.pad_x(), tile.dim_x(), 0, tile.dim_y());
        execute(l.begin(), l.end(), kernel);
        execute(r.begin(), r.end(), kernel);
        execute(t.begin(), t.end(), kernel);
        execute(b.begin(), b.end(), kernel);
    }



struct Cell {
    float vx, vy; // Velocity
    float p; // Pressure
    float d; // Density
    std::array<float, 3> s; // Colored smoke
};

void save_to_bitmap(Tile<Cell> const& tile, const std::string& filename)
{
    bmp::Bitmap bitmap(tile.dim_x(), tile.dim_y());
    // parallelize
    execute_par(
        tile.inner().begin(), tile.inner().end(), [&](auto it) {
            auto idx = it - tile.inner().begin();
            auto & c = it.get();
            // Assuming velocity and pressure are in range [0, 255]
            auto vx = static_cast<std::uint8_t>(
                std::clamp(c.s[0]*255.0, 0.0, 255.0));
            auto vy = static_cast<std::uint8_t>(
                std::clamp(c.s[1]*255.0, 0.0, 255.0));
            auto p = static_cast<std::uint8_t>(
                std::clamp(c.s[2]*255.0, 0.0, 255.0));

            bmp::Pixel pixel(vx, vy, p); // RGB pixel

            bitmap.get(idx % tile.dim_x(), idx / tile.dim_x()) = pixel;
            ++idx;
        });
    bitmap.save(filename);
    std::cout << "Bitmap saved to " << filename << std::endl;
}

struct SimulationParams {
    size_t sim_steps = 10000;
    size_t jacobi_iterations = 50;

    double dt = 0.01; // Time step
    double h = 0.1; // Grid spacing
    double viscosity = 0.0005; // Viscosity of the fluid
    double gravity = 9.81; // Gravity acceleration
    double time = 0.0; // Current simulation time
};

int main()
{
    using elem_t = Cell;
    using tile_t = Tile<elem_t>;
    using iter_2d_t = typename tile_t::iterator_2d_t;


    SimulationParams params;
    // The fluid simulation consists of several steps:
    // 1. Apply external forces to the fluid
    // 2. Advection
    // 3. Pressure solver
    // 4. Apply pressure forces

    auto k_update_padding = Kernel(1, 1,    // 1x1 kernel
        [](iter_2d_t& el) -> void {

            // No pressure differential at boundaries
            auto t_ptr = el.tile();
            elem_t* nearest;
            if(el.x() < t_ptr->pad_x()){
                nearest = &t_ptr->inner().begin().get(0, el.y());
            }
            else if(el.x() >= t_ptr->dim_x() - t_ptr->pad_x()){
                nearest = &t_ptr->inner().begin().get(t_ptr->dim_x() - 1, el.y());
            }
            else if(el.y() < t_ptr->pad_y()){
                nearest = &t_ptr->inner().begin().get(el.x(), 0);
            }
            else if(el.y() >= t_ptr->dim_y() - t_ptr->pad_y()){
                nearest = &t_ptr->inner().begin().get(el.x(), t_ptr->dim_y() - 1);
            }
            if(nearest){
                el.get().p = nearest->p; // Set pressure to the nearest inner cell
                el.get().vx = -nearest->vx; // No-slip 
                el.get().vy = -nearest->vy; // No-slip 
            }
            el.get().vx = 0.0;
            el.get().vy = 0.0;
            el.get().d = 1.0; // Density is set to 1.0
            el.get().s = {0.0}; // Smoke density is also zero
        });

    auto k_external_forces = Kernel(1, 1,    // 1x1 kernel
        [&params](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            // curr.get().vy += params.gravity * params.dt;
            //do nothing
        });


    auto k_advection = Kernel(1, 1, // The dimensions have no meaning here
        [&params](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            using elem_t = typename iter_2d_t::value_type;
            auto vx = prev.get().vx;
            auto vy = prev.get().vy;
            // Trace where fluid might have come from

            // Calculate the 4 possible source cells
            auto true_x = curr.x() - vx * params.dt;
            auto true_y = curr.y() - vy * params.dt;
            auto x0 = static_cast<int>(true_x);
            auto y0 = static_cast<int>(true_y);
            auto x1 = x0 + 1; // Next cell in x
            auto y1 = y0 + 1; // Next cell in y

            // Calculate the fractional part of the coordinates
            double dx = true_x - x0; // Fractional part in x
            double dy = true_y - y0; // Fractional part in y

            // If both are not inner, skip
            auto tp = prev.tile();
            if(!tp->is_inner(x0, y0) &&
               !tp->is_inner(x1, y1)) {
                return; // No valid source cells
            }
            // Interpolate the source cell values
            auto c00 = prev.tile()->begin().get(x0, y0);
            auto c10 = prev.tile()->begin().get(x1, y0);
            auto c01 = prev.tile()->begin().get(x0, y1);
            auto c11 = prev.tile()->begin().get(x1, y1);
            // Calculate the interpolation factors
            dx = true_x - x0; // Fractional part in x
            dy = true_y - y0; // Fractional part in y

        // Do I deserve C++ jail
#define INTERP_PROPERTY(prop)                                    \
    (c00.prop * (1 - dx) * (1 - dy) +                            \
     c10.prop * dx * (1 - dy) +                                  \
     c01.prop * (1 - dx) * dy +                                  \
     c11.prop * dx * dy)


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
            
        });

    auto k_pressure_solver = Kernel(3, 3,
        [&params](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            using elem_t = typename iter_2d_t::value_type;
            // Divergence of velocity field
            auto div = (prev.get(1, 0).vx - prev.get(-1, 0).vx) / 2.0 +
                       (prev.get(0, 1).vy - prev.get(0, -1).vy) / 2.0;

            // Update pressure based on previous values
            auto p_sum = prev.get(1, 0).p + prev.get(-1, 0).p +
                         prev.get(0, 1).p + prev.get(0, -1).p;
            auto new_p = (p_sum - div * params.h * params.h) / 4.0;
            // Update the current cell's pressure
            curr.get().p = new_p;
        });

    auto k_apply_pressure = Kernel(3, 3,
        [&params](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            using elem_t = typename iter_2d_t::value_type;
            // Apply pressure forces to velocity
            auto pressure_force_x = (prev.get(1, 0).p - prev.get(-1, 0).p) / (2.0 * params.h);
            auto pressure_force_y = (prev.get(0, 1).p - prev.get(0, -1).p) / (2.0 * params.h);

            curr.get().vx -= pressure_force_x * params.dt / params.h;
            curr.get().vy -= pressure_force_y * params.dt / params.h;

            // Let's do diffusion here
            float diffusion = params.viscosity * params.dt / (params.h * params.h);
            curr.get().vx += diffusion * (prev.get(1, 0).vx + prev.get(-1, 0).vx +
                                          prev.get(0, 1).vx + prev.get(0, -1).vx - 4 * prev.get().vx);
            curr.get().vy += diffusion * (prev.get(1, 0).vy + prev.get(-1, 0).vy +
                                          prev.get(0, 1).vy + prev.get(0, -1).vy - 4 * prev.get().vy);
            
        });

    
    auto k_make_some_fun = Kernel(1, 1,
        [&params](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            // Create a pressure gradient to create a flow
            // curr.get().p = curr.x() * 5.0 * params.h * params.h / params.dt;
            curr.get().vx = 1.0 / params.dt;

            // if not first column, exit
            if(curr.x() != curr.tile()->pad_x()){
                return;
            }

            float speed = 2;
            float pi = 3.14159265358979323846;
            float intensity = 0.5 * (1.5 + 0.5 * sin(2 * pi * params.time * 10 * speed)); // Oscillation intensity
            float offs = 2 * pi / 3;
            float red = 0.3 + 0.7 * sin(2 * pi * params.time * speed); 
            float green = 0.3 + 0.7 * sin(2 * pi * params.time * speed + offs);
            float blue = 0.3 + 0.7 * sin(2 * pi * params.time * speed + 2 * offs);
            curr.get().s[0] = red * intensity;
            curr.get().s[1] = green * intensity;
            curr.get().s[2] = blue * intensity;
        });

    tile_t tile(500, 500, 1, 1);

    // Initialize tiles
    std::generate(tile.begin(), tile.end(), []() {
        elem_t el;
        el.d = 1.0; // Density
        el.s = {0.0}; // Smoke density
        el.vx = 0.0; // Initial horizontal velocity
        el.vy = 0.0; // Initial vertical velocity
        el.p = 0.0; // Initial pressure
        return el;
    });

    // Set some initial conditions
    tile_t::inner_2d_tile_t inner_view =
        tile.inner(size_t(0.3 * tile.dim_x()), size_t(0.7 * tile.dim_x()),
            size_t(0.3 * tile.dim_y()), size_t(0.7 * tile.dim_y()));
    for (auto& cell : inner_view)
    {
        cell.vx = - (std::rand() % 100) * 0.01f / params.dt; // Initial horizontal velocity
        cell.vy = 0.0; // No vertical velocity
        cell.p = 0.0; // Initial pressure
        float s = (std::rand() % 100) * 0.01f;
        cell.s = {s, s, s};
    }

    tile_t prev_tile(tile); // Buffer tile for previous state


    for (size_t step = 0; step < params.sim_steps; ++step)
    {
        {
            // Update padding
            // ps::timer_scoped timer("Update padding");
            update_padding(tile, k_update_padding);
        }
        {
            // Apply external forces
            // ps::timer_scoped timer("Apply external forces");
            update_tile(tile, prev_tile, k_external_forces);
        }

        {
            // Advection
            // ps::timer_scoped timer("Advection");
            update_tile(tile, prev_tile, k_advection);
        }

        {
            // ps::timer_scoped timer("Pressure solver");
            // Pressure solver
            for (size_t iter = 0; iter < params.jacobi_iterations; ++iter)
            {
                update_tile(tile, prev_tile, k_pressure_solver);
                // Store the current state for the next iteration
                // TODO: Not all properties need to be swapped, probably
                prev_tile = tile;
                std::swap(
                    tile, prev_tile);    // Swap tiles for the next iteration
            }
        }

        {
            // Apply pressure forces
            // ps::timer_scoped timer("Apply pressure forces");
            update_tile(tile, prev_tile, k_apply_pressure);
        }

        {
            // Make some fun oscillations
            // ps::timer_scoped timer("Make some fun oscillations");
            tile_t::inner_2d_tile_t fun_view =
                tile.inner(size_t(0), size_t(0.05 * tile.dim_x()),
                    size_t(0.45 * tile.dim_y()), size_t(0.55 * tile.dim_y()));
            tile_t::inner_2d_tile_t prev_fun_view =
                prev_tile.inner(size_t(0), size_t(0.05 * tile.dim_x()),
                    size_t(0.45 * tile.dim_y()), size_t(0.55 * tile.dim_y()));
            execute(fun_view.begin(), fun_view.end(), prev_fun_view.begin(),
                k_make_some_fun);
        }

        // Draw the current state
        if(step % 1 == 4)
        {
            // ps::timer_scoped timer("Save bitmap");
            save_to_bitmap(tile, "output/" + std::to_string(step) + ".bmp");
            std::cout << "Simulation step " << step + 1 << " completed." << std::endl;
        }

        params.time += params.dt; // Update simulation time
        prev_tile = tile;    // Store the current state for the next step
        std::swap(tile, prev_tile); // Swap tiles for the next step
    }


    return 0;
}
