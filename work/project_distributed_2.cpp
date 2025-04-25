
#include <algorithm>
#include <cassert>
#include <cstddef>    // for size_t
#include <future>
#include <iostream>
#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tile_component.hpp>
#include <utils/BitmapPlusPlus.hpp>
#include <utils/timer.hpp>

using elem_t = float;

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

template <typename elem_t, typename F>
void update_tile_inner_seq(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    std::swap(prev_tile, tile);
    size_t n_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // We must avoid the kernel accessing any pad elements
    auto x0 = kernel.dim_x() / 2;
    auto y0 = kernel.dim_y() / 2;
    auto y1 = tile.dim_y() - y0;
    auto x1 = tile.dim_x() - x0;

    auto curr = tile.inner(x0, x1, y0, y1).begin();
    auto p_curr = prev_tile.inner(x0, x1, y0, y1).begin();
    auto size = tile.inner(x0, x1, y0, y1).size();

    // Sequential impl
    for (; curr != tile.inner(x0, x1, y0, y1).end(); ++curr, ++p_curr)
    {
        kernel(p_curr, curr);
    }
}

template <typename elem_t, typename F>
void update_tile_inner_par(
    Tile<elem_t>& tile, Tile<elem_t>& prev_tile, Kernel<F>& kernel)
{
    std::swap(prev_tile, tile);
    size_t n_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // We must avoid the kernel accessing any pad elements
    auto x0 = kernel.dim_x() / 2;
    auto y0 = kernel.dim_y() / 2;
    auto y1 = tile.dim_y() - y0;
    auto x1 = tile.dim_x() - x0;

    auto curr = tile.inner(x0, x1, y0, y1).begin();
    auto p_curr = prev_tile.inner(x0, x1, y0, y1).begin();
    auto size = tile.inner(x0, x1, y0, y1).size();

    // Parallel impl
    for (size_t tid = 0; tid < n_threads; ++tid)
    {
        // This divides the more evenly than just size / n_threads
        size_t iters = (tid + 1) * size / n_threads - tid * size / n_threads;
        using iter_2d_t = typename Tile<elem_t>::iterator_2d_t;
        iter_2d_t end(curr);
        end += iters;

        futures.push_back(std::async(
            std::launch::async,
            [tid, &kernel](auto it, auto end, auto p_it) {
                // Run the kernel for a subset of the tile
                for (; it != end; ++it, ++p_it)
                {
                    kernel(p_it, it);
                }
            },
            curr, end, p_curr));

        curr += iters;
        p_curr += iters;
    }

    for (auto& fut : futures)
    {
        fut.get();    // Wait for all threads to finish
    }
}

void save_to_bitmap(Tile<float> const& tile, const std::string& filename)
{
    bmp::Bitmap bitmap(tile.dim_x(), tile.dim_y());
    auto idx = 0;
    std::for_each(
        tile.inner().begin(), tile.inner().end(), [&](const float& value) {
            // Assuming value is in range [0, 255] for grayscale
            auto px_val =
                static_cast<std::uint8_t>(std::clamp(value, 0.0f, 255.0f));
            bmp::Pixel pixel(px_val, px_val, px_val);    // Grayscale pixel
            bitmap.get(idx % tile.dim_x(), idx / tile.dim_x()) = pixel;
            ++idx;
        });
    bitmap.save(filename);
    std::cout << "Bitmap saved to " << filename << std::endl;
}

template <typename TileType>
void blur_kernel_remote(hpx::id_type tile_id, hpx::id_type prev_tile_id)
{
    std::cout << "Loc: " << hpx::find_here() << "Tile: " << tile_id
              << ", prev_tile: " << prev_tile_id << std::endl;

    using iter_2d_t = typename TileType::iterator_2d_t;

    auto blur_kernel = Kernel(3, 3,    // 3x3 kernel
        [](iter_2d_t const& prev, iter_2d_t& curr) -> void {
            using elem_t = typename iter_2d_t::value_type;
            // Blur kernel example
            auto sum = prev.get() + prev.get(-1, -1) + prev.get(-1, 0) +
                prev.get(-1, 1) + prev.get(0, -1) + prev.get(0, 0) +
                prev.get(0, 1) + prev.get(1, -1) + prev.get(1, 0) +
                prev.get(1, 1);

            curr.get() = sum / elem_t(9);    // Average the sum
        });

    // Get local tile instances from hpx::id_type
    auto tile_ptr = hpx::get_ptr<TileType>(tile_id).get();
    auto prev_tile_ptr = hpx::get_ptr<TileType>(prev_tile_id).get();

    update_tile_inner_seq(*tile_ptr, *prev_tile_ptr, blur_kernel);
}

HPX_PLAIN_ACTION(blur_kernel_remote<Tile<elem_t>>, blur_kernel_remote_action);

template <typename TileType>
TileType::inner_2d_tile_t get_ghost_cells_view(TileType& t, int dx, int dy)
{
    assert(((dx == -1 || dx == 1) && dy == 0) ||
        ((dy == -1 || dy == 1) && dx == 0));

    using inner_view_t = typename TileType::inner_2d_tile_t;
    inner_view_t view;
    if (dx == -1)    // Left slice
        view = t.view(0, t.dim_x(), 0, t.dim_y() + t.pad_y());
    if (dx == 1)    // Right slice
        view =
            t.view(t.dim_x(), t.dim_x() + t.pad_x(), 0, t.dim_y() + t.pad_y());
    if (dy == -1)    // Upper slice
        view = t.view(0, t.dim_x() + t.pad_x(), 0, t.dim_y());
    if (dy == 1)    // Lower slice
        view =
            t.view(0, t.dim_x() + t.pad_x(), t.dim_y(), t.dim_y() + t.pad_y());

    return view;
}

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

    if (dx != 0)
    {
        to_send.reserve(curr_tile_ptr->dim_y());
    }
    else if (dy != 0)
    {
        to_send.reserve(curr_tile_ptr->dim_x());
    }

    // Now copy the elements to the vector
    for (auto it = inner_view.begin(); it != inner_view.end(); ++it)
    {
        to_send.push_back(it.get());
    }

    // Send the data to the neighbor tile
    hpx::async(receive_ghost_elements_action{},
        hpx::colocated(neighbor_tile_id), neighbor_tile_id, to_send, dx, dy);
}

HPX_PLAIN_ACTION(copy_ghost_elements<Tile<elem_t>>, copy_ghost_elements_action);


template <typename TileType>
hpx::future<void> emit_ghost_elements(TileType& world, typename TileType::iterator_2d_t it){
    auto update_direction = [](auto it, int dx, int dy) {
        hpx::id_type curr_id = it->t_id;
        hpx::id_type neighbor_id = it.get(dx, dy).t_id;
        return hpx::async(copy_ghost_elements_action{},
            hpx::colocated(curr_id), curr_id, neighbor_id, dx, dy);
    };

    std::vector<hpx::shared_future<void>> comm;

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

template <typename TileType>
hpx::future<void> neighbors_ready(
    TileType& world, typename TileType::iterator_2d_t it)
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

struct RemoteTile {
    hpx::id_type t_id;
    hpx::id_type buf_t_id;
    hpx::shared_future<void> fut;
};

int hpx_main()
{
    // A 2d grid. Each grid element consists of two tiles (main and buffer)
    // A future is also used to synchronize the computation
    size_t world_dim_x = 8;
    size_t world_dim_y = 8;
    Tile<RemoteTile> world(world_dim_x, world_dim_y, 0, 0);

    // Initialize the tiles, distributing them across localities
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    size_t n_tiles = world_dim_x * world_dim_y;
    size_t idx = 0;
    for (auto& [id, buff_id, _] : world)
    {
        // Assign each tile to a locality
        size_t loc_idx = idx * localities.size() / n_tiles;
        auto loc = localities[loc_idx];
        id = hpx::new_<Tile<elem_t>>(loc, 1000, 1000, 1, 1).get();
        buff_id = hpx::new_<Tile<elem_t>>(loc, 1000, 1000, 1, 1).get();
        ++idx;
    }

    // Communicate ghost elements between tiles
    for (auto it = world.begin(); it != world.end(); ++it)
    {
        it->fut = emit_ghost_elements(world, it);
    }

    // Now we can run the blur kernel on each tile
    std::vector<hpx::future<void>> blur_futures;
    for (auto it = world.begin(); it != world.end(); ++it)
    {
        // All neighbors must have finished communicating for this tile
        // to proceed
        auto f1 = neighbors_ready(world, it);
        
        auto f2 = f1.then([it](hpx::future<void> /**/) {
            // This will run after all communication is done
            // Schedule the blur kernel on the locality where the tile is located
            hpx::async(blur_kernel_remote_action{}, hpx::colocated(it->t_id),
            it->t_id, it->buf_t_id);
        });

        blur_futures.push_back(std::move(f2));
    }
    // Wait for all blur kernels to finish
    hpx::wait_all(blur_futures);

    return hpx::finalize();    // Shutdown HPX runtime
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}