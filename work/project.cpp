
#include <algorithm>
#include <cassert>
#include <cstddef>    // for size_t
#include <future>
#include <iostream>
#include <vector>

#include "tile.hpp"
#include "utils/BitmapPlusPlus.hpp"
#include "utils/timer.hpp"

class int_sequence
{
};

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

// update_tile_outer(){
//     // Outer tiles are  the ones that depend on ghost elements
// }

// fetch_ghost_elements(){
//     // Ghost elements are fetched from other localities
// }

// update_tile(){
//   fut = fetch_ghost_elements();
//   update_tiles_inner();
//   update_tiles_outer();
// }

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

int main()
{
    using elem_t = float;
    using tile_t = Tile<float>;
    using iter_2d_t = typename tile_t::iterator_2d_t;

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

    tile_t tile(1000, 1000, 1, 1);
    tile_t prev_tile(tile);    // Buffer tile

    // Initialize tiles with random values
    std::generate(
        tile.begin(), tile.end(), []() { return elem_t(rand() % 100); });
    std::generate(prev_tile.begin(), prev_tile.end(),
        []() { return elem_t(rand() % 100); });

    save_to_bitmap(tile, "input.bmp");

    ps::timer_run("Sequential Tile Update",
        [&]() { update_tile_inner_seq(tile, prev_tile, blur_kernel); });

    ps::timer_run("Parallel Tile Update",
        [&]() { update_tile_inner_par(tile, prev_tile, blur_kernel); });

    save_to_bitmap(tile, "output_seq.bmp");

    return 0;
}