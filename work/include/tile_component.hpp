
#include <cstddef>    // std::size_t
#include <vector>
#include <hpx/include/components.hpp>

// Tile class:
// Stores data for one (2D) tile
// Padded with padding/ghost elements (as many as needed for the kernel to operate correctly)
template <typename elem_t>
class Tile : public hpx::components::component_base<Tile<elem_t>>
{
    
public:
    using value_type = elem_t;

    // Default constructor
    Tile() = default;

    Tile(size_t dim_x, size_t dim_y, size_t kernel_x, size_t kernel_y)
      : dim_x_(dim_x)
      , dim_y_(dim_y)
      , pad_x_(kernel_x)
      , pad_y_(kernel_y)
    {
        data_.resize((dim_x_ + 2 * pad_x_) * (dim_y_ + 2 * pad_y_));
    }

    size_t dim_x() const
    {
        return dim_x_;
    }
    size_t dim_y() const
    {
        return dim_y_;
    }
    size_t pad_x() const
    {
        return pad_x_;
    }
    size_t pad_y() const
    {
        return pad_y_;
    }

    // Iterate over 2D coordinates (can be a subset of a tile)
    // Template to allow for both const and non-const references to Tile
    template <typename tile_ptr_t>
    class Iterator_2d
    {
    public:
        template <typename T>
        using undress_t = std::remove_pointer_t<std::remove_reference_t<T>>;

        using value_type = typename undress_t<tile_ptr_t>::value_type;
        using difference_type = std::ptrdiff_t;

        static constexpr bool is_const_v =
            std::is_const_v<std::remove_pointer_t<tile_ptr_t>>;
        using reference_type =
            std::conditional_t<is_const_v, const value_type&, value_type&>;

        Iterator_2d(tile_ptr_t tile, size_t x, size_t y, size_t x_min,
            size_t x_max, size_t y_min, size_t y_max)
          : tile_(tile)
          , x_(x)
          , y_(y)
          , x_min_(x_min)
          , x_max_(x_max)
          , y_min_(y_min)
          , y_max_(y_max)
        {
        }

        bool operator!=(const Iterator_2d& other) const
        {
            return (tile_ != other.tile_) || x_ != other.x_ || y_ != other.y_;
        }
        
        Iterator_2d& operator++()
        {    // Pre-increment
            if (++x_ >= x_max_)
            {
                x_ = x_min_;
                ++y_;
            }
            return *this;
        }

        // Random access
        Iterator_2d& operator+(size_t offset)
        {
            size_t new_x = x_ + offset;
            size_t new_y = y_;
            if (new_x >= x_max_)
            {
                new_y += new_x / (x_max_ - x_min_);
                new_x = new_x % (x_max_ - x_min_);
            }
            return Iterator_2d(
                tile_, new_x, new_y, x_min_, x_max_, y_min_, y_max_);
        }

        difference_type operator-(const Iterator_2d& other) const
        {
            assert(tile_ == other.tile_);
            return (y_ - other.y_) * (x_max_ - x_min_) + (x_ - other.x_);
        }

        Iterator_2d& operator+=(size_t offset)
        {
            size_t new_x = x_ + offset;
            size_t new_y = y_;
            if (new_x >= x_max_)
            {
                new_y += new_x / (x_max_ - x_min_);
                new_x = new_x % (x_max_ - x_min_);
            }
            x_ = new_x;
            y_ = new_y;
            return *this;
        }

        // Access element, optionally with offset. Element may be out of bounds
        // of the Iterator_2d, but must be within the underlying tile's data range.
        reference_type get(size_t offset_x = 0, size_t offset_y = 0) const
        {
            size_t x = x_ + offset_x;
            size_t y = y_ + offset_y;
            assert(tile_ != nullptr);
            assert(x >= 0 && x < tile_->dim_x() + 2 * tile_->pad_x() && y >= 0 &&
                y < tile_->dim_y() + 2 * tile_->pad_y());
            auto idx = y * (tile_->dim_x() + 2 * tile_->pad_x()) + x;
            reference_type data_ref(tile_->data_[idx]);
            return data_ref;
        }

        reference_type operator*() const
        {
            return get();
        }

        size_t x() const
        {
            return x_;
        }
        size_t y() const
        {
            return y_;
        }

        tile_ptr_t tile() const
        {
            return tile_;
        }

    private:
        tile_ptr_t tile_ = nullptr; // Pointer to the tile data
        size_t x_, y_;            // Current position in the tile
        size_t x_min_, x_max_;    // Bounds for x iteration
        size_t y_min_, y_max_;    // Bounds for y iteration
    };

    // There are 2 ways we might want to iterate over the tile:
    // 1. Over the entire tile, including padding/ghost elements
    // 2. Over a subset of the tile, defined by a rectangle (x_min, x_max, y_min, y_max)
    //    This is useful either when applying kernels (to avoid out-of-bounds access), or
    //    to avoid processing padding/ghost elements (for example, if ghost elements have
    //    not been fetched yet).

    // Subset (rectangle) view
    template <typename tile_ptr_t>
    class View_2d
    {
    public:
        using iter_2d_t = Iterator_2d<tile_ptr_t>;

        // Default constructor
        View_2d() = default;

        // Constructor for the inner tile, takes a tile reference and rectangle bounds
        View_2d(tile_ptr_t tile, size_t x_min, size_t x_max, size_t y_min,
            size_t y_max)
          : tile_(tile)
          , x_min_(x_min)
          , x_max_(x_max)
          , y_min_(y_min)
          , y_max_(y_max)
        {
            assert(tile_ != nullptr);
            assert(x_min_ < x_max_ && y_min_ < y_max_);
            assert(x_min_ >= 0 && x_max_ <= tile_->dim_x() + 2 * tile_->pad_x());
            assert(y_min_ >= 0 && y_max_ <= tile_->dim_y() + 2 * tile_->pad_y());
        }

        iter_2d_t begin()
        {
            assert(tile_ != nullptr);
            return iter_2d_t(
                tile_, x_min_, y_min_, x_min_, x_max_, y_min_, y_max_);
        }
        iter_2d_t end()
        {
            assert(tile_ != nullptr);
            return iter_2d_t(
                tile_, x_min_, y_max_, x_min_, x_max_, y_min_, y_max_);
        }

        size_t size_x() const
        {
            return x_max_ - x_min_;
        }

        size_t size_y() const
        {
            return y_max_ - y_min_;
        }

        View_2d subview(
            size_t x_min, size_t x_max, size_t y_min, size_t y_max) const
        {
            assert(x_min >= 0 && x_max <= x_max_ - x_min_);
            assert(y_min >= 0 && y_max <= y_max_ - y_min_);
            return View_2d(tile_, x_min + x_min_, x_max + x_min_,
                y_min + y_min_, y_max + y_min_);
        }

        size_t size() const
        {
            return (x_max_ - x_min_) * (y_max_ - y_min_);
        }

    private:
        tile_ptr_t tile_ = nullptr; // Pointer to the tile data
        size_t x_min_, x_max_;
        size_t y_min_, y_max_;
    };

    using iterator_2d_t = Iterator_2d<Tile*>;
    using const_iterator_2d_t = Iterator_2d<Tile const*>;

    // Full tile, does not skip over padding
    iterator_2d_t begin()
    {
        return iterator_2d_t(
            this, 0, 0, 0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_);
    }
    iterator_2d_t end()
    {
        return iterator_2d_t(this, 0, dim_y_ + 2 * pad_y_, 0,
            dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_);
    }

    // const begin and end
    const_iterator_2d_t begin() const
    {
        return const_iterator_2d_t(
            this, 0, 0, 0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_);
    }
    const_iterator_2d_t end() const
    {
        return const_iterator_2d_t(this, 0, dim_y_ + 2 * pad_y_, 0,
            dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_);
    }

    using inner_2d_tile_t = View_2d<Tile*>;
    using const_inner_2d_tile_t = View_2d<Tile const*>;

    // non-const tile view
    // Does not skip over padding
    inner_2d_tile_t view(
        size_t x_min, size_t x_max, size_t y_min, size_t y_max)
    {
        return inner_2d_tile_t(this, x_min, x_max, y_min, y_max);
    }

    // const view
    // Does not skip over padding
    const_inner_2d_tile_t view(
        size_t x_min, size_t x_max, size_t y_min, size_t y_max) const
    {
        return const_inner_2d_tile_t(this, x_min, x_max, y_min, y_max);
    }

    // non-const inner tile
    // Skips over padding
    inner_2d_tile_t inner(
        size_t x_min, size_t x_max, size_t y_min, size_t y_max)
    {
        return inner_2d_tile_t(this, x_min + pad_x_, x_max + pad_x_,
            y_min + pad_y_, y_max + pad_y_);
    }

    // non-const inner tile
    // Skips over padding    
    inner_2d_tile_t inner()
    {
        return inner(0, dim_x_, 0, dim_y_);
    }

    // const inner tile
    // Skips over padding
    const_inner_2d_tile_t inner(
        size_t x_min, size_t x_max, size_t y_min, size_t y_max) const
    {
        return const_inner_2d_tile_t(this, x_min + pad_x_, x_max + pad_x_,
            y_min + pad_y_, y_max + pad_y_);
    }

    // const inner tile
    // Skips over padding
    const_inner_2d_tile_t inner() const
    {
        return inner(0, dim_x_, 0, dim_y_);
    }

    size_t size() const
    {
        return data_.size();
    }
    
    // Checks if the coordinates are within the inner tile's data range
    bool is_inner(size_t x, size_t y) const
    {
        return (x >= pad_x_ && x < dim_x_ + pad_x_ &&
                y >= pad_y_ && y < dim_y_ + pad_y_);
    }

    // Checks if the coordinates are within the tile's data range
    bool is_valid(size_t x, size_t y) const
    {
        return (x >= 0 && x < dim_x_ + 2 * pad_x_ &&
                y >= 0 && y < dim_y_ + 2 * pad_y_);
    }

private:
    size_t dim_x_, dim_y_;
    size_t pad_x_, pad_y_;
    std::vector<elem_t> data_;
};

