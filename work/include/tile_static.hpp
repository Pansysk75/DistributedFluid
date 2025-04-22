
#include <cstddef>    // std::size_t
#include <vector>
#include <cassert>

// Tile class:
// Stores data for one (2D) tile
// Padded with padding/ghost elements (as many as needed for the kernel to operate correctly)
template <typename elem_t, 
unsigned int dim_x_, unsigned int dim_y_,
unsigned int pad_x_, unsigned int pad_y_>
class Tile
{
    
public:
    using value_type = elem_t;

    // Default constructor
    Tile()
    {
        data_.resize((dim_x_ + 2 * pad_x_) * (dim_y_ + 2 * pad_y_));
    }

    static constexpr size_t dim_x()
    {
        return dim_x_;
    }
    static constexpr size_t dim_y()
    {
        return dim_y_;
    }
    static constexpr size_t pad_x()
    {
        return pad_x_;
    }
    static constexpr size_t pad_y()
    {
        return pad_y_;
    }

    // Iterate over 2D coordinates (can be a subset of a tile)
    // Template to allow for both const and non-const references to Tile
    template <typename tile_ptr_t, 
              unsigned int x_min_, unsigned int x_max_,
                unsigned int y_min_, unsigned int y_max_>
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

        Iterator_2d(tile_ptr_t tile, size_t x, size_t y)
          : tile_(tile)
          , x_(x)
          , y_(y)
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
        reference_type get(int offset_x = 0, int offset_y = 0) const
        {
            assert(tile_ != nullptr);
            size_t x = x_ + offset_x;
            size_t y = y_ + offset_y;
            assert(tile_ != nullptr);
            assert(x >= 0 && x < tile_->dim_x() + 2 * tile_->pad_x() && y >= 0 &&
                y < tile_->dim_y() + 2 * tile_->pad_y());
            auto idx = y * (tile_->dim_x() + 2 * tile_->pad_x()) + x;
            return tile_->data_[idx];
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
    };

    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    using iterator_2d_t = Iterator_2d<Tile*, x_min_, x_max_, y_min_, y_max_>;

    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    using const_iterator_2d_t = Iterator_2d<Tile const*, x_min_, x_max_,
        y_min_, y_max_>;

    // Full tile, does not skip over padding
    auto begin()
    {
        return iterator_2d_t<0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_>(
            this, 0, 0);
    }
    auto end()
    {
        return iterator_2d_t<0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_>(
            this, 0, dim_y_ + 2 * pad_y_);
    }

    // const begin and end
    auto begin() const
    {
        return const_iterator_2d_t<0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_>(
            this, 0, 0);
    }
    auto end() const
    {
        return const_iterator_2d_t<0, dim_x_ + 2 * pad_x_, 0, dim_y_ + 2 * pad_y_>(
            this, 0, dim_y_ + 2 * pad_y_);
    }

    // Subset (rectangle) view
    template <typename tile_ptr_t, 
              unsigned int x_min_, unsigned int x_max_,
                unsigned int y_min_, unsigned int y_max_>
    class View_2d
    {
    public:

        // Default constructor
        View_2d() = default;

        // Constructor for the inner tile, takes a tile reference and rectangle bounds
        View_2d(tile_ptr_t tile)
          : tile_(tile)
        {
            assert(tile_ != nullptr);
            assert(x_min_ < x_max_ && y_min_ < y_max_);
            assert(x_min_ >= 0 && x_max_ <= tile_->dim_x() + 2 * tile_->pad_x());
            assert(y_min_ >= 0 && y_max_ <= tile_->dim_y() + 2 * tile_->pad_y());
        }

        auto begin()
        {
            assert(tile_ != nullptr);
            return Iterator_2d<tile_ptr_t, x_min_, x_max_, y_min_, y_max_>(
                tile_, x_min_, y_min_);
        }
        auto end()
        {
            assert(tile_ != nullptr);
            return Iterator_2d<tile_ptr_t, x_min_, x_max_, y_min_, y_max_>(
                tile_, x_min_, y_max_);
        }

        constexpr size_t size_x() const
        {
            return x_max_ - x_min_;
        }

        constexpr size_t size_y() const
        {
            return y_max_ - y_min_;
        }

        template<unsigned int x_min, unsigned int x_max,
            unsigned int y_min, unsigned int y_max>
        auto subview() const
        {
            static_assert(x_min >= 0 && x_max <= x_max_ - x_min_);
            static_assert(y_min >= 0 && y_max <= y_max_ - y_min_);
            return View_2d<tile_ptr_t, 
                x_min + x_min_, x_max + x_min_, 
                y_min + y_min_, y_max + y_min_>(tile_);
        }

        size_t size() const
        {
            return (x_max_ - x_min_) * (y_max_ - y_min_);
        }

    private:
        tile_ptr_t tile_ = nullptr; // Pointer to the tile data
    };

    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    using inner_2d_tile_t = View_2d<Tile*, x_min_, x_max_, y_min_, y_max_>;

    template<unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    using const_inner_2d_tile_t = View_2d<Tile const*, x_min_, x_max_, y_min_, y_max_>;

    // non-const tile view
    // Does not skip over padding
    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    auto view()
    {
        return inner_2d_tile_t<x_min_, x_max_, y_min_, y_max_>(this);
    }

    // const view
    // Does not skip over padding
    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    auto view() const
    {
        return const_inner_2d_tile_t<x_min_, x_max_, y_min_, y_max_>(this);
    }

    // non-const inner tile
    // Skips over padding
    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    auto inner()
    {
        return inner_2d_tile_t<x_min_ + pad_x_, x_max_ + pad_x_,
            y_min_ + pad_y_, y_max_ + pad_y_>(this);
    }

    // non-const inner tile
    // Skips over padding
    auto inner()
    {
        return inner<0, dim_x_, 0, dim_y_>();
    }

    // const inner tile
    // Skips over padding
    template <unsigned int x_min_, unsigned int x_max_,
        unsigned int y_min_, unsigned int y_max_>
    auto inner() const
    {
        return const_inner_2d_tile_t<x_min_ + pad_x_, x_max_ + pad_x_,
            y_min_ + pad_y_, y_max_ + pad_y_>(this);
    }

    // const inner tile
    // Skips over padding
    auto inner() const
    {
        return inner<0, dim_x_, 0, dim_y_>();
    }

    constexpr size_t size() const
    {
        return (dim_x_ + 2 * pad_x_) * (dim_y_ + 2 * pad_y_);
    }

    // Checks if the coordinates are within the inner tile's data range
    constexpr bool is_inner(size_t x, size_t y) const
    {
        return (x >= pad_x_ && x < dim_x_ + pad_x_ &&
                y >= pad_y_ && y < dim_y_ + pad_y_);
    }

    // Checks if the coordinates are within the tile's data range
    constexpr bool is_valid(size_t x, size_t y) const
    {
        return (x >= 0 && x < dim_x_ + 2 * pad_x_ &&
                y >= 0 && y < dim_y_ + 2 * pad_y_);
    }

private:
    std::vector<elem_t> data_;
};

// Iterator to traverse a given 2d view in sub-views (blocks)
// Dereferencing returns a 2d view
// incrementing moves to the next block
// This is useful for cache-friendly access patterns
template <typename view_2d_t>
class blocked_2d_view_iterator
{
public:
    using value_type = view_2d_t;
    using difference_type = std::ptrdiff_t;

    blocked_2d_view_iterator(view_2d_t view, size_t block_size_x,
        size_t block_size_y, size_t x, size_t y)
      : view_(view)
      , bl_size_x_(block_size_x)
      , bl_size_y_(block_size_y)
      , x_(x)
      , y_(y)
    {
    }

    bool operator!=(const blocked_2d_view_iterator& other) const
    {
        return (x_ != other.x_) || (y_ != other.y_);
    }

    // Pre-increment
    blocked_2d_view_iterator& operator++()
    {
        x_ += bl_size_x_;
        if (x_ > view_.size_x())
        {
            x_ = 0;
            y_ += bl_size_y_;
        }
        return *this;
    }

    blocked_2d_view_iterator& operator+=(size_t offset)
    {
        size_t blocks_per_row = (view_.size_x() + bl_size_x_ - 1) / bl_size_x_;
        x_ += (offset % blocks_per_row) * bl_size_x_;
        y_ += (offset / blocks_per_row) * bl_size_y_;
        if (x_ >= view_.size_x())
        {
            x_ = 0;
            y_ += bl_size_y_;
        }
        return *this;
    }

    difference_type operator-(const blocked_2d_view_iterator& other) const
    {
        size_t blocks_per_row = (view_.size_x() + bl_size_x_ - 1) / bl_size_x_;
        return ((x_ - other.x_) / bl_size_x_) +
            ((y_ - other.y_) / bl_size_y_) * blocks_per_row;
    }

    value_type operator*() const
    {
        size_t x_max = std::min(x_ + bl_size_x_, view_.size_x());
        size_t y_max = std::min(y_ + bl_size_y_, view_.size_y());
        return view_.subview(x_, x_max, y_, y_max);
    }

private:
    size_t bl_size_x_, bl_size_y_;
    size_t x_, y_;    // Current block position in the view
    view_2d_t view_;
};

template <typename view_2d_t>
class blocked_2d_view
{
public:
    using iterator = blocked_2d_view_iterator<view_2d_t>;

    blocked_2d_view(view_2d_t view, size_t block_size_x, size_t block_size_y)
      : view_(view)
      , bl_size_x_(block_size_x)
      , bl_size_y_(block_size_y)
    {
    }

    iterator begin()
    {
        return iterator(view_, bl_size_x_, bl_size_y_, 0, 0);
    }
    iterator end()
    {
        return iterator(view_, bl_size_x_, bl_size_y_, 0, view_.size_y());
    }
    size_t size() const
    {
        return ((view_.size_x() + bl_size_x_ - 1) / bl_size_x_) *
            ((view_.size_y() + bl_size_y_ - 1) / bl_size_y_);
    }
    size_t block_size_x() const
    {
        return bl_size_x_;
    }
    size_t block_size_y() const
    {
        return bl_size_y_;
    }
    view_2d_t view() const
    {
        return view_;
    }

private:
    view_2d_t view_;
    size_t bl_size_x_, bl_size_y_;
};
