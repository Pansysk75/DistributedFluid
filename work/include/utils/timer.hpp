#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <numeric>

namespace ps {

    class timer_scoped
    {
    public:
        timer_scoped(std::string name)
          : name_(name)
          , start_(std::chrono::high_resolution_clock::now())
        {
        }

        ~timer_scoped()
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end - start_);

            double duration_seconds =
                static_cast<double>(duration.count()) / 1e6;
            std::cout << "[" << name_ << "]: " << duration_seconds << " s"
                      << std::endl;
        }

    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    template <typename Func, typename... Args>
    auto timer_run(std::string name, Func&& func, Args&&... args)
    {
        timer_scoped timer(name);
        return func(std::forward<Args>(args)...);
    }

    // Stores a map<string, vec<time_t>>
    // Prints the average time for each timer
    class timer_registry
    {
    public:
        enum class out_mode
        {
            average,
            median,
            sum,
            array
        };

        enum class out_format
        {
            json,
            text
        };

        timer_registry(
            out_mode m = out_mode::average, out_format f = out_format::text)
          : out_mode_(m)
          , out_format_(f)
        {
        }

    private:
        void add_timer(std::string name, std::chrono::microseconds duration)
        {
            timers_[name].push_back(duration);
        }

    public:
        void print()
        {
            if (out_format_ == out_format::json)
                std::cout << "{" << std::endl;

            size_t idx = 0;
            for (const auto& [name, durations] : timers_)
            {
                if (durations.empty())
                    continue;

                std::string str;

                if (out_mode_ == out_mode::array)
                {
                    str = "[";
                    for (size_t i = 0; i < durations.size(); ++i)
                    {
                        str += std::to_string(
                            static_cast<double>(durations[i].count()) / 1e6);
                        if (i < durations.size() - 1)
                            str += ", ";
                    }
                    str += "]";
                }
                else if (out_mode_ == out_mode::median)
                {
                    auto sorted = durations;
                    std::sort(sorted.begin(), sorted.end());
                    auto median = sorted[sorted.size() / 2];
                    str = std::to_string(
                        static_cast<double>(median.count()) / 1e6);
                }
                else
                {
                    auto sum_ms = std::accumulate(durations.begin(),
                        durations.end(), std::chrono::microseconds(0));
                    double sum = static_cast<double>(sum_ms.count()) / 1e6;
                    if (out_mode_ == out_mode::average)
                    {
                        str = std::to_string(sum / durations.size());
                    }
                    else if (out_mode_ == out_mode::sum)
                    {
                        str = std::to_string(sum);
                    }
                }

                // Now print according to the format
                if (out_format_ == out_format::json)
                {
                    std::cout << "  \"" << name << "\": " << str;
                    bool is_last = (idx == timers_.size() - 1);
                    if (!is_last)
                        std::cout << ",";
                    std::cout << std::endl;
                }
                else
                {
                    std::cout << "[" << name << "]: " << str << std::endl;
                }

                ++idx;
            }

            if (out_format_ == out_format::json)
                std::cout << "}" << std::endl;
        }

        class timer_scoped
        {
        public:
            timer_scoped(timer_registry& registry, std::string name)
              : registry_(registry)
              , name_(std::move(name))
              , start_(std::chrono::high_resolution_clock::now())
            {
            }

            ~timer_scoped()
            {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start_);
                registry_.add_timer(name_, duration);
            }

        private:
            timer_registry& registry_;
            std::string name_;
            std::chrono::high_resolution_clock::time_point start_;
        };

    public:
        timer_scoped time_scope(std::string name)
        {
            return timer_scoped(*this, std::move(name));
        }

        ~timer_registry()
        {
            print();
        }

    private:
        std::map<std::string, std::vector<std::chrono::microseconds>>
            timers_;
        out_mode out_mode_;
        out_format out_format_;
    };

}    // namespace ps
