#include <chrono>
#include <iostream>
#include <string>

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

}    // namespace ps
