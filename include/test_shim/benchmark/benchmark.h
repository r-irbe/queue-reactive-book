#pragma once

// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace benchmark {

template <typename T>
inline void DoNotOptimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

inline void ClobberMemory() {
    asm volatile("" : : : "memory");
}

struct Counter {
    double value;
    enum Flags { kDefaults = 0, kIsRate = 1, kAvgIterations = 2 };
    Flags flags;
    Counter(double v = 0.0, Flags f = kDefaults) : value(v), flags(f) {}
    operator double() const { return value; }
};

class State {
public:
    explicit State(std::vector<int64_t> ranges = {}, int max_iters = 50)
        : ranges_(std::move(ranges)), max_iters_(max_iters) {
        if (ranges_.empty()) ranges_.push_back(0);
    }

    int64_t range(std::size_t idx = 0) const noexcept {
        return idx < ranges_.size() ? ranges_[idx] : ranges_[0];
    }

    void PauseTiming() {
        if (running_) {
            accum_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - start_).count();
            running_ = false;
        }
    }

    void ResumeTiming() {
        if (!running_) {
            start_ = std::chrono::high_resolution_clock::now();
            running_ = true;
        }
    }

    void SetItemsProcessed(int64_t n) noexcept { items_processed_ = n; }
    int64_t iterations() const noexcept { return iters_; }

    std::map<std::string, Counter> counters;

    struct Iterator {
        State* state;
        int current;

        bool operator!=(const Iterator& other) const noexcept {
            return current != other.current;
        }

        Iterator& operator++() {
            ++current;
            ++state->iters_;
            return *this;
        }

        int operator*() const noexcept { return current; }
    };

    Iterator begin() {
        start_ = std::chrono::high_resolution_clock::now();
        running_ = true;
        return Iterator{this, 0};
    }

    Iterator end() const noexcept {
        return Iterator{const_cast<State*>(this), max_iters_};
    }

    int64_t total_time_ns() {
        PauseTiming();
        return accum_ns_;
    }

    int64_t items_processed() const noexcept { return items_processed_; }

private:
    std::vector<int64_t> ranges_;
    int max_iters_;
    int iters_ = 0;
    int64_t items_processed_ = 0;
    int64_t accum_ns_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_{};
    bool running_ = false;
};

using BenchmarkFn = void(*)(State&);

struct BenchmarkTarget {
    std::string name;
    BenchmarkFn fn;
    std::vector<std::vector<int64_t>> arg_lists;

    BenchmarkTarget* Arg(int64_t a) {
        arg_lists.push_back({a});
        return this;
    }

    BenchmarkTarget* Args(const std::vector<int64_t>& list) {
        arg_lists.push_back(list);
        return this;
    }

    BenchmarkTarget* Range(int64_t start, int64_t end) {
        for (int64_t i = start; i <= end; i *= 4) {
            arg_lists.push_back({i});
        }
        return this;
    }

    BenchmarkTarget* Unit(int) { return this; }
};

inline std::vector<std::unique_ptr<BenchmarkTarget>>& get_bench_registry() {
    static std::vector<std::unique_ptr<BenchmarkTarget>> reg;
    return reg;
}

inline BenchmarkTarget* RegisterBenchmark(const char* name, BenchmarkFn fn) {
    auto t = std::make_unique<BenchmarkTarget>(BenchmarkTarget{name, fn, {}});
    auto* ptr = t.get();
    get_bench_registry().push_back(std::move(t));
    return ptr;
}

struct BenchmarkRunRecord {
    std::string name;
    int64_t iterations = 0;
    double real_time_ns = 0.0;
    double cpu_time_ns = 0.0;
    double items_per_second = 0.0;
    std::map<std::string, double> counters;
};

inline int RunSpecifiedBenchmarks(int argc = 0, char* argv[] = nullptr) {
    std::string filter;
    std::string out_path;
    for (int i = 1; i < argc && argv != nullptr; ++i) {
        const char* arg = argv[i];
        if (std::strncmp(arg, "--benchmark_filter=", 19) == 0) {
            filter = arg + 19;
        } else if (std::strncmp(arg, "--benchmark_out=", 16) == 0) {
            out_path = arg + 16;
        }
    }

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(42) << "Benchmark"
              << std::right << std::setw(14) << "Time (ns)"
              << std::setw(12) << "Iterations"
              << std::setw(16) << "Throughput" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<BenchmarkRunRecord> records;

    for (const auto& target : get_bench_registry()) {
        std::vector<std::vector<int64_t>> runs = target->arg_lists;
        if (runs.empty()) runs.push_back({0});

        for (const auto& args : runs) {
            std::string label = target->name;
            if (!target->arg_lists.empty()) {
                label += "/";
                for (std::size_t k = 0; k < args.size(); ++k) {
                    if (k > 0) label += "/";
                    label += std::to_string(args[k]);
                }
            }
            if (!filter.empty() && label.find(filter) == std::string::npos &&
                target->name.find(filter) == std::string::npos) {
                continue;
            }

            State state(args, 25);
            target->fn(state);

            const int64_t ns = state.total_time_ns();
            const int64_t iters = std::max<int64_t>(1, state.iterations());
            const double per_iter_ns = static_cast<double>(ns) / static_cast<double>(iters);

            std::cout << std::left << std::setw(42) << label
                      << std::right << std::setw(14) << std::fixed << std::setprecision(1) << per_iter_ns
                      << std::setw(12) << iters;

            double items_per_sec = 0.0;
            if (state.items_processed() > 0 && ns > 0) {
                items_per_sec = (static_cast<double>(state.items_processed()) / static_cast<double>(ns)) * 1e9;
                std::cout << std::setw(13) << std::fixed << std::setprecision(2)
                          << (items_per_sec / 1e6) << "M/s";
            } else {
                std::cout << std::setw(16) << "-";
            }

            BenchmarkRunRecord rec{};
            rec.name = label;
            rec.iterations = iters;
            rec.real_time_ns = per_iter_ns;
            rec.cpu_time_ns = per_iter_ns;
            rec.items_per_second = items_per_sec;

            for (const auto& [k, c] : state.counters) {
                double val = c.value;
                if ((c.flags & Counter::kAvgIterations) != 0) {
                    val /= static_cast<double>(iters);
                }
                rec.counters[k] = val;
                std::cout << "  " << k << "=" << std::fixed << std::setprecision(1) << val;
            }
            std::cout << "\n";
            records.push_back(std::move(rec));
        }
    }
    std::cout << "--------------------------------------------------------------------------------\n";

    if (!out_path.empty()) {
        std::ofstream ofs(out_path);
        if (ofs.is_open()) {
            ofs << "{\n"
                << "  \"context\": {\n"
                << "    \"executable\": \"sim_bench\",\n"
                << "    \"num_benchmarks\": " << records.size() << "\n"
                << "  },\n"
                << "  \"benchmarks\": [\n";
            for (std::size_t i = 0; i < records.size(); ++i) {
                const auto& r = records[i];
                ofs << "    {\n"
                    << "      \"name\": \"" << r.name << "\",\n"
                    << "      \"run_name\": \"" << r.name << "\",\n"
                    << "      \"run_type\": \"iteration\",\n"
                    << "      \"iterations\": " << r.iterations << ",\n"
                    << "      \"real_time\": " << std::fixed << std::setprecision(4) << r.real_time_ns << ",\n"
                    << "      \"cpu_time\": " << std::fixed << std::setprecision(4) << r.cpu_time_ns << ",\n"
                    << "      \"time_unit\": \"ns\",\n"
                    << "      \"items_per_second\": " << std::fixed << std::setprecision(2) << r.items_per_second;
                for (const auto& [ck, cv] : r.counters) {
                    ofs << ",\n      \"" << ck << "\": " << std::fixed << std::setprecision(4) << cv;
                }
                ofs << "\n    }" << (i + 1 < records.size() ? "," : "") << "\n";
            }
            ofs << "  ]\n}\n";
        }
    }

    return 0;
}

} // namespace benchmark

#define BENCHMARK(fn) \
    static ::benchmark::BenchmarkTarget* benchmark_reg_##fn = \
        ::benchmark::RegisterBenchmark(#fn, fn)

#define BENCHMARK_MAIN() \
    int main(int argc, char* argv[]) { \
        return ::benchmark::RunSpecifiedBenchmarks(argc, argv); \
    }
