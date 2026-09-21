#include <benchmark/benchmark.h>

#include "offline/veb_static_search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

// =============================================================================
// OFFLINE micro-benchmark + deterministic timing-envelope reporter for the
// cache-oblivious (van-Emde-Boas layout) static search index vs a flat
// sorted-vector baseline. This is the benchmark half of the gate-G8 closure
// evidence for `moc-offline-cache-oblivious-data-structures`.
//
// HARD BOUNDARY: this benchmark links NOTHING from the matching kernel. It uses
// only `offline/veb_static_search.hpp` and the standard library. Registered
// into the existing Google Benchmark binary (bench/main.cpp owns
// BENCHMARK_MAIN()); this translation unit only adds BENCHMARK registrations.
//
// DETERMINISTIC parts (reproducible regardless of host):
//   * The workload (keys + query stream) is built from FIXED seeds.
//   * A self-contained parity check asserts vEB results == flat baseline before
//     timing; a mismatch aborts the benchmark (the timing envelope is only
//     meaningful when parity holds).
//   * Per-query latency p50/p90/p99 are computed over the fixed workload and
//     reported as benchmark counters. Absolute nanoseconds are
//     environment-dependent; the percentile STRUCTURE is deterministic.
// =============================================================================

namespace {

constexpr std::uint64_t kKeySeed = 0xBADC0FFEE0DDF00DULL;
constexpr std::uint64_t kQuerySeed = 0x1571F0CA1A7ED0DEULL;
constexpr std::size_t kNumKeys = 1u << 16; // 65536 keys
constexpr std::size_t kNumQueries = 1u << 16;

std::vector<std::int64_t> build_keys() {
    std::mt19937_64 eng(kKeySeed);
    std::uniform_int_distribution<std::int64_t> dist(0, 1'000'000'000);
    std::vector<std::int64_t> keys;
    keys.reserve(kNumKeys);
    for (std::size_t i = 0; i < kNumKeys; ++i) keys.push_back(dist(eng));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

std::vector<std::int64_t> build_queries() {
    std::mt19937_64 eng(kQuerySeed);
    std::uniform_int_distribution<std::int64_t> dist(-10, 1'000'000'010);
    std::vector<std::int64_t> q;
    q.reserve(kNumQueries);
    for (std::size_t i = 0; i < kNumQueries; ++i) q.push_back(dist(eng));
    return q;
}

// Flat sorted-array baseline lower_bound rank.
inline std::size_t flat_rank(const std::vector<std::int64_t>& keys, std::int64_t q) {
    return static_cast<std::size_t>(
        std::lower_bound(keys.begin(), keys.end(), q) - keys.begin());
}

// Abort the benchmark if vEB does not match the flat baseline on every query.
// Keeping this in the benchmark TU makes the timing envelope self-falsifying.
void require_parity(const std::vector<std::int64_t>& keys,
                    const offline::VebStaticSearch<std::int64_t>& idx,
                    const std::vector<std::int64_t>& queries) {
    for (std::int64_t q : queries) {
        if (idx.lower_bound_rank(q) != flat_rank(keys, q)) {
            std::fprintf(stderr,
                         "FATAL: offline vEB scan-parity violated for q=%lld\n",
                         static_cast<long long>(q));
            std::abort();
        }
    }
}

void report_percentiles(benchmark::State& state, std::vector<double>& lat_ns) {
    std::sort(lat_ns.begin(), lat_ns.end());
    auto pct = [&](double p) -> double {
        if (lat_ns.empty()) return 0.0;
        const std::size_t i = static_cast<std::size_t>(
            p * static_cast<double>(lat_ns.size() - 1));
        return lat_ns[i];
    };
    state.counters["p50_ns"] = pct(0.50);
    state.counters["p90_ns"] = pct(0.90);
    state.counters["p99_ns"] = pct(0.99);
    state.counters["queries"] =
        benchmark::Counter(static_cast<double>(lat_ns.size()));
}

} // namespace

// ---- Throughput micro-benchmarks (Google Benchmark timing) -----------------

void BM_Offline_Veb_LowerBound(benchmark::State& state) {
    const auto keys = build_keys();
    const auto queries = build_queries();
    auto idx = offline::VebStaticSearch<std::int64_t>::from_sorted(keys);
    require_parity(keys, idx, queries);

    std::size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(idx.lower_bound_rank(queries[i]));
        if (++i == queries.size()) i = 0;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Offline_Veb_LowerBound);

void BM_Offline_Flat_LowerBound(benchmark::State& state) {
    const auto keys = build_keys();
    const auto queries = build_queries();

    std::size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(flat_rank(keys, queries[i]));
        if (++i == queries.size()) i = 0;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Offline_Flat_LowerBound);

// ---- Deterministic per-query latency envelope (p50/p90/p99 counters) -------
//
// These two benchmarks sweep the FIXED query workload once per iteration,
// timing each individual query, then report the percentile envelope. The
// reported counters are the deterministic-structure part of the gate evidence.

void BM_Offline_Veb_LatencyEnvelope(benchmark::State& state) {
    const auto keys = build_keys();
    const auto queries = build_queries();
    auto idx = offline::VebStaticSearch<std::int64_t>::from_sorted(keys);
    require_parity(keys, idx, queries);

    std::vector<double> lat_ns;
    lat_ns.reserve(queries.size());
    for (auto _ : state) {
        lat_ns.clear();
        for (std::int64_t q : queries) {
            const auto t0 = std::chrono::steady_clock::now();
            benchmark::DoNotOptimize(idx.lower_bound_rank(q));
            const auto t1 = std::chrono::steady_clock::now();
            lat_ns.push_back(
                std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        benchmark::ClobberMemory();
    }
    report_percentiles(state, lat_ns);
}
BENCHMARK(BM_Offline_Veb_LatencyEnvelope);

void BM_Offline_Flat_LatencyEnvelope(benchmark::State& state) {
    const auto keys = build_keys();
    const auto queries = build_queries();

    std::vector<double> lat_ns;
    lat_ns.reserve(queries.size());
    for (auto _ : state) {
        lat_ns.clear();
        for (std::int64_t q : queries) {
            const auto t0 = std::chrono::steady_clock::now();
            benchmark::DoNotOptimize(flat_rank(keys, q));
            const auto t1 = std::chrono::steady_clock::now();
            lat_ns.push_back(
                std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        benchmark::ClobberMemory();
    }
    report_percentiles(state, lat_ns);
}
BENCHMARK(BM_Offline_Flat_LatencyEnvelope);
