#include <benchmark/benchmark.h>

#include "offline/veb_static_search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// OFFLINE, KERNEL-ADJACENT *multi-threaded contention* benchmark.
//
// PURPOSE (produces evidence for `needs-parallel-contention-evidence` /
//   `COB-PARALLEL` on `moc-offline-cache-oblivious-data-structures`):
//   The single-threaded hot-path bench (bench/offline_cache_oblivious_hotpath_bench.cpp)
//   already showed a direct-indexed fixed-domain prefix-rank array dominates both
//   a flat sorted vector + std::lower_bound and the cache-oblivious vEB index by
//   ~6-40x. That bench is single-threaded and answers nothing about CONTENTION,
//   NUMA placement, or multi-thread TAIL behaviour. This translation unit closes
//   that gap with REAL multi-threaded measurements: T concurrent read-only
//   lookup threads (T in {1,2,4,8}) hammer each of the three immutable structures
//   over representative bounded ladder sizes, and we report:
//     * aggregate throughput scaling (Mops/s) vs thread count, and
//     * per-thread p50/p99/p999 latency tail under contention.
//
// HARD BOUNDARY (NEVER VIOLATED):
//   This translation unit links NOTHING from the matching kernel. It does NOT
//   #include "sim/flat_ladder_book.hpp", does NOT reference any `sim::` symbol,
//   and never touches FlatLadderBook. It depends only on the self-contained
//   header `offline/veb_static_search.hpp`, the C++ standard library, and
//   std::thread/std::atomic. (bench/main.cpp owns BENCHMARK_MAIN(); this file
//   only adds BENCHMARK registrations and is auto-globbed by the Makefile
//   BENCH_SRC wildcard.)
//
// STRUCTURES UNDER TEST (read-only, built once, shared by all threads):
//   (a) offline vEB static search index (offline/veb_static_search.hpp),
//   (b) flat sorted std::vector + std::lower_bound,
//   (c) direct-indexed fixed-domain prefix-rank array (the bounded-tick default,
//       a single O(1) clamped load).
//
// SHARED-READ / FALSE-SHARING LAYOUT NOTE:
//   All three structures are IMMUTABLE during timing: every thread performs only
//   reads, so there is NO true sharing and NO cache-line invalidation traffic
//   from the data itself (read-shared lines stay in Shared/`S` state across
//   cores). The only mutable per-thread state (a checksum accumulator and a
//   latency vector) is kept thread-LOCAL and, where shared back to the parent,
//   is stored in a 64-byte cache-line-aligned `alignas(64)` slot per thread so
//   adjacent threads never write the same line (avoids false sharing on the
//   result array). This isolates the structures' intrinsic concurrent-read
//   scalability (working-set / cache-footprint pressure) from synthetic
//   write-contention artefacts. The fixed-domain array's tiny hot working set
//   (the clamped prefix load touches one cache line per query, and the hot 1/32
//   window fits in L1/L2) is exactly what we expect to scale flattest.
//
// SEMANTIC CONTRACT / SELF-ABORT:
//   For every query q in [0, D], all three structures MUST return the identical
//   lower_bound_rank. A parity check runs before ANY timing; ANY mismatch calls
//   std::abort() so the bench can never report contention numbers on a broken
//   structure.
//
// DETERMINISM / HONESTY:
//   Keys + queries derive from FIXED seeds, so the workload is reproducible on
//   any host. Aggregate throughput (ops / wall-time of the whole burst) is the
//   RELIABLE cross-structure / cross-thread-count signal. The per-op p50/p99/p999
//   latency tail is collected with a per-op std::chrono read whose fixed overhead
//   dominates these sub-100ns operations, so those ABSOLUTE nanoseconds OVERSTATE
//   cost; their value is the deterministic tail STRUCTURE and its RELATIVE change
//   as contention grows, not an isolated latency claim. This is a commodity
//   (non-NUMA-isolated, possibly single-socket) host: we report what scaling and
//   tail we measured and explicitly mark NUMA-placement as not separately
//   pinned (no libnuma / no core affinity on this platform).
// =============================================================================

namespace {

using Key = std::int64_t;

constexpr std::uint64_t kKeySeed = 0xC0FFEE17ACEB00C1ULL;
constexpr std::uint64_t kQuerySeed = 0x5EED1A77E12EB001ULL;
constexpr double kDensity = 0.5; // fraction of the bounded domain that is populated

// Direct-indexed fixed-domain prefix-rank table. prefix_[i] = #{present keys < i}.
// lower_bound_rank(q) is a single clamped array load -- the bounded-tick default.
class FixedDomainRank {
public:
    FixedDomainRank() = default;

    static FixedDomainRank from_present(const std::vector<Key>& sorted_present,
                                        std::size_t domain) {
        FixedDomainRank t;
        t.domain_ = domain;
        t.n_ = sorted_present.size();
        t.prefix_.assign(domain + 1, 0);
        std::size_t j = 0;
        for (std::size_t i = 0; i < domain; ++i) {
            t.prefix_[i] = j;
            while (j < sorted_present.size() &&
                   static_cast<std::size_t>(sorted_present[j]) == i) {
                ++j;
            }
        }
        t.prefix_[domain] = j; // == n_
        return t;
    }

    std::size_t lower_bound_rank(Key q) const noexcept {
        if (q <= 0) return prefix_[0];
        if (static_cast<std::size_t>(q) >= domain_) return n_;
        return prefix_[static_cast<std::size_t>(q)];
    }

    std::size_t size() const noexcept { return n_; }

private:
    std::vector<std::size_t> prefix_;
    std::size_t domain_ = 0;
    std::size_t n_ = 0;
};

inline std::size_t flat_rank(const std::vector<Key>& keys, Key q) noexcept {
    return static_cast<std::size_t>(
        std::lower_bound(keys.begin(), keys.end(), q) - keys.begin());
}

enum class Pattern { kUniform, kHotCold };
enum class Structure { kFixed, kFlat, kVeb };

struct Workload {
    std::size_t domain = 0;
    std::vector<Key> keys; // sorted present keys
    offline::VebStaticSearch<Key> veb;
    FixedDomainRank fixed;
};

std::vector<Key> build_present_keys(std::size_t domain) {
    std::mt19937_64 eng(kKeySeed ^ (static_cast<std::uint64_t>(domain) * 0x9E3779B97F4A7C15ULL));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::vector<Key> keys;
    keys.reserve(static_cast<std::size_t>(static_cast<double>(domain) * kDensity) + 1);
    for (std::size_t i = 0; i < domain; ++i) {
        if (coin(eng) < kDensity) keys.push_back(static_cast<Key>(i));
    }
    if (keys.empty()) keys.push_back(0);
    return keys;
}

// Self-aborting parity gate: all three structures must agree for every q in
// [0, domain]. Runs before any timing.
void require_parity(const Workload& w) {
    for (std::size_t q = 0; q <= w.domain; ++q) {
        const Key qk = static_cast<Key>(q);
        const std::size_t r_flat = flat_rank(w.keys, qk);
        const std::size_t r_veb = w.veb.lower_bound_rank(qk);
        const std::size_t r_fix = w.fixed.lower_bound_rank(qk);
        if (r_flat != r_veb || r_flat != r_fix) {
            std::fprintf(stderr,
                         "FATAL: contention parity violated at domain=%zu q=%zu "
                         "(flat=%zu veb=%zu fixed=%zu)\n",
                         w.domain, q, r_flat, r_veb, r_fix);
            std::abort();
        }
    }
}

const Workload& workload_for(std::size_t domain) {
    static std::map<std::size_t, Workload> cache;
    auto it = cache.find(domain);
    if (it != cache.end()) return it->second;
    Workload w;
    w.domain = domain;
    w.keys = build_present_keys(domain);
    w.veb = offline::VebStaticSearch<Key>::from_sorted(w.keys);
    w.fixed = FixedDomainRank::from_present(w.keys, domain);
    require_parity(w);
    auto res = cache.emplace(domain, std::move(w));
    return res.first->second;
}

// Deterministic per-thread query stream: each thread t gets its own seeded slice
// so the worker is reproducible and the streams do not trivially alias.
std::vector<Key> build_thread_queries(std::size_t domain, Pattern pattern,
                                      unsigned thread_id, std::size_t count) {
    std::mt19937_64 eng(kQuerySeed ^ (static_cast<std::uint64_t>(domain) << 1) ^
                        (static_cast<std::uint64_t>(pattern) << 33) ^
                        (static_cast<std::uint64_t>(thread_id) << 50));
    std::vector<Key> q;
    q.reserve(count);
    const std::size_t hot_hi = domain / 32 + 1; // top-of-book hot window
    std::uniform_int_distribution<std::size_t> cold(0, domain ? domain - 1 : 0);
    std::uniform_int_distribution<std::size_t> hot(0, hot_hi - 1);
    std::uniform_real_distribution<double> sel(0.0, 1.0);
    switch (pattern) {
    case Pattern::kUniform:
        for (std::size_t i = 0; i < count; ++i) q.push_back(static_cast<Key>(cold(eng)));
        break;
    case Pattern::kHotCold:
        for (std::size_t i = 0; i < count; ++i) {
            q.push_back(static_cast<Key>(sel(eng) < 0.9 ? hot(eng) : cold(eng)));
        }
        break;
    }
    return q;
}

const char* pattern_name(Pattern p) {
    switch (p) {
    case Pattern::kUniform: return "uniform";
    case Pattern::kHotCold: return "hotcold";
    }
    return "?";
}

const char* structure_name(Structure s) {
    switch (s) {
    case Structure::kFixed: return "Fixed";
    case Structure::kFlat: return "Flat";
    case Structure::kVeb: return "Veb";
    }
    return "?";
}

// 64-byte-aligned per-thread output slot: prevents false sharing between the
// values adjacent worker threads write back to the parent's result array.
struct alignas(64) ThreadResult {
    std::uint64_t checksum = 0;
    double p50_ns = 0.0;
    double p99_ns = 0.0;
    double p999_ns = 0.0;
    char pad[64 - (sizeof(std::uint64_t) + 3 * sizeof(double)) % 64];
};

template <typename Lookup>
std::uint64_t run_lookups(const std::vector<Key>& q, Lookup&& lookup) {
    std::uint64_t sum = 0;
    for (Key k : q) sum += static_cast<std::uint64_t>(lookup(k));
    return sum;
}

template <typename Lookup>
void timed_lookups(const std::vector<Key>& q, Lookup&& lookup, ThreadResult& out) {
    std::vector<double> lat;
    lat.reserve(q.size());
    std::uint64_t sum = 0;
    for (Key k : q) {
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t r = lookup(k);
        const auto t1 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(r);
        sum += static_cast<std::uint64_t>(r);
        lat.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) -> double {
        if (lat.empty()) return 0.0;
        const std::size_t i =
            static_cast<std::size_t>(p * static_cast<double>(lat.size() - 1));
        return lat[i];
    };
    out.checksum = sum;
    out.p50_ns = pct(0.50);
    out.p99_ns = pct(0.99);
    out.p999_ns = pct(0.999);
}

constexpr std::size_t kOpsPerThread = 1u << 18; // 262144 lookups per thread per burst

// One T-thread burst against a concrete (non-type-erased) lookup callable, so the
// optimizer fully inlines each structure's lookup -- no std::function indirection
// that would flatten the cross-structure comparison.
template <typename MakeLookup>
double run_burst(const Workload& w, Pattern pattern, unsigned threads,
                 bool collect_tail, std::vector<ThreadResult>& results,
                 MakeLookup make_lookup) {
    // Pre-build each thread's deterministic query slice (outside the timed region).
    std::vector<std::vector<Key>> qs(threads);
    for (unsigned t = 0; t < threads; ++t) {
        qs[t] = build_thread_queries(w.domain, pattern, t, kOpsPerThread);
    }
    results.assign(threads, ThreadResult{});

    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    pool.reserve(threads);

    std::atomic<double> elapsed_ns{0.0};
    auto start_gate = std::chrono::steady_clock::now();

    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t]() {
            auto lookup = make_lookup();
            // Spin-barrier: every worker signals ready, then all start together so
            // the timed window is genuinely concurrent (maximize contention).
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                // busy-wait
            }
            if (collect_tail) {
                timed_lookups(qs[t], lookup, results[t]);
            } else {
                results[t].checksum = run_lookups(qs[t], lookup);
            }
        });
    }

    // Wait until all workers reached the barrier, then release + time the burst.
    while (ready.load(std::memory_order_acquire) != threads) {
        std::this_thread::yield();
    }
    const auto t0 = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();
    const auto t1 = std::chrono::steady_clock::now();
    (void)start_gate;
    elapsed_ns.store(std::chrono::duration<double, std::nano>(t1 - t0).count());

    // Aggregate-checksum sink so the optimizer cannot elide the work.
    std::uint64_t agg = 0;
    for (const auto& r : results) agg += r.checksum;
    benchmark::DoNotOptimize(agg);

    return elapsed_ns.load();
}

// Dispatch on structure to a fully-typed lookup callable, then run the burst.
template <typename Body>
void with_lookup(const Workload& w, Structure s, Body&& body) {
    switch (s) {
    case Structure::kFixed:
        body([&w]() { return [&w](Key q) { return w.fixed.lower_bound_rank(q); }; });
        return;
    case Structure::kFlat:
        body([&w]() { return [&w](Key q) { return flat_rank(w.keys, q); }; });
        return;
    case Structure::kVeb:
        body([&w]() { return [&w](Key q) { return w.veb.lower_bound_rank(q); }; });
        return;
    }
}

// The contention benchmark body. Each Google Benchmark iteration runs ONE full
// T-thread burst with manual timing; throughput is ops/wall-time of that burst.
// Tail percentiles are collected once (a separate timed burst) and attached as
// counters. Aggregate Mops/s is the reliable cross-config signal.
void BM_Contention(benchmark::State& state, std::size_t domain, Pattern pattern,
                   Structure structure, unsigned threads) {
    const Workload& w = workload_for(domain);

    with_lookup(w, structure, [&](auto make_lookup) {
        std::vector<ThreadResult> results;
        for (auto _ : state) {
            const double ns = run_burst(w, pattern, threads, /*collect_tail=*/false,
                                        results, make_lookup);
            state.SetIterationTime(ns / 1e9);
        }

        const std::size_t ops = static_cast<std::size_t>(threads) * kOpsPerThread;
        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                                static_cast<std::int64_t>(ops));

        // One tail-collection burst (per-op chrono inflates absolute ns; the value
        // is tail STRUCTURE + how it shifts with contention). Report pooled p50 and
        // the WORST-thread p99/p999 (the tail a latency-sensitive consumer feels).
        std::vector<ThreadResult> tail;
        run_burst(w, pattern, threads, /*collect_tail=*/true, tail, make_lookup);
        double p50_sum = 0.0, p99_max = 0.0, p999_max = 0.0;
        for (const auto& r : tail) {
            p50_sum += r.p50_ns;
            p99_max = std::max(p99_max, r.p99_ns);
            p999_max = std::max(p999_max, r.p999_ns);
        }
        state.counters["threads"] = static_cast<double>(threads);
        state.counters["p50_ns"] =
            tail.empty() ? 0.0 : p50_sum / static_cast<double>(tail.size());
        state.counters["p99_ns_worstthread"] = p99_max;
        state.counters["p999_ns_worstthread"] = p999_max;
    });
}

struct ContentionRegistrar {
    ContentionRegistrar() {
        const int exps[] = {12, 16, 20}; // representative bounded ladder sizes
        const Pattern pats[] = {Pattern::kUniform, Pattern::kHotCold};
        const Structure structs[] = {Structure::kFixed, Structure::kFlat, Structure::kVeb};
        const unsigned thread_counts[] = {1, 2, 4, 8};
        for (int b : exps) {
            const std::size_t domain = static_cast<std::size_t>(1) << b;
            for (Pattern p : pats) {
                for (Structure s : structs) {
                    for (unsigned T : thread_counts) {
                        const std::string name =
                            std::string("BM_Contention_") + structure_name(s) +
                            "/d2^" + std::to_string(b) + "/" + pattern_name(p) +
                            "/T" + std::to_string(T);
                        benchmark::RegisterBenchmark(
                            name.c_str(),
                            [domain, p, s, T](benchmark::State& st) {
                                BM_Contention(st, domain, p, s, T);
                            })
                            ->UseManualTime();
                    }
                }
            }
        }
    }
};

const ContentionRegistrar g_contention_registrar;

} // namespace
