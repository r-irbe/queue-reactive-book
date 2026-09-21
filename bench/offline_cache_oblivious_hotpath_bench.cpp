#include <benchmark/benchmark.h>

#include "offline/veb_static_search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

// =============================================================================
// OFFLINE, KERNEL-ADJACENT hot-path comparison benchmark.
//
// PURPOSE (closes `needs-hotpath-benchmark` for
//   `moc-offline-cache-oblivious-data-structures`):
//   Produce REAL deterministic evidence on whether the cache-oblivious
//   van-Emde-Boas (vEB) static layout helps a price-ladder-sized integer-key
//   LOOKUP hot path, versus the two simple baselines the project actually uses:
//     (a) offline vEB static search index (offline/veb_static_search.hpp),
//     (b) flat sorted std::vector + std::lower_bound,
//     (c) a DIRECT-INDEXED FIXED-DOMAIN array (prefix-rank table) — the
//         project's default for bounded ticks: rank(q) is a single O(1) load.
//
// HARD BOUNDARY (NEVER VIOLATED):
//   This translation unit links NOTHING from the matching kernel. It does NOT
//   #include "sim/flat_ladder_book.hpp", does NOT reference any `sim::` symbol,
//   and never touches FlatLadderBook. It depends only on the self-contained
//   header `offline/veb_static_search.hpp` and the C++ standard library. It is a
//   self-contained offline comparison. (bench/main.cpp owns BENCHMARK_MAIN();
//   this file only adds BENCHMARK registrations and is auto-globbed by the
//   Makefile BENCH_SRC wildcard.)
//
// WORKLOAD (representative of a bounded price ladder):
//   * A bounded integer domain [0, D) of D = 2^b ticks, b in {8,10,12,14,16,18,20}.
//     the domain (a populated ladder is sparse-to-dense within fixed bounds).
//   * The query stream is deterministic and seeded, in three access patterns:
//       - uniform : queries uniform over [0, D).
//       - hotcold : 90% drawn from a small hot 1/32 window, 10% cold uniform
//                   (models a ladder whose action clusters near top-of-book).
//       - sequential : queries sweep 0,1,2,... wrapping (cache-friendly ascending).
//
// SEMANTIC CONTRACT / SELF-ABORT:
//   For every query q in [0, D], all three structures MUST return the identical
//   lower_bound_rank (number of present keys < q). A parity check runs at build
//   time; ANY mismatch calls std::abort() so the bench can never report timings
//   on a broken structure. The fixed-domain prefix-rank table is, by construction,
//   the canonical O(1) answer; vEB and flat must match it.
//
// DETERMINISM / HONESTY:
//   Keys + queries derive from FIXED seeds, so the workload is reproducible on
//   any host. Per-query p50/p90/p99/p999 are computed from a single timed sweep
//   and reported as counters; ABSOLUTE nanoseconds are environment-dependent and
//   the per-query std::chrono read adds fixed overhead that dominates these
//   sub-100ns operations, so the steady-clock percentiles OVERSTATE absolute
//   cost. The reliable comparison is the Google Benchmark batched mean
//   (ns/op + items/s) reported by each BENCHMARK row; the percentile counters
//   capture deterministic STRUCTURE (tail shape), not an isolated speedup claim.
// =============================================================================

namespace {

using Key = std::int64_t;

constexpr std::uint64_t kKeySeed = 0xC0FFEE17ACEB00C1ULL;
constexpr std::uint64_t kQuerySeed = 0x5EED1A77E12EB001ULL;
constexpr double kDensity = 0.5; // fraction of the bounded domain that is populated

// Direct-indexed fixed-domain prefix-rank table. For a bounded tick domain
// [0, D), prefix_[i] holds the number of present keys strictly less than i.
// lower_bound_rank(q) is then a single clamped array load — the canonical answer
// the project's bounded-tick default already provides.
class FixedDomainRank {
public:
    FixedDomainRank() = default;

    static FixedDomainRank from_present(const std::vector<Key>& sorted_present,
                                        std::size_t domain) {
        FixedDomainRank t;
        t.domain_ = domain;
        t.n_ = sorted_present.size();
        // prefix_ has domain+1 entries: prefix_[i] = #{k in present : k < i}.
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

    // Clamped to the bounded domain: q < 0 -> 0, q >= domain -> n_.
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

// Flat sorted-array baseline lower_bound rank.
inline std::size_t flat_rank(const std::vector<Key>& keys, Key q) noexcept {
    return static_cast<std::size_t>(
        std::lower_bound(keys.begin(), keys.end(), q) - keys.begin());
}

enum class Pattern { kUniform, kHotCold, kSequential };

// One immutable, deterministic workload for a given domain exponent: the bounded
// domain, the sorted present-key set, and all three lookup structures. Built once
// and cached so the 9 (pattern x structure) benchmarks at a domain share it.
struct Workload {
    std::size_t domain = 0;
    std::vector<Key> keys;                       // sorted present keys
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
    if (keys.empty()) keys.push_back(0); // guarantee a non-empty ladder
    return keys; // already ascending and unique by construction
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
                         "FATAL: hotpath parity violated at domain=%zu q=%zu "
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

std::vector<Key> build_queries(std::size_t domain, Pattern pattern,
                               std::size_t count) {
    std::mt19937_64 eng(kQuerySeed ^ (static_cast<std::uint64_t>(domain) << 1) ^
                        (static_cast<std::uint64_t>(pattern) << 33));
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
    case Pattern::kSequential:
        for (std::size_t i = 0; i < count; ++i) {
            q.push_back(static_cast<Key>(domain ? (i % domain) : 0));
        }
        break;
    }
    return q;
}

const std::vector<Key>& queries_for(std::size_t domain, Pattern pattern,
                                    std::size_t count) {
    static std::map<std::uint64_t, std::vector<Key>> cache;
    const std::uint64_t key = (static_cast<std::uint64_t>(domain) << 8) ^
                              (static_cast<std::uint64_t>(pattern) << 4) ^ count;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto res = cache.emplace(key, build_queries(domain, pattern, count));
    return res.first->second;
}

const char* pattern_name(Pattern p) {
    switch (p) {
    case Pattern::kUniform: return "uniform";
    case Pattern::kHotCold: return "hotcold";
    case Pattern::kSequential: return "sequential";
    }
    return "?";
}

// Compute the deterministic per-query latency envelope (p50/p90/p99/p999) over a
// single timed sweep and attach it as benchmark counters. NOTE: the per-query
// std::chrono read dominates these sub-100ns ops, so these numbers OVERSTATE
// absolute cost; their VALUE is the deterministic tail STRUCTURE, not isolated
// speedup. The reliable cross-structure comparison is the batched ns/op below.
template <typename Fn>
void attach_envelope(benchmark::State& state, const std::vector<Key>& queries,
                     Fn&& lookup) {
    std::vector<double> lat_ns;
    lat_ns.reserve(queries.size());
    for (Key q : queries) {
        const auto t0 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(lookup(q));
        const auto t1 = std::chrono::steady_clock::now();
        lat_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(lat_ns.begin(), lat_ns.end());
    auto pct = [&](double p) -> double {
        if (lat_ns.empty()) return 0.0;
        const std::size_t i =
            static_cast<std::size_t>(p * static_cast<double>(lat_ns.size() - 1));
        return lat_ns[i];
    };
    state.counters["p50_ns"] = pct(0.50);
    state.counters["p90_ns"] = pct(0.90);
    state.counters["p99_ns"] = pct(0.99);
    state.counters["p999_ns"] = pct(0.999);
}

constexpr std::size_t kNumQueries = 1u << 16; // 65536 deterministic queries per sweep

// ---- Batched throughput benchmarks (reliable mean ns/op) + envelope counters --

void BM_Hotpath_Veb(benchmark::State& state, std::size_t domain, Pattern pattern) {
    const Workload& w = workload_for(domain);
    const std::vector<Key>& queries = queries_for(domain, pattern, kNumQueries);
    std::size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(w.veb.lower_bound_rank(queries[i]));
        if (++i == queries.size()) i = 0;
    }
    state.SetItemsProcessed(state.iterations());
    attach_envelope(state, queries, [&](Key q) { return w.veb.lower_bound_rank(q); });
}

void BM_Hotpath_Flat(benchmark::State& state, std::size_t domain, Pattern pattern) {
    const Workload& w = workload_for(domain);
    const std::vector<Key>& queries = queries_for(domain, pattern, kNumQueries);
    std::size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(flat_rank(w.keys, queries[i]));
        if (++i == queries.size()) i = 0;
    }
    state.SetItemsProcessed(state.iterations());
    attach_envelope(state, queries, [&](Key q) { return flat_rank(w.keys, q); });
}

void BM_Hotpath_Fixed(benchmark::State& state, std::size_t domain, Pattern pattern) {
    const Workload& w = workload_for(domain);
    const std::vector<Key>& queries = queries_for(domain, pattern, kNumQueries);
    std::size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(w.fixed.lower_bound_rank(queries[i]));
        if (++i == queries.size()) i = 0;
    }
    state.SetItemsProcessed(state.iterations());
    attach_envelope(state, queries, [&](Key q) { return w.fixed.lower_bound_rank(q); });
}

// Register veb/flat/fixed for every (domain exponent x access pattern) cell.
struct HotpathRegistrar {
    HotpathRegistrar() {
        const int exps[] = {8, 10, 12, 14, 16, 18, 20};
        const Pattern pats[] = {Pattern::kUniform, Pattern::kHotCold,
                                Pattern::kSequential};
        for (int b : exps) {
            const std::size_t domain = static_cast<std::size_t>(1) << b;
            for (Pattern p : pats) {
                const std::string suffix =
                    "/d2^" + std::to_string(b) + "/" + pattern_name(p);
                benchmark::RegisterBenchmark(
                    ("BM_Hotpath_Veb" + suffix).c_str(),
                    [domain, p](benchmark::State& s) { BM_Hotpath_Veb(s, domain, p); });
                benchmark::RegisterBenchmark(
                    ("BM_Hotpath_Flat" + suffix).c_str(),
                    [domain, p](benchmark::State& s) { BM_Hotpath_Flat(s, domain, p); });
                benchmark::RegisterBenchmark(
                    ("BM_Hotpath_Fixed" + suffix).c_str(),
                    [domain, p](benchmark::State& s) { BM_Hotpath_Fixed(s, domain, p); });
            }
        }
    }
};

const HotpathRegistrar g_hotpath_registrar;

} // namespace
