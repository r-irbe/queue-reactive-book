#include <catch2/catch_test_macros.hpp>

#include "offline/veb_static_search.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// =============================================================================
// Deterministic scan-parity / replay-equivalence falsifier for the OFFLINE
// cache-oblivious (van-Emde-Boas layout) static search index.
//
// This is the gate-G8 parity fixture for
// `moc-offline-cache-oblivious-data-structures`: it proves the cache-oblivious
// layout returns byte-identical query results to a flat sorted-array baseline
// (std::lower_bound / std::binary_search) across randomized-but-SEEDED
// workloads. It touches NOTHING in the matching kernel; it depends only on the
// self-contained header `offline/veb_static_search.hpp`.
//
// Determinism: a fixed std::mt19937_64 seed per case drives both the key set
// and the query stream, so the falsifier is fully reproducible.
// =============================================================================

namespace {

using offline::VebStaticSearch;

// Flat sorted-array baseline (the "canonical scan" the index must match).
struct FlatBaseline {
    std::vector<std::int64_t> keys; // sorted ascending

    std::size_t lower_bound_rank(std::int64_t q) const {
        return static_cast<std::size_t>(
            std::lower_bound(keys.begin(), keys.end(), q) - keys.begin());
    }
    bool contains(std::int64_t q) const {
        return std::binary_search(keys.begin(), keys.end(), q);
    }
};

// Build a sorted key vector of `n` keys from a seeded engine. `allow_dups`
// controls whether duplicate keys are permitted (lower_bound semantics differ
// from binary_search there, so both are exercised).
std::vector<std::int64_t> make_sorted_keys(std::uint64_t seed, std::size_t n,
                                           std::int64_t lo, std::int64_t hi,
                                           bool allow_dups) {
    std::mt19937_64 eng(seed);
    std::uniform_int_distribution<std::int64_t> dist(lo, hi);
    std::vector<std::int64_t> keys;
    keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i) keys.push_back(dist(eng));
    std::sort(keys.begin(), keys.end());
    if (!allow_dups) keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

// Run a seeded query stream and assert byte-identical parity for every query.
// Returns the number of queries actually compared.
std::size_t assert_parity(std::uint64_t seed, const std::vector<std::int64_t>& keys,
                          std::int64_t qlo, std::int64_t qhi, std::size_t queries) {
    FlatBaseline flat{keys};
    auto idx = VebStaticSearch<std::int64_t>::from_sorted(keys);

    REQUIRE(idx.size() == keys.size());

    std::mt19937_64 eng(seed);
    std::uniform_int_distribution<std::int64_t> qdist(qlo, qhi);
    std::size_t compared = 0;

    // Always probe deterministic boundary queries first.
    std::vector<std::int64_t> probes;
    probes.push_back(qlo);
    probes.push_back(qhi);
    if (!keys.empty()) {
        probes.push_back(keys.front());
        probes.push_back(keys.back());
        probes.push_back(keys.front() - 1);
        probes.push_back(keys.back() + 1);
        probes.push_back(keys[keys.size() / 2]);
    }
    for (std::size_t i = 0; i < queries; ++i) probes.push_back(qdist(eng));

    for (std::int64_t q : probes) {
        const std::size_t got_rank = idx.lower_bound_rank(q);
        const std::size_t exp_rank = flat.lower_bound_rank(q);
        REQUIRE(got_rank == exp_rank);

        const bool got_has = idx.contains(q);
        const bool exp_has = flat.contains(q);
        REQUIRE(got_has == exp_has);
        ++compared;
    }
    return compared;
}

} // namespace

TEST_CASE("offline vEB static search: empty and singleton parity",
          "[offline][cache-oblivious][scan-parity]") {
    // Empty index: lower_bound is always 0, contains always false.
    auto empty = VebStaticSearch<std::int64_t>::from_sorted({});
    REQUIRE(empty.empty());
    REQUIRE(empty.lower_bound_rank(0) == 0);
    REQUIRE(empty.lower_bound_rank(-5) == 0);
    REQUIRE_FALSE(empty.contains(0));

    // Singleton.
    auto one = VebStaticSearch<std::int64_t>::from_sorted({42});
    REQUIRE(one.lower_bound_rank(41) == 0);
    REQUIRE(one.lower_bound_rank(42) == 0);
    REQUIRE(one.lower_bound_rank(43) == 1);
    REQUIRE(one.contains(42));
    REQUIRE_FALSE(one.contains(41));
    REQUIRE_FALSE(one.contains(43));
}

TEST_CASE("offline vEB static search: exhaustive small-domain parity",
          "[offline][cache-oblivious][scan-parity]") {
    // Dense domain so every query value is exercised against every structure.
    const std::vector<std::int64_t> keys = {-3, -1, 0, 2, 5, 8, 8, 13, 21};
    FlatBaseline flat{keys};
    auto idx = VebStaticSearch<std::int64_t>::from_sorted(keys);
    for (std::int64_t q = -6; q <= 25; ++q) {
        REQUIRE(idx.lower_bound_rank(q) == flat.lower_bound_rank(q));
        REQUIRE(idx.contains(q) == flat.contains(q));
    }
}

TEST_CASE("offline vEB static search: seeded randomized parity (no duplicates)",
          "[offline][cache-oblivious][scan-parity]") {
    std::size_t total = 0;
    // Fixed seeds and sizes -> deterministic, reproducible falsifier.
    const std::size_t sizes[] = {2, 3, 7, 15, 16, 63, 100, 255, 1000, 4096};
    std::uint64_t seed = 0xC0FFEEULL;
    for (std::size_t n : sizes) {
        auto keys = make_sorted_keys(seed, n, -100000, 100000, /*allow_dups=*/false);
        total += assert_parity(seed ^ 0x9E3779B97F4A7C15ULL, keys, -120000, 120000,
                               /*queries=*/512);
        seed += 0x1234567ULL;
    }
    REQUIRE(total > 0);
}

TEST_CASE("offline vEB static search: seeded randomized parity (with duplicates)",
          "[offline][cache-oblivious][scan-parity]") {
    std::size_t total = 0;
    const std::size_t sizes[] = {8, 32, 200, 777, 2048};
    std::uint64_t seed = 0xABCDEF01ULL;
    for (std::size_t n : sizes) {
        // Narrow key domain forces many duplicates (exercises lower_bound vs
        // binary_search divergence under parity).
        auto keys = make_sorted_keys(seed, n, 0, 50, /*allow_dups=*/true);
        total += assert_parity(seed ^ 0xD1B54A32D192ED03ULL, keys, -5, 60,
                               /*queries=*/512);
        seed += 0x77777ULL;
    }
    REQUIRE(total > 0);
}

TEST_CASE("offline vEB static search: unsigned key type parity",
          "[offline][cache-oblivious][scan-parity]") {
    std::mt19937_64 eng(0x5151515151515151ULL);
    std::uniform_int_distribution<std::uint32_t> dist(0, 1000000);
    std::vector<std::uint32_t> keys;
    for (int i = 0; i < 1500; ++i) keys.push_back(dist(eng));
    std::sort(keys.begin(), keys.end());

    auto idx = VebStaticSearch<std::uint32_t>::from_sorted(keys);
    std::uniform_int_distribution<std::uint32_t> qdist(0, 1100000);
    for (int i = 0; i < 2000; ++i) {
        const std::uint32_t q = qdist(eng);
        const std::size_t exp =
            static_cast<std::size_t>(std::lower_bound(keys.begin(), keys.end(), q) -
                                     keys.begin());
        REQUIRE(idx.lower_bound_rank(q) == exp);
        REQUIRE(idx.contains(q) == std::binary_search(keys.begin(), keys.end(), q));
    }
}
