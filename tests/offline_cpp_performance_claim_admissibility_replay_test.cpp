#include <catch2/catch_test_macros.hpp>

#include "offline/cpp_performance_claim_admissibility_replay.hpp"

#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline falsifier for moc-cpp-performance-claim-admissibility's
// code/replay channel (gate G4). It proves, with ZERO matching-kernel / sim
// coupling, that:
//   (a) the admissibility CHECKER is constructible and deterministic;
//   (b) an admissible performance claim carries a deterministic correctness
//       ORACLE (optimized == naive bit-for-bit over an exhaustive/representative
//       integer domain) AND a REPLAY-NEUTRALITY guarantee (identical output-
//       stream hash), demonstrated on Hacker's-Delight bit-twiddle transforms;
//   (c) a broken "optimized" routine is detected as INADMISSIBLE (oracle
//       mismatch + hash divergence) -- the negative control;
//   (d) the checker is strictly read-only / non-feedback (const-correctness
//       enforced at compile time via static_assert);
//   (e) timing is illustrative and is NEVER asserted on.
//
// Depends only on the self-contained header
// `offline/cpp_performance_claim_admissibility_replay.hpp`.
// =============================================================================

namespace {

using offline::AdmissibilityVerdict;
using offline::PerformanceClaimAdmissibilityReplay;
using offline::Transform;
using offline::u32;
using offline::u64;

using i32_window_t = std::int32_t;

// Build a representative + boundary + exhaustive-16-bit-window integer domain so
// the bit-for-bit oracle is exercised across the corners that break naive code
// (INT_MIN, INT_MAX, +-1, 0, powers of two, all-ones), an exhaustive low window,
// a strided high sweep, and a seeded pseudo-random sample.
std::vector<u32> representative_domain() {
    std::vector<u32> d;

    // Explicit boundaries.
    const u32 corners[] = {
        0u, 1u, 2u, 3u,
        0x7FFFFFFFu,                 // INT_MAX
        0x80000000u,                 // INT_MIN
        0x80000001u,                 // INT_MIN + 1
        0xFFFFFFFFu,                 // -1
        0xFFFFFFFEu,                 // -2
        0x40000000u, 0x40000001u,    // around 2^30
    };
    for (u32 c : corners) {
        d.push_back(c);
    }

    // Powers of two and their neighbours (clp2 boundaries).
    for (int s = 0; s < 32; ++s) {
        const u32 p = (s < 31) ? (1u << s) : 0x80000000u;
        d.push_back(p);
        d.push_back(p - 1u);
        d.push_back(p + 1u);
    }

    // Exhaustive low window: every value in [-2048, 2048].
    for (i32_window_t v = -2048; v <= 2048; ++v) {
        d.push_back(static_cast<u32>(v));
    }

    // Strided sweep across the full 32-bit range.
    for (u64 v = 0; v <= 0xFFFFFFFFull; v += 0x00100001ull) {
        d.push_back(static_cast<u32>(v));
    }

    // Seeded pseudo-random sample (deterministic across runs).
    std::mt19937_64 eng(0xC0FFEEULL);
    std::uniform_int_distribution<u32> dist(0u, 0xFFFFFFFFu);
    for (int i = 0; i < 4096; ++i) {
        d.push_back(dist(eng));
    }

    return d;
}

// A genuinely exhaustive 16-bit domain (all of [0, 65535] plus their sign-
// extended negatives), used to prove popcount/clp2 over a fully exhaustive set.
std::vector<u32> exhaustive_16bit_domain() {
    std::vector<u32> d;
    d.reserve(0x20000);
    for (u32 v = 0; v <= 0xFFFFu; ++v) {
        d.push_back(v);
        d.push_back(0xFFFF0000u | v); // negative half (high 16 bits set)
    }
    return d;
}

} // namespace

// (d) Compile-time read-only / non-feedback guarantees mirrored in the test TU
// so a regression that makes the checker mutating fails THIS translation unit's
// compile too.
static_assert(
    std::is_invocable_r_v<AdmissibilityVerdict,
                          decltype(&PerformanceClaimAdmissibilityReplay::check_claim),
                          const PerformanceClaimAdmissibilityReplay&, Transform,
                          Transform>,
    "check_claim() must be callable on a const checker (read-only)");
static_assert(
    std::is_invocable_r_v<u64,
                          decltype(&PerformanceClaimAdmissibilityReplay::hash_transform),
                          const PerformanceClaimAdmissibilityReplay&, Transform>,
    "hash_transform() must be callable on a const checker (read-only)");

TEST_CASE("offline perf-claim admissibility: admissible claims pass the oracle "
          "and are replay-neutral",
          "[offline][cpp-perf][admissibility]") {
    const PerformanceClaimAdmissibilityReplay checker(representative_domain());
    REQUIRE_FALSE(checker.empty());

    struct Claim {
        const char* name;
        Transform reference;
        Transform optimized;
    };
    const Claim claims[] = {
        {"sign", &offline::naive_sign, &offline::opt_sign},
        {"abs", &offline::naive_abs, &offline::opt_abs},
        {"clp2", &offline::naive_clp2, &offline::opt_clp2},
        {"popcount", &offline::naive_popcount, &offline::opt_popcount},
    };

    std::size_t admissible_count = 0;
    for (const Claim& c : claims) {
        const AdmissibilityVerdict v =
            checker.check_claim(c.reference, c.optimized);

        // (b) deterministic correctness oracle: optimized == naive bit-for-bit.
        REQUIRE(v.samples_checked == checker.size());
        REQUIRE(v.mismatch_count == 0);
        REQUIRE(v.oracle_matched);

        // (b) replay-neutrality: identical output-stream hash, so swapping the
        // optimized path into a replay changes no observable state.
        REQUIRE(v.reference_hash == v.optimized_hash);
        REQUIRE(v.replay_neutral);
        REQUIRE(v.reference_hash != 0);

        REQUIRE(v.admissible());
        ++admissible_count;
    }
    REQUIRE(admissible_count == 4);
}

TEST_CASE("offline perf-claim admissibility: exhaustive 16-bit oracle",
          "[offline][cpp-perf][admissibility][exhaustive]") {
    const PerformanceClaimAdmissibilityReplay checker(exhaustive_16bit_domain());
    REQUIRE(checker.size() == 0x20000);

    // Over a fully exhaustive 16-bit (+ sign-extended) domain, popcount and clp2
    // optimized forms match their references bit-for-bit and are replay-neutral.
    const AdmissibilityVerdict pc =
        checker.check_claim(&offline::naive_popcount, &offline::opt_popcount);
    REQUIRE(pc.mismatch_count == 0);
    REQUIRE(pc.admissible());

    const AdmissibilityVerdict cl =
        checker.check_claim(&offline::naive_clp2, &offline::opt_clp2);
    REQUIRE(cl.mismatch_count == 0);
    REQUIRE(cl.admissible());
}

TEST_CASE("offline perf-claim admissibility: a broken claim is INADMISSIBLE "
          "(negative control)",
          "[offline][cpp-perf][admissibility][negative]") {
    const PerformanceClaimAdmissibilityReplay checker(representative_domain());

    // A deliberately broken "optimized" sign that drops the positive case must
    // be detected: the oracle finds mismatches and the output-stream hash
    // diverges, so the claim is NOT replay-neutral and NOT admissible.
    const AdmissibilityVerdict v =
        checker.check_claim(&offline::naive_sign, &offline::broken_opt_sign);

    REQUIRE(v.mismatch_count > 0);
    REQUIRE_FALSE(v.oracle_matched);
    REQUIRE(v.reference_hash != v.optimized_hash);
    REQUIRE_FALSE(v.replay_neutral);
    REQUIRE_FALSE(v.admissible());
}

TEST_CASE("offline perf-claim admissibility: verdicts are deterministic / "
          "replay-hash invariant",
          "[offline][cpp-perf][admissibility][determinism]") {
    const PerformanceClaimAdmissibilityReplay a(representative_domain());
    const PerformanceClaimAdmissibilityReplay b(representative_domain());

    // Same domain + same transforms -> identical verdict, every time.
    const AdmissibilityVerdict v1 =
        a.check_claim(&offline::naive_abs, &offline::opt_abs);
    const AdmissibilityVerdict v2 =
        a.check_claim(&offline::naive_abs, &offline::opt_abs); // idempotent
    const AdmissibilityVerdict v3 =
        b.check_claim(&offline::naive_abs, &offline::opt_abs); // independent build

    REQUIRE(v1.reference_hash == v2.reference_hash);
    REQUIRE(v1.optimized_hash == v2.optimized_hash);
    REQUIRE(v1.reference_hash == v3.reference_hash);
    REQUIRE(v1.optimized_hash == v3.optimized_hash);

    // hash_transform agrees with the stream hashes computed inside check_claim.
    REQUIRE(a.hash_transform(&offline::naive_abs) == v1.reference_hash);
    REQUIRE(a.hash_transform(&offline::opt_abs) == v1.optimized_hash);

    // A different transform produces a different output-stream hash.
    REQUIRE(a.hash_transform(&offline::naive_sign) !=
            a.hash_transform(&offline::naive_abs));
}

TEST_CASE("offline perf-claim admissibility: empty domain is well-defined",
          "[offline][cpp-perf][admissibility][determinism]") {
    const PerformanceClaimAdmissibilityReplay checker{};
    REQUIRE(checker.empty());

    // With no samples, the oracle trivially matches and the paths are vacuously
    // replay-neutral; the verdict is admissible but checked zero samples.
    const AdmissibilityVerdict v =
        checker.check_claim(&offline::naive_sign, &offline::opt_sign);
    REQUIRE(v.samples_checked == 0);
    REQUIRE(v.mismatch_count == 0);
    REQUIRE(v.oracle_matched);
    REQUIRE(v.replay_neutral);
    REQUIRE(v.admissible());

    // Even a broken transform is vacuously admissible over an empty domain --
    // which is exactly why a NON-empty representative domain is load-bearing.
    const AdmissibilityVerdict broken =
        checker.check_claim(&offline::naive_sign, &offline::broken_opt_sign);
    REQUIRE(broken.admissible());
}

TEST_CASE("offline perf-claim admissibility: timing is illustrative and never "
          "decides admissibility",
          "[offline][cpp-perf][admissibility][timing]") {
    const PerformanceClaimAdmissibilityReplay checker(representative_domain());

    // Timing is collected purely to document the separate speed axis. It is
    // NON-DETERMINISTIC, so we assert ONLY that it is well-formed (finite,
    // non-negative) -- never that one path is faster, and never as part of the
    // admissibility verdict.
    const auto t = checker.illustrative_timing(&offline::naive_popcount,
                                               &offline::opt_popcount);
    REQUIRE(t.reference_ns >= 0.0);
    REQUIRE(t.optimized_ns >= 0.0);

    // Admissibility is decided entirely by the deterministic oracle, regardless
    // of which path happened to be faster on this run.
    const AdmissibilityVerdict v =
        checker.check_claim(&offline::naive_popcount, &offline::opt_popcount);
    REQUIRE(v.admissible());
}
