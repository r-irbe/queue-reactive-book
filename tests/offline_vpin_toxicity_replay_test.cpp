#include <catch2/catch_test_macros.hpp>

#include "offline/vpin_toxicity_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic MOC-level VPIN / flow-toxicity bucket replay fixture opening the
//
// Proves the properties the MOC's load-bearing toxicity-as-diagnostic boundary
// needs:
//   (a) REPLAY-HASH DETERMINISM + ORDER-SENSITIVITY -- same signed-trade stream
//       + bucket size -> identical buckets + FNV hash; reorder / tamper changes
//       the hash.
//   (b) EQUAL-VOLUME BUCKETING + SPLIT CORRECTNESS -- trades are packed into
//       buckets of exactly `bucket_size`, splitting a trade across a boundary;
//       buy/sell volume, signed imbalance, and bounded toxicity match
//       hand-checked integer-tick values; a trailing partial bucket is excluded.
//   (c) WINDOWED VPIN -- the volume-weighted mean absolute imbalance over the
//       last `window` buckets matches hand-checked values and is bounded in
//       [0, SCALE].
//   (d) TOXICITY EXTREMES -- a one-sided stream saturates toxicity at SCALE; a
//       perfectly balanced stream yields zero toxicity.
//   (e) DIAGNOSTIC BOUNDARY -- a VPIN bucket / result is flagged NOT a
//       matching-kernel input (compile-time + value level), while a positive
//       control kernel gate IS; the VPIN computer consumes a const trade view
//       and the offline types carry no matching-engine coupling (static_asserts
//       + positive controls).
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/vpin_toxicity_replay.hpp` (zero `sim::` symbols).
// =============================================================================

namespace {

using offline::SignedTrade;
using offline::VolumeBucket;
using offline::VpinResult;
using offline::VpinToxicityReplay;

SignedTrade trade(std::uint64_t seq, std::uint64_t ts, std::int64_t price,
                  std::int64_t volume, std::int8_t side) {
    SignedTrade t;
    t.seq = seq;
    t.ts_ns = ts;
    t.price = price;
    t.volume = volume;
    t.side = side;
    return t;
}

// A hand-built signed-trade stream with bucket size 10. Packed:
//   bucket0: buy 6, sell 4   (full)  imbalance +2  abs 2  tox 200000
//   bucket1: buy 8, sell 2   (full)  imbalance +6  abs 6  tox 600000
//   bucket2: sell 10         (full)  imbalance -10 abs 10 tox 1000000
//   residual: buy 4 (partial, excluded)
std::vector<SignedTrade> sample_stream() {
    return {
        trade(1, 1000, 100, 6, +1),   // buy 6  -> bucket0
        trade(2, 1100, 100, 4, -1),   // sell 4 -> bucket0 full
        trade(3, 1200, 101, 3, +1),   // buy 3  -> bucket1
        trade(4, 1300, 101, 5, +1),   // buy 5  -> bucket1
        trade(5, 1400, 102, 2, -1),   // sell 2 -> bucket1 full
        trade(6, 1500, 102, 10, -1),  // sell 10-> bucket2 full
        trade(7, 1600, 103, 4, +1),   // buy 4  -> residual (partial)
    };
}

constexpr std::int64_t kBucketSize = 10;

}  // namespace

TEST_CASE("vpin replay-hash is deterministic and order-sensitive") {
    const std::vector<SignedTrade> a = sample_stream();
    const std::vector<SignedTrade> b = sample_stream();

    const VpinResult ra = offline::evaluate_vpin(a, kBucketSize);
    const VpinResult rb = offline::evaluate_vpin(b, kBucketSize);

    // Identical streams + bucket size -> identical buckets and identical hash,
    // and a re-run is bit-stable.
    REQUIRE(ra.replay_hash == rb.replay_hash);
    REQUIRE(offline::evaluate_vpin(a, kBucketSize).replay_hash == ra.replay_hash);
    REQUIRE(ra.buckets.size() == rb.buckets.size());

    // The read-only wrapper agrees with the free function.
    const VpinToxicityReplay replay(a);
    REQUIRE(replay.hash(kBucketSize) == ra.replay_hash);
    REQUIRE(replay.evaluate(kBucketSize).buckets.size() == ra.buckets.size());
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<SignedTrade> empty;
    REQUIRE(offline::evaluate_vpin(empty, kBucketSize).replay_hash ==
            offline::evaluate_vpin(empty, kBucketSize).replay_hash);
    REQUIRE(offline::evaluate_vpin(empty, kBucketSize).replay_hash !=
            ra.replay_hash);

    SECTION("reordering two distinct trades changes the hash") {
        std::vector<SignedTrade> reordered = a;
        std::swap(reordered[0], reordered[5]);  // move buy6 and sell10
        REQUIRE(offline::evaluate_vpin(reordered, kBucketSize).replay_hash !=
                ra.replay_hash);
    }

    SECTION("tampering with a trade volume changes the hash") {
        std::vector<SignedTrade> tampered = a;
        tampered[2].volume += 1;
        REQUIRE(offline::evaluate_vpin(tampered, kBucketSize).replay_hash !=
                ra.replay_hash);
    }

    SECTION("changing the bucket size changes the hash") {
        REQUIRE(offline::evaluate_vpin(a, 5).replay_hash != ra.replay_hash);
    }
}

TEST_CASE("vpin equal-volume bucketing matches a hand-checked series") {
    const std::vector<SignedTrade> s = sample_stream();
    const VpinResult r = offline::evaluate_vpin(s, kBucketSize);

    // 7 trades, 3 full buckets, a partial trailing bucket of volume 4.
    REQUIRE(r.n_trades == 7);
    REQUIRE(r.n_buckets == 3);
    REQUIRE(r.bucketed_volume == 30);
    REQUIRE(r.residual_volume == 4);

    // bucket0: buy 6, sell 4.
    REQUIRE(r.buckets[0].buy_vol == 6);
    REQUIRE(r.buckets[0].sell_vol == 4);
    REQUIRE(r.buckets[0].total_vol == 10);
    REQUIRE(r.buckets[0].imbalance == 2);
    REQUIRE(r.buckets[0].abs_imbalance == 2);
    REQUIRE(r.buckets[0].toxicity_scaled == 200000);

    // bucket1: buy 8, sell 2.
    REQUIRE(r.buckets[1].buy_vol == 8);
    REQUIRE(r.buckets[1].sell_vol == 2);
    REQUIRE(r.buckets[1].imbalance == 6);
    REQUIRE(r.buckets[1].abs_imbalance == 6);
    REQUIRE(r.buckets[1].toxicity_scaled == 600000);

    // bucket2: sell 10 (fully one-sided).
    REQUIRE(r.buckets[2].buy_vol == 0);
    REQUIRE(r.buckets[2].sell_vol == 10);
    REQUIRE(r.buckets[2].imbalance == -10);
    REQUIRE(r.buckets[2].abs_imbalance == 10);
    REQUIRE(r.buckets[2].toxicity_scaled == 1000000);

    // sum |imbalance| = 2 + 6 + 10 = 18 over 30 volume.
    REQUIRE(r.sum_abs_imbalance == 18);
    // vpin_scaled = 18 * 1e6 / 30 = 600000.
    REQUIRE(r.vpin_scaled == 600000);
}

TEST_CASE("vpin splits a trade across a bucket boundary") {
    // bucket size 10:
    //   buy 10            -> bucket0 full (imbalance +10)
    //   sell 15           -> bucket1 sell 10 full (-10); carry sell 5
    //   buy 5             -> bucket2 buy 5 + carried sell 5 = full (imbalance 0)
    const std::vector<SignedTrade> s = {
        trade(1, 1000, 100, 10, +1),
        trade(2, 1100, 100, 15, -1),
        trade(3, 1200, 100, 5, +1),
    };
    const VpinResult r = offline::evaluate_vpin(s, kBucketSize);

    REQUIRE(r.n_buckets == 3);
    REQUIRE(r.residual_volume == 0);

    REQUIRE(r.buckets[0].buy_vol == 10);
    REQUIRE(r.buckets[0].imbalance == 10);

    // The split sell fills bucket1 entirely on the sell side...
    REQUIRE(r.buckets[1].sell_vol == 10);
    REQUIRE(r.buckets[1].buy_vol == 0);
    REQUIRE(r.buckets[1].imbalance == -10);

    // ...and its carried remainder (5) plus the buy 5 balances bucket2 to zero.
    REQUIRE(r.buckets[2].buy_vol == 5);
    REQUIRE(r.buckets[2].sell_vol == 5);
    REQUIRE(r.buckets[2].imbalance == 0);
    REQUIRE(r.buckets[2].toxicity_scaled == 0);

    // sum |imbalance| = 10 + 10 + 0 = 20 over 30 volume.
    REQUIRE(r.sum_abs_imbalance == 20);
    REQUIRE(r.vpin_scaled == 20 * offline::kVpinScale / 30);  // 666666 (floor)
}

TEST_CASE("windowed vpin matches hand-checked values and is bounded") {
    const std::vector<SignedTrade> s = sample_stream();
    const VpinResult r = offline::evaluate_vpin(s, kBucketSize);

    // Last 2 buckets: |imbalance| 6 and 10 -> sum 16 over 20 volume.
    // vpin = 16 * 1e6 / 20 = 800000.
    REQUIRE(offline::vpin_scaled_over_window(r, 2) == 800000);

    // Last 1 bucket: the fully toxic sell bucket -> 10/10 = SCALE.
    REQUIRE(offline::vpin_scaled_over_window(r, 1) == offline::kVpinScale);

    // A window >= bucket count falls back to all buckets (== res.vpin_scaled).
    REQUIRE(offline::vpin_scaled_over_window(r, 99) == r.vpin_scaled);
    REQUIRE(offline::vpin_scaled_over_window(r, 3) == 600000);

    // Every bucket toxicity and the aggregate VPIN are bounded in [0, SCALE].
    for (const VolumeBucket& b : r.buckets) {
        REQUIRE(b.toxicity_scaled >= 0);
        REQUIRE(b.toxicity_scaled <= offline::kVpinScale);
    }
    REQUIRE(r.vpin_scaled >= 0);
    REQUIRE(r.vpin_scaled <= offline::kVpinScale);
}

TEST_CASE("vpin saturates on one-sided flow and vanishes on balanced flow") {
    // Fully one-sided (all buys): every bucket maximally toxic.
    const std::vector<SignedTrade> toxic = {
        trade(1, 1000, 100, 10, +1),
        trade(2, 1100, 100, 10, +1),
    };
    const VpinResult rt = offline::evaluate_vpin(toxic, kBucketSize);
    REQUIRE(rt.n_buckets == 2);
    REQUIRE(rt.vpin_scaled == offline::kVpinScale);
    for (const VolumeBucket& b : rt.buckets) {
        REQUIRE(b.toxicity_scaled == offline::kVpinScale);
    }

    // Perfectly balanced (alternating equal buy/sell within each bucket).
    const std::vector<SignedTrade> balanced = {
        trade(1, 1000, 100, 5, +1),
        trade(2, 1100, 100, 5, -1),
        trade(3, 1200, 100, 5, +1),
        trade(4, 1300, 100, 5, -1),
    };
    const VpinResult rb = offline::evaluate_vpin(balanced, kBucketSize);
    REQUIRE(rb.n_buckets == 2);
    REQUIRE(rb.vpin_scaled == 0);
    for (const VolumeBucket& b : rb.buckets) {
        REQUIRE(b.imbalance == 0);
        REQUIRE(b.toxicity_scaled == 0);
    }
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed a diagnostic back into the kernel
};

// Positive control for the kernel-admissibility discriminator: a type that DOES
// publish `is_matching_kernel_input == true`. Together with the VPIN types
// (publishing `false`) it proves the discriminator is not vacuous.
struct KernelGate {
    static constexpr bool is_matching_kernel_input = true;
};

// A type that does not publish the flag at all is treated as non-input.
struct UnflaggedQuantity {};

}  // namespace

TEST_CASE("vpin is a post-trade diagnostic, excluded from matching-kernel inputs") {
    const std::vector<SignedTrade> s = sample_stream();
    const VpinResult r = offline::evaluate_vpin(s, kBucketSize);

    // VALUE-LEVEL gate: the result and every bucket are toxicity diagnostics,
    // never matching inputs.
    REQUIRE(offline::is_toxicity_diagnostic(r));
    REQUIRE(r.n_buckets > 0);
    for (const VolumeBucket& b : r.buckets) {
        REQUIRE(offline::is_toxicity_diagnostic(b));
    }

    // COMPILE-TIME gate: the VPIN types are excluded from the matching-input set,
    // while a positive-control kernel gate is admitted.
    static_assert(!offline::is_matching_kernel_input_v<offline::VolumeBucket>,
                  "a VPIN bucket must NOT be a matching-kernel input");
    static_assert(!offline::is_matching_kernel_input_v<offline::VpinResult>,
                  "a VPIN result must NOT be a matching-kernel input");
    static_assert(offline::is_matching_kernel_input_v<KernelGate>,
                  "the positive-control kernel gate MUST be a matching input");
    static_assert(!offline::is_matching_kernel_input_v<UnflaggedQuantity>,
                  "an unflagged quantity is not a matching-kernel input");
    // The types' own flags state they are post-trade diagnostics.
    static_assert(!offline::VolumeBucket::is_matching_kernel_input,
                  "the VPIN bucket is a post-trade diagnostic");
    static_assert(!offline::VpinResult::is_matching_kernel_input,
                  "the VPIN result is a post-trade diagnostic");
}

TEST_CASE("vpin diagnostic is read-only with no matching coupling") {
    std::vector<SignedTrade> s = sample_stream();
    const std::vector<SignedTrade> before = s;

    const VpinResult r = offline::evaluate_vpin(s, kBucketSize);
    (void)r;

    // NO mutation of the input stream: evaluation is offline/read-only and cannot
    // alter matching/fills because it never touches engine state.
    REQUIRE(s.size() == before.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        REQUIRE(s[i].seq == before[i].seq);
        REQUIRE(s[i].ts_ns == before[i].ts_ns);
        REQUIRE(s[i].price == before[i].price);
        REQUIRE(s[i].volume == before[i].volume);
        REQUIRE(s[i].side == before[i].side);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::SignedTrade>,
                  "SignedTrade must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::VolumeBucket>,
                  "VolumeBucket must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::VpinResult>,
                  "VpinResult must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::VpinToxicityReplay>,
                  "VpinToxicityReplay must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::TradeStreamView>,
                  "TradeStreamView must not be a matching input");

    // evaluate() is invocable on a const wrapper and returns a derived
    // diagnostic: compile-time evidence it cannot mutate engine/replay state.
    static_assert(
        std::is_invocable_r_v<offline::VpinResult,
                              decltype(&offline::VpinToxicityReplay::evaluate),
                              const offline::VpinToxicityReplay&, std::int64_t>,
        "evaluate must be a read-only (const) diagnostic");
    static_assert(
        std::is_same_v<decltype(offline::evaluate_vpin(s, kBucketSize)),
                       offline::VpinResult>,
        "evaluate must return a derived diagnostic, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::VolumeBucket>,
                  "VolumeBucket must be pure data");
}
