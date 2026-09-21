#include <catch2/catch_test_macros.hpp>

#include "offline/markout_diagnostics_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic MOC-level markout / TCA-diagnostics replay fixture opening the
//
// Proves the properties the MOC's load-bearing leakage-hygiene boundary needs:
//   (a) REPLAY-HASH DETERMINISM + ORDER-SENSITIVITY -- same (fill, midprice)
//       stream + horizons -> identical rows + FNV hash; reorder / tamper changes
//       the hash.
//   (b) MARKOUT DOMAIN CORRECTNESS -- the signed markout
//       markout_h = side * (mid_{t+h} - fill_price) matches hand-checked values
//       at several horizons on a KNOWN trade+midprice path (integer ticks).
//   (c) HORIZON MONOTONICITY -- on a constructed drift path the markout is
//       strictly monotone in the horizon (increasing for a buy, decreasing for a
//       sell).
//   (d) SIGN CONVENTION -- a buy followed by a price rise yields a positive
//       markout; a sell followed by a rise yields a negative markout.
//   (e) LEAKAGE HYGIENE -- the markout label is STRICTLY FORWARD-LOOKING; it is
//       excluded from any at-decision-time feature set (compile-time + value
//       level), while a genuine causal feature is admitted; the markout computer
//       consumes a const future view and the offline types carry no
//       matching-engine coupling (static_asserts + positive controls).
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/markout_diagnostics_replay.hpp` (zero `sim::`
// symbols).
// =============================================================================

namespace {

using offline::Markout;
using offline::MarkoutDiagnosticsReplay;
using offline::MarkoutResult;
using offline::MarkoutTick;

MarkoutTick mid_tick(std::uint64_t seq, std::uint64_t ts, std::int64_t mid) {
    MarkoutTick t;
    t.seq = seq;
    t.ts_ns = ts;
    t.mid = mid;
    t.side = 0;
    t.fill_price = 0;
    return t;
}

MarkoutTick fill_tick(std::uint64_t seq, std::uint64_t ts, std::int64_t mid,
                      std::int8_t side, std::int64_t fill_price) {
    MarkoutTick t;
    t.seq = seq;
    t.ts_ns = ts;
    t.mid = mid;
    t.side = side;
    t.fill_price = fill_price;
    return t;
}

// A hand-built (fill, midprice) path with a BUY fill at index 1 and a SELL fill
// at index 0, so markouts at several horizons can be checked by hand.
//   index : 0    1    2    3    4    5
//   mid   : 100  101  103  102  105  104
//   side  : -1   +1    0    0    0    0
//   fillpx: 100  101   -    -    -    -
std::vector<MarkoutTick> sample_stream() {
    return {
        fill_tick(1, 1000, 100, -1, 100),  // sell @ 100
        fill_tick(2, 1100, 101, +1, 101),  // buy  @ 101
        mid_tick(3, 1200, 103),
        mid_tick(4, 1300, 102),
        mid_tick(5, 1400, 105),
        mid_tick(6, 1500, 104),
    };
}

const std::vector<int> kHorizons = {1, 2, 3, 4};

// A strictly drifting midprice path (+2 ticks/step) with a single BUY fill at
// index 0 to exercise horizon monotonicity.
//   index : 0    1    2    3    4    5
//   mid   : 100  102  104  106  108  110
std::vector<MarkoutTick> drift_stream() {
    return {
        fill_tick(1, 1000, 100, +1, 100),  // buy @ 100 on a rising path
        mid_tick(2, 1100, 102),
        mid_tick(3, 1200, 104),
        mid_tick(4, 1300, 106),
        mid_tick(5, 1400, 108),
        mid_tick(6, 1500, 110),
    };
}

}  // namespace

TEST_CASE("markout replay-hash is deterministic and order-sensitive") {
    const std::vector<MarkoutTick> a = sample_stream();
    const std::vector<MarkoutTick> b = sample_stream();

    const MarkoutResult ra = offline::evaluate_markouts(a, kHorizons);
    const MarkoutResult rb = offline::evaluate_markouts(b, kHorizons);

    // Identical streams + horizons -> identical rows and identical hash, and a
    // re-run is bit-stable.
    REQUIRE(ra.replay_hash == rb.replay_hash);
    REQUIRE(offline::evaluate_markouts(a, kHorizons).replay_hash ==
            ra.replay_hash);
    REQUIRE(ra.rows.size() == rb.rows.size());

    // The read-only wrapper agrees with the free function.
    const MarkoutDiagnosticsReplay replay(a);
    REQUIRE(replay.hash(kHorizons) == ra.replay_hash);
    REQUIRE(replay.evaluate(kHorizons).rows.size() == ra.rows.size());
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<MarkoutTick> empty;
    REQUIRE(offline::evaluate_markouts(empty, kHorizons).replay_hash ==
            offline::evaluate_markouts(empty, kHorizons).replay_hash);
    REQUIRE(offline::evaluate_markouts(empty, kHorizons).replay_hash !=
            ra.replay_hash);

    SECTION("reordering two distinct events changes the hash") {
        std::vector<MarkoutTick> reordered = a;
        std::swap(reordered[2], reordered[4]);
        REQUIRE(offline::evaluate_markouts(reordered, kHorizons).replay_hash !=
                ra.replay_hash);
    }

    SECTION("tampering with a future midprice changes the hash") {
        std::vector<MarkoutTick> tampered = a;
        tampered[3].mid += 1;
        REQUIRE(offline::evaluate_markouts(tampered, kHorizons).replay_hash !=
                ra.replay_hash);
    }

    SECTION("changing the horizon list changes the hash") {
        const std::vector<int> fewer = {1, 2};
        REQUIRE(offline::evaluate_markouts(a, fewer).replay_hash !=
                ra.replay_hash);
    }
}

TEST_CASE("markout matches a hand-checked series at several horizons") {
    const std::vector<MarkoutTick> s = sample_stream();
    const MarkoutResult r = offline::evaluate_markouts(s, kHorizons);

    // Fills: sell@100 (index 0), buy@101 (index 1). 2 fills.
    REQUIRE(r.n_fills == 2);

    // Rows for the SELL fill at index 0 (full horizons 1..4 -> mids[1..4]):
    //   h=1: -1*(101-100) = -1
    //   h=2: -1*(103-100) = -3
    //   h=3: -1*(102-100) = -2
    //   h=4: -1*(105-100) = -5
    // Rows for the BUY fill at index 1 (full horizons 1..4 -> mids[2..5]):
    //   h=1: +1*(103-101) = +2
    //   h=2: +1*(102-101) = +1
    //   h=3: +1*(105-101) = +4
    //   h=4: +1*(104-101) = +3
    REQUIRE(r.n_rows == 8);
    REQUIRE(r.rows.size() == 8);

    // --- sell fill (index 0) rows ---
    REQUIRE(r.rows[0].fill_seq == 1);
    REQUIRE(r.rows[0].side == -1);
    REQUIRE(r.rows[0].horizon == 1);
    REQUIRE(r.rows[0].mid_future == 101);
    REQUIRE(r.rows[0].markout == -1);
    REQUIRE(r.rows[1].horizon == 2);
    REQUIRE(r.rows[1].markout == -3);
    REQUIRE(r.rows[2].horizon == 3);
    REQUIRE(r.rows[2].markout == -2);
    REQUIRE(r.rows[3].horizon == 4);
    REQUIRE(r.rows[3].markout == -5);

    // --- buy fill (index 1) rows ---
    REQUIRE(r.rows[4].fill_seq == 2);
    REQUIRE(r.rows[4].side == 1);
    REQUIRE(r.rows[4].horizon == 1);
    REQUIRE(r.rows[4].mid_future == 103);
    REQUIRE(r.rows[4].markout == 2);
    REQUIRE(r.rows[5].horizon == 2);
    REQUIRE(r.rows[5].markout == 1);
    REQUIRE(r.rows[6].horizon == 3);
    REQUIRE(r.rows[6].markout == 4);
    REQUIRE(r.rows[7].horizon == 4);
    REQUIRE(r.rows[7].markout == 3);

    // Sum check: (-1-3-2-5) + (2+1+4+3) = -11 + 10 = -1.
    REQUIRE(r.markout_sum == -1);
}

TEST_CASE("markout sign convention: buy-then-rise is positive, sell-then-rise is negative") {
    const std::vector<MarkoutTick> s = sample_stream();
    const MarkoutResult r = offline::evaluate_markouts(s, kHorizons);

    // Buy @ 101 with the next mid rising to 103 => positive markout.
    REQUIRE(r.rows[4].side == 1);
    REQUIRE(r.rows[4].mid_future > r.rows[4].fill_price);
    REQUIRE(r.rows[4].markout > 0);

    // Sell @ 100 with the next mid rising to 101 => negative markout (the seller
    // is worse off when the price rises after the trade).
    REQUIRE(r.rows[0].side == -1);
    REQUIRE(r.rows[0].mid_future > r.rows[0].fill_price);
    REQUIRE(r.rows[0].markout < 0);
}

TEST_CASE("markout is strictly monotone in the horizon on a drift path") {
    const std::vector<MarkoutTick> up = drift_stream();
    const MarkoutResult ru = offline::evaluate_markouts(up, kHorizons);

    // Single buy fill at index 0, horizons 1..4 -> mids 102,104,106,108.
    // markout_h = +1*(mid_{h} - 100) = 2,4,6,8 : strictly increasing.
    REQUIRE(ru.n_rows == 4);
    for (std::size_t i = 1; i < ru.rows.size(); ++i) {
        REQUIRE(ru.rows[i].horizon > ru.rows[i - 1].horizon);
        REQUIRE(ru.rows[i].markout > ru.rows[i - 1].markout);
    }
    REQUIRE(ru.rows[0].markout == 2);
    REQUIRE(ru.rows[3].markout == 8);

    // Same drift, but a SELL fill -> markout strictly DECREASING in the horizon.
    std::vector<MarkoutTick> down = drift_stream();
    down[0].side = -1;  // flip the fill to a sell at the same price
    const MarkoutResult rd = offline::evaluate_markouts(down, kHorizons);
    REQUIRE(rd.n_rows == 4);
    for (std::size_t i = 1; i < rd.rows.size(); ++i) {
        REQUIRE(rd.rows[i].markout < rd.rows[i - 1].markout);
    }
    REQUIRE(rd.rows[0].markout == -2);
    REQUIRE(rd.rows[3].markout == -8);
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed a diagnostic back into the kernel
};

// Positive control for the decision-time discriminator: a type that DOES NOT
// publish the flag at all is non-admissible; the genuine causal feature type
// (publishing `true`) is admissible. Together with the markout label (publishing
// `false`) these prove the discriminator is not vacuous.
struct UnflaggedQuantity {};

}  // namespace

TEST_CASE("markout is a forward-looking label, excluded from decision-time features") {
    const std::vector<MarkoutTick> s = sample_stream();
    const MarkoutResult r = offline::evaluate_markouts(s, kHorizons);

    // VALUE-LEVEL gate: every markout reads the future, so using any of them as
    // an at-decision-time feature would be look-ahead leakage.
    REQUIRE(r.n_rows > 0);
    for (const Markout& row : r.rows) {
        REQUIRE(row.reads_future);
        REQUIRE(offline::would_leak_as_decision_feature(row));
    }

    // COMPILE-TIME gate: the markout LABEL type is excluded from the
    // decision-time feature set, while a genuine causal feature is admitted.
    static_assert(!offline::is_decision_time_feature_v<offline::Markout>,
                  "markout must NOT be admissible as a decision-time feature");
    static_assert(
        offline::is_decision_time_feature_v<offline::TrailingReturnFeature>,
        "a causal at-or-before-t feature MUST be admissible at decision time");
    static_assert(!offline::is_decision_time_feature_v<UnflaggedQuantity>,
                  "an unflagged quantity is not a decision-time feature");
    // The label's own flag states it is not available at decision time.
    static_assert(!offline::Markout::available_at_decision_time,
                  "the markout label is a post-hoc, future-reading quantity");
}

TEST_CASE("markout diagnostic is read-only with no matching coupling") {
    std::vector<MarkoutTick> s = sample_stream();
    const std::vector<MarkoutTick> before = s;

    const MarkoutResult r = offline::evaluate_markouts(s, kHorizons);
    (void)r;

    // NO mutation of the input stream: evaluation is offline/read-only and cannot
    // alter matching/fills because it never touches engine state.
    REQUIRE(s.size() == before.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        REQUIRE(s[i].seq == before[i].seq);
        REQUIRE(s[i].ts_ns == before[i].ts_ns);
        REQUIRE(s[i].mid == before[i].mid);
        REQUIRE(s[i].side == before[i].side);
        REQUIRE(s[i].fill_price == before[i].fill_price);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::MarkoutTick>,
                  "MarkoutTick must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::Markout>,
                  "Markout must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::MarkoutResult>,
                  "MarkoutResult must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<offline::MarkoutDiagnosticsReplay>,
        "MarkoutDiagnosticsReplay must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FutureMidView>,
                  "FutureMidView must not be a matching input");

    // The markout computer consumes a CONST future view and returns a plain
    // value: compile-time evidence it cannot mutate engine/replay state.
    static_assert(
        std::is_invocable_r_v<std::int64_t, decltype(&offline::compute_markout),
                              const offline::FutureMidView&, std::size_t,
                              std::int8_t, std::int64_t, int>,
        "compute_markout must read a const future view (read-only)");
    static_assert(
        std::is_same_v<decltype(offline::evaluate_markouts(s, kHorizons)),
                       offline::MarkoutResult>,
        "evaluate must return a derived diagnostic, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::Markout>,
                  "Markout must be pure data");
}
