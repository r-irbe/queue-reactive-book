#include <catch2/catch_test_macros.hpp>

#include "offline/fill_probability_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic MOC-level replay/censoring fixture closing gate
//
// Proves the three properties the MOC's load-bearing off-kernel boundary needs:
//   (a) REPLAY-HASH INVARIANCE  -- same trace -> identical labels + FNV hash;
//       reorder / tamper changes the hash (order-sensitive integrity).
//   (b) CENSORING CORRECTNESS   -- orders that never fully fill are RIGHT-
//       CENSORED (cancel or end-of-data), NOT counted as fills; partial fills
//       alone never count as a fill.
//   (c) NON-FEEDBACK / READ-ONLY -- the label computation takes a const trace,
//       cannot mutate it, and the offline types carry no matching-engine
//       coupling (static_asserts + a positive-control discriminator), i.e. the
//       estimates are provably NOT a matching input.
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/fill_probability_replay.hpp` (zero `sim::` symbols).
// =============================================================================

namespace {

using offline::FillEventKind;
using offline::FillLabel;
using offline::FillOutcome;
using offline::FillSide;
using offline::FillTraceEvent;

FillTraceEvent submit(std::uint64_t seq, std::uint64_t id, FillSide side,
                      std::int64_t price, std::int64_t qty, std::int64_t qty_ahead,
                      std::uint64_t ts) {
    FillTraceEvent e;
    e.seq = seq;
    e.kind = FillEventKind::Submit;
    e.side = side;
    e.order_id = id;
    e.price = price;
    e.qty = qty;
    e.qty_ahead = qty_ahead;
    e.ts_ns = ts;
    return e;
}

FillTraceEvent level_trade(std::uint64_t seq, FillSide side, std::int64_t price,
                           std::int64_t qty, std::uint64_t ts) {
    FillTraceEvent e;
    e.seq = seq;
    e.kind = FillEventKind::LevelTrade;
    e.side = side;
    e.price = price;
    e.qty = qty;
    e.ts_ns = ts;
    return e;
}

FillTraceEvent level_cancel(std::uint64_t seq, FillSide side, std::int64_t price,
                            std::int64_t qty, bool position_known, bool ahead,
                            std::uint64_t ts) {
    FillTraceEvent e;
    e.seq = seq;
    e.kind = FillEventKind::LevelCancel;
    e.side = side;
    e.price = price;
    e.qty = qty;
    e.position_known = position_known;
    e.ahead = ahead;
    e.ts_ns = ts;
    return e;
}

FillTraceEvent own_fill(std::uint64_t seq, std::uint64_t id, std::int64_t qty,
                        std::uint64_t ts) {
    FillTraceEvent e;
    e.seq = seq;
    e.kind = FillEventKind::OwnFill;
    e.order_id = id;
    e.qty = qty;
    e.ts_ns = ts;
    return e;
}

FillTraceEvent own_cancel(std::uint64_t seq, std::uint64_t id, std::uint64_t ts) {
    FillTraceEvent e;
    e.seq = seq;
    e.kind = FillEventKind::OwnCancel;
    e.order_id = id;
    e.ts_ns = ts;
    return e;
}

// A representative trace exercising every outcome class:
//   order 1 (Bid 100): seeded 5 ahead, a level trade clears the queue, then two
//                      own fills complete it -> Filled.
//   order 2 (Bid 100): seeded 8 ahead, partially filled then own-cancelled
//                      -> RightCensoredCancel (partial fill must NOT count).
//   order 3 (Ask 101): seeded 3 ahead, never fills; trace ends
//                      -> RightCensoredEndOfData.
//   order 4 (Bid  99): an unknown-position level cancel censors its queue-ahead
//                      estimate; never fills -> RightCensoredEndOfData + censored.
std::vector<FillTraceEvent> sample_trace() {
    return {
        submit(1, 1, FillSide::Bid, 100, 10, 5, 1000),
        submit(2, 2, FillSide::Bid, 100, 6, 8, 1100),
        submit(3, 3, FillSide::Ask, 101, 4, 3, 1200),
        submit(4, 4, FillSide::Bid, 99, 7, 2, 1300),
        level_trade(5, FillSide::Bid, 100, 5, 1400),   // clears order 1's 5-ahead
        own_fill(6, 1, 4, 1500),                       // order 1: 4/10
        own_fill(7, 1, 6, 1600),                       // order 1: 10/10 -> Filled
        own_fill(8, 2, 2, 1700),                       // order 2: 2/6 (partial)
        level_cancel(9, FillSide::Bid, 99, 2, false, false, 1800),  // censor order 4
        own_cancel(10, 2, 1900),                       // order 2 -> censored-cancel
        // trace ends: orders 3 and 4 still resting -> end-of-data censoring
    };
}

const FillLabel* find_label(const offline::FillReplayResult& r, std::uint64_t id) {
    for (const FillLabel& l : r.labels)
        if (l.order_id == id) return &l;
    return nullptr;
}

}  // namespace

TEST_CASE("fill-probability replay-hash is invariant for identical traces") {
    const std::vector<FillTraceEvent> a = sample_trace();
    const std::vector<FillTraceEvent> b = sample_trace();

    const offline::FillReplayResult ra = offline::replay_fill_labels(a);
    const offline::FillReplayResult rb = offline::replay_fill_labels(b);

    // Identical traces -> identical labels and identical hash, and replaying the
    // same trace again is bit-stable.
    REQUIRE(ra.hash == rb.hash);
    REQUIRE(offline::replay_fill_labels(a).hash == ra.hash);
    REQUIRE(ra.labels.size() == rb.labels.size());
    REQUIRE(offline::fill_labels_hash(ra.labels) == ra.hash);

    // The read-only wrapper agrees with the free function.
    const offline::FillProbabilityReplay replay(a);
    REQUIRE(replay.hash() == ra.hash);
    REQUIRE(replay.replay().labels.size() == ra.labels.size());
    REQUIRE(replay.size() == a.size());

    // The empty trace is deterministic and distinct from any populated one.
    const std::vector<FillTraceEvent> empty;
    REQUIRE(offline::replay_fill_labels(empty).hash ==
            offline::replay_fill_labels(empty).hash);
    REQUIRE(offline::replay_fill_labels(empty).hash != ra.hash);
}

TEST_CASE("fill-probability replay-hash changes under reorder or tamper") {
    const std::vector<FillTraceEvent> base = sample_trace();
    const offline::ReplayHash h_base = offline::replay_fill_labels(base).hash;

    SECTION("reordering two distinct submits changes the hash") {
        // Labels are emitted in submit order, so swapping two submit events
        // reorders the label set (ordinal-folded) -> the hash changes.
        std::vector<FillTraceEvent> reordered = base;
        std::swap(reordered[0], reordered[1]);  // submit order 1 <-> order 2
        REQUIRE(offline::replay_fill_labels(reordered).hash != h_base);
    }

    SECTION("dropping the completing fill flips an outcome and changes the hash") {
        std::vector<FillTraceEvent> tampered = base;
        tampered.erase(tampered.begin() + 6);  // remove own_fill(7,1,6)
        const offline::FillReplayResult r = offline::replay_fill_labels(tampered);
        REQUIRE(r.hash != h_base);
        // order 1 no longer fully fills -> right-censored at end-of-data.
        const FillLabel* l1 = find_label(r, 1);
        REQUIRE(l1 != nullptr);
        REQUIRE(l1->outcome == FillOutcome::RightCensoredEndOfData);
        REQUIRE_FALSE(l1->is_fill());
    }

    SECTION("mutating a traded qty changes the hash") {
        std::vector<FillTraceEvent> tampered = base;
        tampered[4].qty += 1;  // level trade now over-consumes -> censors order 1
        REQUIRE(offline::replay_fill_labels(tampered).hash != h_base);
    }
}

TEST_CASE("fill-probability censoring: non-fills are right-censored, not fills") {
    const std::vector<FillTraceEvent> trace = sample_trace();
    const offline::FillReplayResult r = offline::replay_fill_labels(trace);

    REQUIRE(r.labels.size() == 4);

    // order 1: fully filled within the trace -> Filled.
    const FillLabel* l1 = find_label(r, 1);
    REQUIRE(l1 != nullptr);
    REQUIRE(l1->outcome == FillOutcome::Filled);
    REQUIRE(l1->is_fill());
    REQUIRE_FALSE(l1->is_right_censored());
    REQUIRE(l1->filled_qty == l1->original_qty);
    REQUIRE(l1->queue_ahead_qty == 0);

    // order 2: PARTIAL fill then cancel -> right-censored cancel, NOT a fill.
    const FillLabel* l2 = find_label(r, 2);
    REQUIRE(l2 != nullptr);
    REQUIRE(l2->outcome == FillOutcome::RightCensoredCancel);
    REQUIRE_FALSE(l2->is_fill());
    REQUIRE(l2->is_right_censored());
    REQUIRE(l2->filled_qty == 2);             // partial progress recorded...
    REQUIRE(l2->filled_qty < l2->original_qty);  // ...but never counted as a fill

    // order 3: never touched, trace ends -> right-censored end-of-data.
    const FillLabel* l3 = find_label(r, 3);
    REQUIRE(l3 != nullptr);
    REQUIRE(l3->outcome == FillOutcome::RightCensoredEndOfData);
    REQUIRE_FALSE(l3->is_fill());
    REQUIRE(l3->filled_qty == 0);

    // order 4: unknown-position level cancel censored its queue-ahead estimate;
    // it never fills -> right-censored, and the censor flag is set (the estimate
    // is a lower-confidence value, never ground-truth that fabricates a fill).
    const FillLabel* l4 = find_label(r, 4);
    REQUIRE(l4 != nullptr);
    REQUIRE(l4->outcome == FillOutcome::RightCensoredEndOfData);
    REQUIRE(l4->queue_ahead_censored);
    REQUIRE_FALSE(l4->is_fill());

    // Exactly one fill; the other three are right-censored. A censored order is
    // NEVER promoted to a fill -- the falsifier for "estimates are derived/
    // offline, never a matching guarantee".
    REQUIRE(r.fill_count() == 1);
    REQUIRE(r.right_censored_count() == 3);
    REQUIRE(r.fill_count() + r.right_censored_count() == r.labels.size());
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed an estimate back into the kernel
};

}  // namespace

TEST_CASE("fill-probability labels are read-only and have no matching coupling") {
    std::vector<FillTraceEvent> trace = sample_trace();
    const std::vector<FillTraceEvent> before = trace;

    const offline::FillReplayResult r = offline::replay_fill_labels(trace);
    (void)r;

    // (c) NO mutation of the input trace: replay is offline/read-only and cannot
    // alter matching/fills because it never touches engine state.
    REQUIRE(trace.size() == before.size());
    for (std::size_t i = 0; i < trace.size(); ++i) {
        REQUIRE(trace[i].seq == before[i].seq);
        REQUIRE(trace[i].order_id == before[i].order_id);
        REQUIRE(trace[i].qty == before[i].qty);
        REQUIRE(trace[i].qty_ahead == before[i].qty_ahead);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::FillLabel>,
                  "FillLabel must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FillTraceEvent>,
                  "FillTraceEvent must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FillReplayResult>,
                  "FillReplayResult must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FillProbabilityReplay>,
                  "FillProbabilityReplay must not be a matching input");

    // The replay accepts only a const trace and returns a plain value: it has no
    // mutating engine API surface (compile-time evidence of the offline boundary).
    static_assert(
        std::is_same_v<decltype(offline::replay_fill_labels(trace)),
                       offline::FillReplayResult>,
        "replay must return derived labels, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::FillLabel>,
                  "FillLabel must be pure data");
}
