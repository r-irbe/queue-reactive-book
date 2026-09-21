#include <catch2/catch_test_macros.hpp>

#include "offline/causal_leakage_hygiene_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic MOC-level causal-leakage-hygiene replay fixture closing gate
//
// Proves the properties the MOC's load-bearing off-kernel hygiene boundary needs:
//   (a) REPLAY-HASH DETERMINISM + ORDER-SENSITIVITY -- same stream + config ->
//       identical rows + FNV hash; reorder / tamper changes the hash.
//   (b) CAUSAL vs LEAKY DISCRIMINATOR -- a "leaky" config that peeks at the
//       future produces a DIFFERENT, detectably-biased association than the
//       causal config, and the leaky run is FLAGGED (the hygiene assertion),
//       while the causal run is admissible.
//   (c) MARKOUT CORRECTNESS -- the forward markout label and the at-or-before-t
//       causal feature match a hand-built series exactly.
//   (d) NON-FEEDBACK / READ-ONLY -- the evaluation takes a const stream, cannot
//       mutate it, and the offline types carry no matching-engine coupling
//       (static_asserts + a positive-control discriminator), i.e. the diagnostic
//       is provably NOT a matching input.
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/causal_leakage_hygiene_replay.hpp` (zero `sim::`
// symbols).
// =============================================================================

namespace {

using offline::CausalConfig;
using offline::CausalEvent;
using offline::CausalHygieneResult;
using offline::MarkoutRow;

CausalEvent ev(std::uint64_t seq, std::uint64_t ts, std::int64_t price) {
    CausalEvent e;
    e.seq = seq;
    e.ts_ns = ts;
    e.price = price;
    return e;
}

// A hand-built price series with mixed-sign trailing/forward moves so the causal
// association is NON-trivial, while the future-peeking (leaky) feature trivially
// equals the forward label.
//   index : 0    1    2    3    4    5
//   price : 100  101  103  102  105  104
//   diffs :   +1   +2   -1   +3   -1
std::vector<CausalEvent> sample_stream() {
    return {
        ev(1, 1000, 100),
        ev(2, 1100, 101),
        ev(3, 1200, 103),
        ev(4, 1300, 102),
        ev(5, 1400, 105),
        ev(6, 1500, 104),
    };
}

CausalConfig causal_cfg() {
    CausalConfig c;
    c.horizon = 1;
    c.lookback = 1;
    c.leaky_future_peek = false;
    return c;
}

CausalConfig leaky_cfg() {
    CausalConfig c;
    c.horizon = 1;
    c.lookback = 1;
    c.leaky_future_peek = true;
    return c;
}

}  // namespace

TEST_CASE("causal-hygiene replay-hash is deterministic and order-sensitive") {
    const std::vector<CausalEvent> a = sample_stream();
    const std::vector<CausalEvent> b = sample_stream();
    const CausalConfig cfg = causal_cfg();

    const CausalHygieneResult ra = offline::evaluate_causal_hygiene(a, cfg);
    const CausalHygieneResult rb = offline::evaluate_causal_hygiene(b, cfg);

    // Identical streams + config -> identical rows and identical hash, and a
    // re-run is bit-stable.
    REQUIRE(ra.replay_hash == rb.replay_hash);
    REQUIRE(offline::evaluate_causal_hygiene(a, cfg).replay_hash == ra.replay_hash);
    REQUIRE(ra.rows.size() == rb.rows.size());

    // The read-only wrapper agrees with the free function.
    const offline::CausalLeakageHygieneReplay replay(a);
    REQUIRE(replay.hash(cfg) == ra.replay_hash);
    REQUIRE(replay.evaluate(cfg).rows.size() == ra.rows.size());
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<CausalEvent> empty;
    REQUIRE(offline::evaluate_causal_hygiene(empty, cfg).replay_hash ==
            offline::evaluate_causal_hygiene(empty, cfg).replay_hash);
    REQUIRE(offline::evaluate_causal_hygiene(empty, cfg).replay_hash !=
            ra.replay_hash);

    SECTION("reordering two distinct events changes the hash") {
        std::vector<CausalEvent> reordered = a;
        std::swap(reordered[1], reordered[3]);
        REQUIRE(offline::evaluate_causal_hygiene(reordered, cfg).replay_hash !=
                ra.replay_hash);
    }

    SECTION("tampering with a price changes the hash") {
        std::vector<CausalEvent> tampered = a;
        tampered[2].price += 1;
        REQUIRE(offline::evaluate_causal_hygiene(tampered, cfg).replay_hash !=
                ra.replay_hash);
    }
}

TEST_CASE("causal-hygiene markout/feature computation matches a hand series") {
    const std::vector<CausalEvent> s = sample_stream();
    const CausalHygieneResult r = offline::evaluate_causal_hygiene(s, causal_cfg());

    // Causal config (lookback=1, horizon=1): rows for decision indices t=1..4
    // (need price[t-1] and price[t+1]).
    REQUIRE(r.n_rows == 4);
    REQUIRE(r.rows.size() == 4);

    // t=1: feature = 101-100 = +1 ; label = 103-101 = +2
    REQUIRE(r.rows[0].event_seq == 2);
    REQUIRE(r.rows[0].feature == 1);
    REQUIRE(r.rows[0].label == 2);
    REQUIRE(r.rows[0].feature_admissible);
    // t=2: feature = 103-101 = +2 ; label = 102-103 = -1
    REQUIRE(r.rows[1].feature == 2);
    REQUIRE(r.rows[1].label == -1);
    // t=3: feature = 102-103 = -1 ; label = 105-102 = +3
    REQUIRE(r.rows[2].feature == -1);
    REQUIRE(r.rows[2].label == 3);
    // t=4: feature = 105-102 = +3 ; label = 104-105 = -1
    REQUIRE(r.rows[3].feature == 3);
    REQUIRE(r.rows[3].label == -1);

    // Association: only t=1 agrees in sign; t=2,3,4 disagree.
    REQUIRE(r.sign_agree == 1);
    REQUIRE(r.sign_disagree == 3);
    // assoc_score = (1*2) + (2*-1) + (-1*3) + (3*-1) = 2 - 2 - 3 - 3 = -6.
    REQUIRE(r.assoc_score == -6);

    // Every feature window is at-or-before t -> no leakage, admissible.
    REQUIRE_FALSE(r.leakage_detected);
    REQUIRE(r.is_admissible());
}

TEST_CASE("causal-hygiene flags the leaky future-peeking config (the gate)") {
    const std::vector<CausalEvent> s = sample_stream();

    const CausalHygieneResult causal =
        offline::evaluate_causal_hygiene(s, causal_cfg());
    const CausalHygieneResult leaky =
        offline::evaluate_causal_hygiene(s, leaky_cfg());

    // The hygiene assertion: the leaky run is FLAGGED, the causal run is not.
    REQUIRE_FALSE(causal.leakage_detected);
    REQUIRE(causal.is_admissible());
    REQUIRE(leaky.leakage_detected);
    REQUIRE_FALSE(leaky.is_admissible());

    // The leaky config peeks at the FUTURE (feature window shifted to t+1), so
    // every row's feature is marked inadmissible.
    REQUIRE(leaky.n_rows == 5);  // t=0..4
    for (const MarkoutRow& row : leaky.rows) {
        REQUIRE_FALSE(row.feature_admissible);
    }
    // The future-peeking feature trivially EQUALS the forward label, producing a
    // detectably-biased, too-perfect association (all signs agree, score >> 0).
    REQUIRE(leaky.sign_agree == leaky.n_rows);
    REQUIRE(leaky.sign_disagree == 0);
    // assoc_score = 1 + 4 + 1 + 9 + 1 = 16 (a sum of squares -> always >= 0).
    REQUIRE(leaky.assoc_score == 16);
    for (const MarkoutRow& row : leaky.rows) {
        REQUIRE(row.feature == row.label);
    }

    // Causal vs leaky are DETECTABLY different: different results AND a different
    // replay hash. This is the leakage discriminator the MOC demands.
    REQUIRE(causal.assoc_score != leaky.assoc_score);
    REQUIRE(causal.n_rows != leaky.n_rows);
    REQUIRE(causal.replay_hash != leaky.replay_hash);
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed a diagnostic back into the kernel
};

}  // namespace

TEST_CASE("causal-hygiene diagnostic is read-only with no matching coupling") {
    std::vector<CausalEvent> s = sample_stream();
    const std::vector<CausalEvent> before = s;

    const CausalHygieneResult r =
        offline::evaluate_causal_hygiene(s, causal_cfg());
    (void)r;

    // NO mutation of the input stream: evaluation is offline/read-only and cannot
    // alter matching/fills because it never touches engine state.
    REQUIRE(s.size() == before.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        REQUIRE(s[i].seq == before[i].seq);
        REQUIRE(s[i].ts_ns == before[i].ts_ns);
        REQUIRE(s[i].price == before[i].price);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::CausalEvent>,
                  "CausalEvent must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::MarkoutRow>,
                  "MarkoutRow must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::CausalHygieneResult>,
                  "CausalHygieneResult must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<offline::CausalLeakageHygieneReplay>,
        "CausalLeakageHygieneReplay must not be a matching input");

    // The evaluation accepts only a const stream and returns a plain value: it
    // has no mutating engine API surface (compile-time evidence of the boundary).
    static_assert(
        std::is_same_v<decltype(offline::evaluate_causal_hygiene(s, causal_cfg())),
                       offline::CausalHygieneResult>,
        "evaluate must return a derived diagnostic, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::MarkoutRow>,
                  "MarkoutRow must be pure data");
}
