#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/adverse_selection_spread_decomposition_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture opening AND closing the
// `needs-code-cross-validation` (worked-fixture / diagnostics-on-off
// (the channel asks for a worked, replay-neutral example where an
// adverse-selection / toxicity / spread-decomposition export is reduced from a
// recorded trade-and-quote sample without changing a replay hash -- supplied here
// as the effective/realized/adverse-selection half-spread decomposition plus a
// single-bucket VPIN-style order-flow-toxicity ratio).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash is deterministic AND order-sensitive: the same ordered
//       sample -> the same hash; a reordering of the SAME multiset of fills ->
//       a different hash (even though the buy/sell split, volumes, the spread
//       decomposition sums, and the toxicity ratio are order-invariant);
//   (2) the diagnostics are correct on hand-built cases with KNOWN integer-tick
//       values: the three half-spread sums satisfy the EXACT identity
//       effective == realized + adverse_selection, the toxicity ratio is the
//       hand-checked |signed_volume| / total_volume, and the cumulative
//       adverse-selection path peak matches the order-sensitive running sum;
//   (3) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// HONEST SCOPE: this validates the diagnostic / DECOMPOSITION discipline (the
// effective half-spread splits exactly into a realized half-spread plus an
// adverse-selection component, evaluated post hoc) -- it is implementation
// evidence only. It does NOT prove any structural adverse-selection
// (Glosten-Milgrom) or toxicity (VPIN) model is the true market mechanism; those
// are empirically motivated but contested measurement conventions, not theorems.
// Nothing here feeds back into matching/replay.
//
// Depends only on the self-contained header
// `offline/adverse_selection_spread_decomposition_replay.hpp` (std-includes
// only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::AdverseSelectionDiagnostics;
using offline::AdverseSelectionReplay;
using offline::SpreadFill;

namespace analytic = offline::analytic;

// Build an ordered sample from (direction, trade_price, mid_at_trade, mid_after,
// size) tuples; seq = position so the order is explicit.
struct FillSpec {
    std::int64_t direction;
    std::int64_t trade_price;
    std::int64_t mid_at_trade;
    std::int64_t mid_after;
    std::int64_t size;
};

std::vector<SpreadFill> make_sample(const std::vector<FillSpec>& specs) {
    std::vector<SpreadFill> out;
    out.reserve(specs.size());
    std::uint64_t seq = 0;
    for (const FillSpec& s : specs) {
        out.push_back(SpreadFill{seq, s.direction, s.trade_price, s.mid_at_trade,
                                 s.mid_after, s.size});
        ++seq;
    }
    return out;
}

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (3) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<AdverseSelectionDiagnostics>,
              "AdverseSelectionDiagnostics must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<SpreadFill>,
              "SpreadFill must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<AdverseSelectionDiagnostics,
                          decltype(&AdverseSelectionReplay::compute),
                          const AdverseSelectionReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline adverse-selection: EXACT integer sample aggregates "
          "(buy/sell split, total + signed volume)",
          "[offline][adverse-selection][aggregates]") {
    // Three fills: buy 10, sell 20, buy 30 -> 2 buys, 1 sell, total 60,
    // signed = +10 - 20 + 30 = +20.
    const auto sample = make_sample({{+1, 102, 100, 101, 10},
                                     {-1, 98, 100, 99, 20},
                                     {+1, 105, 104, 106, 30}});
    const AdverseSelectionDiagnostics d =
        AdverseSelectionReplay(sample).compute();

    REQUIRE(d.n == 3);
    REQUIRE(d.num_buy_initiated == 2);
    REQUIRE(d.num_sell_initiated == 1);
    REQUIRE(d.total_volume == 60);
    REQUIRE(d.signed_volume == 20);
}

TEST_CASE("offline adverse-selection: effective/realized/adverse-selection "
          "decomposition sums match HAND-CHECKED integer ticks and satisfy the "
          "EXACT identity effective == realized + adverse_selection",
          "[offline][adverse-selection][decomposition][identity]") {
    // Hand-checked per fill (d, P, M0, M_tau):
    //   buy : eff=+1*(102-100)=2, real=+1*(102-101)=1, adv=+1*(101-100)=1
    //   sell: eff=-1*(98-100)=2,  real=-1*(98-99)=1,    adv=-1*(99-100)=1
    //   buy : eff=+1*(105-104)=1, real=+1*(105-106)=-1, adv=+1*(106-104)=2
    // sums: eff=2+2+1=5, real=1+1-1=1, adv=1+1+2=4 -> 5 == 1 + 4.
    const auto sample = make_sample({{+1, 102, 100, 101, 10},
                                     {-1, 98, 100, 99, 20},
                                     {+1, 105, 104, 106, 30}});
    const AdverseSelectionDiagnostics d =
        AdverseSelectionReplay(sample).compute();

    REQUIRE(d.sum_effective_half == 5);
    REQUIRE(d.sum_realized_half == 1);
    REQUIRE(d.sum_adverse_selection == 4);
    // EXACT integer decomposition identity.
    REQUIRE(d.sum_effective_half ==
            d.sum_realized_half + d.sum_adverse_selection);

    // Means (order-invariant): 5/3, 1/3, 4/3.
    REQUIRE(d.mean_effective_half == Approx(5.0 / 3.0).epsilon(1e-12));
    REQUIRE(d.mean_realized_half == Approx(1.0 / 3.0).epsilon(1e-12));
    REQUIRE(d.mean_adverse_selection == Approx(4.0 / 3.0).epsilon(1e-12));
}

TEST_CASE("offline adverse-selection: the per-observation decomposition identity "
          "holds tick-for-tick for arbitrary (direction, P, M0, M_tau)",
          "[offline][adverse-selection][identity][analytic]") {
    // The algebra d*(P-M0) == d*(P-Mtau) + d*(Mtau-M0) holds exactly in integers
    // for every direction including the degenerate sign-0 case.
    struct Case {
        std::int64_t d, P, M0, Mt;
    };
    for (const Case& c : std::vector<Case>{{+1, 102, 100, 101},
                                           {-1, 98, 100, 99},
                                           {+1, 105, 104, 106},
                                           {-1, 50, 55, 40},
                                           {0, 77, 70, 90},
                                           {+1, 100, 100, 100}}) {
        REQUIRE(analytic::decomposition_identity_holds(c.d, c.P, c.M0, c.Mt));
        const std::int64_t eff =
            analytic::effective_half_spread(c.d, c.P, c.M0);
        const std::int64_t real =
            analytic::realized_half_spread(c.d, c.P, c.Mt);
        const std::int64_t adv = analytic::adverse_selection(c.d, c.M0, c.Mt);
        REQUIRE(eff == real + adv);
    }

    // Spot hand-checked values for the sell case (d=-1, P=98, M0=100, Mt=99):
    REQUIRE(analytic::effective_half_spread(-1, 98, 100) == 2);
    REQUIRE(analytic::realized_half_spread(-1, 98, 99) == 1);
    REQUIRE(analytic::adverse_selection(-1, 100, 99) == 1);
}

TEST_CASE("offline adverse-selection: single-bucket VPIN-style order-flow "
          "toxicity ratio is the HAND-CHECKED |signed_volume| / total_volume",
          "[offline][adverse-selection][toxicity][vpin]") {
    // buy 10, sell 20, buy 30: signed = +20, total = 60 -> toxicity = 20/60.
    const auto sample = make_sample({{+1, 102, 100, 101, 10},
                                     {-1, 98, 100, 99, 20},
                                     {+1, 105, 104, 106, 30}});
    const AdverseSelectionDiagnostics d =
        AdverseSelectionReplay(sample).compute();
    REQUIRE(d.toxicity_valid);
    REQUIRE(d.orderflow_toxicity == Approx(20.0 / 60.0).epsilon(1e-12));

    // Perfectly balanced flow -> toxicity 0; perfectly one-sided flow -> 1.
    const auto balanced =
        make_sample({{+1, 101, 100, 100, 40}, {-1, 99, 100, 100, 40}});
    REQUIRE(AdverseSelectionReplay(balanced).compute().orderflow_toxicity ==
            Approx(0.0).margin(1e-12));
    const auto one_sided =
        make_sample({{+1, 101, 100, 100, 15}, {+1, 102, 100, 100, 25}});
    REQUIRE(AdverseSelectionReplay(one_sided).compute().orderflow_toxicity ==
            Approx(1.0).epsilon(1e-12));
}

TEST_CASE("offline adverse-selection: decomposition sums + toxicity are "
          "order-invariant while the cumulative adverse-selection path peak and "
          "the hash are order-sensitive",
          "[offline][adverse-selection][order][determinism]") {
    // Same multiset of three fills in two orders. The decomposition sums, volumes,
    // and toxicity are order-invariant; the sequential cumulative
    // adverse-selection peak and the hash are order-sensitive. The per-fill
    // adverse-selection components are {+10, +10, -16}, summing to a small final
    // excursion (4) so the worst intermediate peak depends on order.
    //   X: d=+1, M0=100, M_tau=110 -> adv = +10
    //   Y: d=+1, M0=200, M_tau=210 -> adv = +10
    //   Z: d=+1, M0=300, M_tau=284 -> adv = -16
    const auto a = make_sample({{+1, 101, 100, 110, 10},
                                {+1, 201, 200, 210, 10},
                                {+1, 301, 300, 284, 10}});
    const auto b = make_sample({{+1, 301, 300, 284, 10},
                                {+1, 101, 100, 110, 10},
                                {+1, 201, 200, 210, 10}});

    const AdverseSelectionDiagnostics ra = AdverseSelectionReplay(a).compute();
    const AdverseSelectionDiagnostics rb = AdverseSelectionReplay(b).compute();

    // Order-INVARIANT decomposition + aggregates + toxicity.
    REQUIRE(ra.sum_effective_half == rb.sum_effective_half);
    REQUIRE(ra.sum_realized_half == rb.sum_realized_half);
    REQUIRE(ra.sum_adverse_selection == rb.sum_adverse_selection);
    REQUIRE(ra.sum_adverse_selection == 4);  // 10 + 10 - 16
    REQUIRE(ra.total_volume == rb.total_volume);
    REQUIRE(ra.signed_volume == rb.signed_volume);
    REQUIRE(ra.orderflow_toxicity == Approx(rb.orderflow_toxicity).epsilon(1e-12));

    // Order-SENSITIVE cumulative adverse-selection path peak:
    //   order A: running 10 -> 20 -> 4, peak = 20.
    //   order B: running -16 -> -6 -> 4, peak = 16.
    REQUIRE(ra.peak_abs_cumulative_adverse_selection == 20);
    REQUIRE(rb.peak_abs_cumulative_adverse_selection == 16);

    // Therefore the replay hash MUST differ even though the decomposition matches.
    REQUIRE(ra.replay_hash != rb.replay_hash);
}

TEST_CASE("offline adverse-selection: replay-hash determinism",
          "[offline][adverse-selection][determinism]") {
    const auto sample = make_sample({{+1, 102, 100, 101, 10},
                                     {-1, 98, 100, 99, 20},
                                     {+1, 105, 104, 106, 30}});

    // Same ordered input -> identical hash, every time and across instances.
    const AdverseSelectionReplay x(sample);
    const AdverseSelectionReplay y(sample);
    const std::uint64_t h1 = x.compute().replay_hash;
    const std::uint64_t h2 = x.compute().replay_hash;  // idempotent
    const std::uint64_t h3 = y.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // An extra fill changes the hash, and the count fold prevents a prefix from
    // colliding.
    const auto extended = make_sample({{+1, 102, 100, 101, 10},
                                       {-1, 98, 100, 99, 20},
                                       {+1, 105, 104, 106, 30},
                                       {+1, 100, 100, 100, 5}});
    REQUIRE(AdverseSelectionReplay(extended).hash() != h1);

    // Changing a single midpoint changes the hash.
    const auto perturbed = make_sample({{+1, 102, 100, 102, 10},
                                        {-1, 98, 100, 99, 20},
                                        {+1, 105, 104, 106, 30}});
    REQUIRE(AdverseSelectionReplay(perturbed).hash() != h1);
}

TEST_CASE("offline adverse-selection: degenerate and empty inputs are "
          "well-defined",
          "[offline][adverse-selection][determinism]") {
    // A single zero-direction fill: it contributes no volume and zero spread
    // components (sign 0), so toxicity is invalid but the diagnostic is still
    // well-defined with a stable hash.
    const auto flat = make_sample({{0, 105, 100, 110, 50}});
    const AdverseSelectionDiagnostics rf =
        AdverseSelectionReplay(flat).compute();
    REQUIRE(rf.num_buy_initiated == 0);
    REQUIRE(rf.num_sell_initiated == 0);
    REQUIRE(rf.total_volume == 0);
    REQUIRE(rf.signed_volume == 0);
    REQUIRE(rf.sum_effective_half == 0);
    REQUIRE(rf.sum_realized_half == 0);
    REQUIRE(rf.sum_adverse_selection == 0);
    REQUIRE_FALSE(rf.toxicity_valid);
    REQUIRE(rf.peak_abs_cumulative_adverse_selection == 0);
    REQUIRE(rf.replay_hash != 0);

    // Empty -> well-defined zeros, invalid toxicity, but a stable deterministic
    // hash.
    const std::vector<SpreadFill> none;
    const AdverseSelectionReplay empty(none);
    REQUIRE(empty.empty());
    const AdverseSelectionDiagnostics e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.num_buy_initiated == 0);
    REQUIRE_FALSE(e.toxicity_valid);
    REQUIRE(e.peak_abs_cumulative_adverse_selection == 0);
    REQUIRE(e.replay_hash == empty.hash());  // deterministic
}
