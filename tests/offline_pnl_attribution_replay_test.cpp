#include <catch2/catch_test_macros.hpp>

#include "offline/pnl_attribution_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the `needs-code-validation`
// (the channel asked for a worked, replay-neutral example in which an ordered
// fill stream is decomposed into the MOC's separately-reconciled accounting
// fields -- realized cash, remaining position, fee/rebate sign, terminal-mark
// unrealized, and carry/financing -- with golden rows for flat, long, short,
// partial-fill, and terminal-position cases).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the mark-to-market net PnL, the fee/rebate ledger, the carry total, and
//       the final position are EXACT integers and ORDER-INVARIANT;
//       the SAME multiset of fills changes realized and unrealized) yet ALWAYS
//       reconciles: realized + unrealized == gross mark-to-market;
//   (3) golden rows hold for flat, long, short, partial-fill, and
//       terminal-position cases with hand-checked integer-tick arithmetic;
//   (4) the replay hash is deterministic AND order-sensitive;
//   (5) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// Depends only on the self-contained header
// `offline/pnl_attribution_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using offline::Fill;
using offline::OpenLot;
using offline::PnlAttribution;
using offline::PnlAttributionReplay;

namespace analytic = offline::analytic;

// Build an ordered fill series from (quantity, price, fee, rebate, carry)
// tuples; seq = position so the order is explicit.
struct FillSpec {
    std::int64_t quantity;
    std::int64_t price;
    std::int64_t fee_paid;
    std::int64_t rebate_recv;
    std::int64_t carry_charge;
};

std::vector<Fill> make_fills(const std::vector<FillSpec>& specs) {
    std::vector<Fill> out;
    out.reserve(specs.size());
    std::uint64_t seq = 0;
    for (const FillSpec& s : specs) {
        out.push_back(Fill{seq, s.quantity, s.price, s.fee_paid, s.rebate_recv,
                           s.carry_charge});
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

// (5) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<PnlAttribution>,
              "PnlAttribution must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<Fill>,
              "Fill must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<PnlAttribution,
                          decltype(&PnlAttributionReplay::compute),
                          const PnlAttributionReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline pnl: FLAT round-trip golden row (fully closed, zero residue)",
          "[offline][pnl][accounting][golden][flat]") {
    // Buy 5 @ 100, Sell 5 @ 130. Mark irrelevant (flat at end).
    //   accumulated_cash = -5*100 + 5*130 = -500 + 650 = 150
    //   final_position = 0 -> position_mark_value = 0 -> gross_mtm = 150
    //   FIFO realized = (130-100)*5 = 150; unrealized = 0; reconciles to 150.
    const auto fills = make_fills({{5, 100, 0, 0, 0}, {-5, 130, 0, 0, 0}});
    const PnlAttribution r = PnlAttributionReplay(fills, 999).compute();

    REQUIRE(r.final_position == 0);
    REQUIRE(r.accumulated_cash == 150);
    REQUIRE(r.gross_mtm_pnl == 150);
    REQUIRE(r.realized_pnl == 150);
    REQUIRE(r.unrealized_pnl == 0);
    REQUIRE(r.open_lot_count == 0);
    // Reconciliation identity holds.
    REQUIRE(r.realized_pnl + r.unrealized_pnl == r.gross_mtm_pnl);
    REQUIRE(r.net_pnl == 150);  // no fees/rebates/carry
}

TEST_CASE("offline pnl: LONG open-position golden row (unrealized at terminal mark)",
          "[offline][pnl][accounting][golden][long]") {
    // Buy 3 @ 100, Buy 2 @ 110, terminal mark 130. Position stays long 5.
    //   accumulated_cash = -3*100 - 2*110 = -300 - 220 = -520
    //   position_mark_value = 5*130 = 650; gross_mtm = -520 + 650 = 130
    //   realized = 0 (no closes); unrealized over lots:
    //     (130-100)*3 + (130-110)*2 = 90 + 40 = 130; reconciles.
    const auto fills = make_fills({{3, 100, 0, 0, 0}, {2, 110, 0, 0, 0}});
    const PnlAttribution r = PnlAttributionReplay(fills, 130).compute();

    REQUIRE(r.final_position == 5);
    REQUIRE(r.accumulated_cash == -520);
    REQUIRE(r.position_mark_value == 650);
    REQUIRE(r.gross_mtm_pnl == 130);
    REQUIRE(r.realized_pnl == 0);
    REQUIRE(r.unrealized_pnl == 130);
    REQUIRE(r.open_lot_count == 2);
    REQUIRE(r.realized_pnl + r.unrealized_pnl == r.gross_mtm_pnl);
}

TEST_CASE("offline pnl: SHORT open-position golden row (sign-correct unrealized)",
          "[offline][pnl][accounting][golden][short]") {
    // Sell 4 @ 200 (open short), terminal mark 180 (favourable for short).
    //   accumulated_cash = -(-4)*200 = +800
    //   position_mark_value = (-4)*180 = -720; gross_mtm = 800 - 720 = 80
    //   realized = 0; unrealized for short lot = sign(-)*(180-200)*4
    //     = -1 * (-20) * 4 = +80; reconciles.
    const auto fills = make_fills({{-4, 200, 0, 0, 0}});
    const PnlAttribution r = PnlAttributionReplay(fills, 180).compute();

    REQUIRE(r.final_position == -4);
    REQUIRE(r.accumulated_cash == 800);
    REQUIRE(r.position_mark_value == -720);
    REQUIRE(r.gross_mtm_pnl == 80);
    REQUIRE(r.unrealized_pnl == 80);
    REQUIRE(r.open_lot_count == 1);
    REQUIRE(r.open_lots[0].quantity == -4);
    REQUIRE(r.open_lots[0].price == 200);
    REQUIRE(r.realized_pnl + r.unrealized_pnl == r.gross_mtm_pnl);
}

TEST_CASE("offline pnl: PARTIAL-fill golden row (partial close leaves residue)",
          "[offline][pnl][accounting][golden][partial]") {
    // Buy 10 @ 100, Sell 4 @ 120, terminal mark 115. Position long 6 remaining.
    //   accumulated_cash = -10*100 + 4*120 = -1000 + 480 = -520
    //   position_mark_value = 6*115 = 690; gross_mtm = -520 + 690 = 170
    //   FIFO realized = (120-100)*4 = 80
    //   residue lot +6 @ 100 -> unrealized = (115-100)*6 = 90; reconciles 170.
    const auto fills = make_fills({{10, 100, 0, 0, 0}, {-4, 120, 0, 0, 0}});
    const PnlAttribution r = PnlAttributionReplay(fills, 115).compute();

    REQUIRE(r.final_position == 6);
    REQUIRE(r.accumulated_cash == -520);
    REQUIRE(r.gross_mtm_pnl == 170);
    REQUIRE(r.realized_pnl == 80);
    REQUIRE(r.unrealized_pnl == 90);
    REQUIRE(r.open_lot_count == 1);
    REQUIRE(r.open_lots[0].quantity == 6);
    REQUIRE(r.open_lots[0].price == 100);
    REQUIRE(r.realized_pnl + r.unrealized_pnl == r.gross_mtm_pnl);
}

TEST_CASE("offline pnl: TERMINAL-position golden row with fees, rebates, carry",
          "[offline][pnl][accounting][golden][terminal][fees]") {
    // Buy 2 @ 100 (taker fee 3), Buy 2 @ 110 (maker rebate 1, carry 2),
    // Sell 1 @ 130 (taker fee 1, carry 4). Terminal mark 120. Position long 3.
    //   accumulated_cash = -2*100 - 2*110 + 1*130 = -200 - 220 + 130 = -290
    //   position_mark_value = 3*120 = 360; gross_mtm = -290 + 360 = 70
    //   total_fees = 3 + 1 = 4; total_rebates = 1; total_carry = 2 + 4 = 6
    //   net_pnl = 70 - 4 + 1 - 6 = 61
    //   FIFO realized: Sell 1 closes oldest lot (+2 @100): (130-100)*1 = 30
    //   residue lots: +1 @100, +2 @110
    //     unrealized = (120-100)*1 + (120-110)*2 = 20 + 20 = 40; 30+40 = 70.
    const auto fills = make_fills(
        {{2, 100, 3, 0, 0}, {2, 110, 0, 1, 2}, {-1, 130, 1, 0, 4}});
    const PnlAttribution r = PnlAttributionReplay(fills, 120).compute();

    REQUIRE(r.final_position == 3);
    REQUIRE(r.accumulated_cash == -290);
    REQUIRE(r.gross_mtm_pnl == 70);
    REQUIRE(r.total_fees == 4);
    REQUIRE(r.total_rebates == 1);
    REQUIRE(r.total_carry == 6);
    REQUIRE(r.net_fee_pnl == -3);
    REQUIRE(r.net_pnl == 61);
    REQUIRE(r.realized_pnl == 30);
    REQUIRE(r.unrealized_pnl == 40);
    REQUIRE(r.open_lot_count == 2);
    REQUIRE(r.realized_pnl + r.unrealized_pnl == r.gross_mtm_pnl);
}

TEST_CASE("offline pnl: FIFO realized/unrealized split is order-sensitive while "
          "mark-to-market net PnL reconciles and is order-invariant",
          "[offline][pnl][accounting][order]") {
    // Same multiset {Buy@100, Buy@110, Sell@120}, terminal mark 130, in two
    // orders. gross_mtm is order-invariant; FIFO split differs.
    //   Order A [Buy@100, Buy@110, Sell@120]: Sell closes +1@100 -> realized 20;
    //     residue +1@110 -> unrealized (130-110)=20; total 40.
    //   Order B [Buy@110, Buy@100, Sell@120]: Sell closes +1@110 -> realized 10;
    //     residue +1@100 -> unrealized (130-100)=30; total 40.
    const auto a = make_fills({{1, 100, 0, 0, 0}, {1, 110, 0, 0, 0}, {-1, 120, 0, 0, 0}});
    const auto b = make_fills({{1, 110, 0, 0, 0}, {1, 100, 0, 0, 0}, {-1, 120, 0, 0, 0}});

    const PnlAttribution ra = PnlAttributionReplay(a, 130).compute();
    const PnlAttribution rb = PnlAttributionReplay(b, 130).compute();

    // ORDER-INVARIANT accounting identity.
    REQUIRE(ra.final_position == rb.final_position);
    REQUIRE(ra.accumulated_cash == rb.accumulated_cash);
    REQUIRE(ra.gross_mtm_pnl == 40);
    REQUIRE(rb.gross_mtm_pnl == 40);
    REQUIRE(ra.net_pnl == rb.net_pnl);

    // ORDER-SENSITIVE realized/unrealized split.
    REQUIRE(ra.realized_pnl == 20);
    REQUIRE(ra.unrealized_pnl == 20);
    REQUIRE(rb.realized_pnl == 10);
    REQUIRE(rb.unrealized_pnl == 30);

    // Reconciliation holds in BOTH orders despite the different split.
    REQUIRE(ra.realized_pnl + ra.unrealized_pnl == ra.gross_mtm_pnl);
    REQUIRE(rb.realized_pnl + rb.unrealized_pnl == rb.gross_mtm_pnl);

    // Therefore the replay hash differs even though the net PnL matches.
    REQUIRE(ra.replay_hash != rb.replay_hash);
}

TEST_CASE("offline pnl: peak absolute inventory is an order-sensitive path metric",
          "[offline][pnl][accounting][inventory][order]") {
    // Same multiset of signed quantities {+3, -2, +1} contributed in two orders.
    // Final position is order-invariant = 3 - 2 + 1 = 2 in both.
    //   Order A [+3, -2, +1]: running 3 -> 1 -> 2, peak |.| = 3.
    //   Order B [-2, +3, +1]: running -2 -> 1 -> 2, peak |.| = 2.
    const auto a = make_fills({{3, 100, 0, 0, 0}, {-2, 100, 0, 0, 0}, {1, 100, 0, 0, 0}});
    const auto b = make_fills({{-2, 100, 0, 0, 0}, {3, 100, 0, 0, 0}, {1, 100, 0, 0, 0}});

    const PnlAttribution ra = PnlAttributionReplay(a, 100).compute();
    const PnlAttribution rb = PnlAttributionReplay(b, 100).compute();

    REQUIRE(ra.final_position == 2);
    REQUIRE(rb.final_position == 2);
    REQUIRE(ra.peak_abs_position == 3);
    REQUIRE(rb.peak_abs_position == 2);
    REQUIRE(ra.replay_hash != rb.replay_hash);
}

TEST_CASE("offline pnl: closed-form analytic accounting relations",
          "[offline][pnl][accounting][analytic]") {
    // Round-trip realized PnL identity.
    REQUIRE(analytic::round_trip_pnl(100, 130, 5) == 150);
    REQUIRE(analytic::round_trip_pnl(200, 180, -4) == 80);  // short: (180-200)*-4

    // Fee/rebate sign convention.
    REQUIRE(analytic::net_fee(/*rebate=*/1, /*fee=*/4) == -3);
    REQUIRE(analytic::net_fee(/*rebate=*/5, /*fee=*/0) == 5);

    // Explicit integer carry/financing charge, no hidden zero default.
    REQUIRE(analytic::carry_cost(/*rate=*/2, /*abs_position=*/3, /*periods=*/4) == 24);
    REQUIRE(analytic::carry_cost(2, -3, 4) == 24);  // magnitude only

    // Net-after-costs fold ties the separate fields together.
    REQUIRE(analytic::net_after_costs(/*gross=*/70, /*fees=*/4, /*rebates=*/1,
                                      /*carry=*/6) == 61);
}

TEST_CASE("offline pnl: replay-hash determinism",
          "[offline][pnl][accounting][determinism]") {
    const auto fills = make_fills(
        {{3, 100, 2, 0, 1}, {-1, 130, 1, 0, 0}, {2, 110, 0, 3, 2}});

    // Same ordered input -> identical hash, every time and across instances.
    const PnlAttributionReplay x(fills, 125);
    const PnlAttributionReplay y(fills, 125);
    const std::uint64_t h1 = x.compute().replay_hash;
    const std::uint64_t h2 = x.compute().replay_hash;  // idempotent
    const std::uint64_t h3 = y.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // A different terminal mark changes the hash.
    REQUIRE(PnlAttributionReplay(fills, 126).hash() != h1);

    // An extra fill changes the hash, and the count fold prevents a prefix
    // collision.
    const auto extended = make_fills(
        {{3, 100, 2, 0, 1}, {-1, 130, 1, 0, 0}, {2, 110, 0, 3, 2}, {1, 105, 0, 0, 0}});
    REQUIRE(PnlAttributionReplay(extended, 125).hash() != h1);

    // Changing a single fill field (a fee) changes the hash.
    const auto perturbed = make_fills(
        {{3, 100, 9, 0, 1}, {-1, 130, 1, 0, 0}, {2, 110, 0, 3, 2}});
    REQUIRE(PnlAttributionReplay(perturbed, 125).hash() != h1);
}

TEST_CASE("offline pnl: degenerate and empty inputs are well-defined",
          "[offline][pnl][accounting][determinism]") {
    // Single zero-quantity fill: no position, no realized/unrealized, but the
    // explicit fee/carry still ledger and a stable nonzero hash is produced.
    const auto zero = make_fills({{0, 100, 7, 0, 5}});
    const PnlAttribution rz = PnlAttributionReplay(zero, 200).compute();
    REQUIRE(rz.final_position == 0);
    REQUIRE(rz.gross_mtm_pnl == 0);
    REQUIRE(rz.realized_pnl == 0);
    REQUIRE(rz.unrealized_pnl == 0);
    REQUIRE(rz.open_lot_count == 0);
    REQUIRE(rz.total_fees == 7);
    REQUIRE(rz.total_carry == 5);
    REQUIRE(rz.net_pnl == -12);  // 0 - 7 + 0 - 5
    REQUIRE(rz.replay_hash != 0);

    // Empty -> well-defined zeros, no lots, but a stable deterministic hash.
    const std::vector<Fill> none;
    const PnlAttributionReplay empty(none, 100);
    REQUIRE(empty.empty());
    const PnlAttribution e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.gross_mtm_pnl == 0);
    REQUIRE(e.net_pnl == 0);
    REQUIRE(e.open_lots.empty());
    REQUIRE(e.replay_hash == empty.hash());  // deterministic
}
