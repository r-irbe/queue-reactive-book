#include <catch2/catch_test_macros.hpp>

#include "offline/liquidity_admissibility_replay.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline falsifier for `moc-liquidity-epistemics`'s code/replay
// channel (Wave-I gate G8). It proves, with ZERO FlatLadderBook / sim::
// coupling, that:
//   (a) reconstruction is deterministic / replay-hash invariant;
//   (b) the censoring bias has the correct DIRECTION -- an observer using only
//       displayed depth under-estimates available liquidity by exactly the
//       hidden amount (observed <= true), and a fill is a right-censored
//       observation when latent size remains;
//   (c) the diagnostic is strictly read-only / non-feedback (const-correctness
//       enforced at compile time via static_assert).
//
// Depends only on the self-contained header
// `offline/liquidity_admissibility_replay.hpp`.
// =============================================================================

namespace {

using offline::FillOutcome;
using offline::LiquidityAdmissibilityReplay;
using offline::LiquidityReconstruction;
using offline::LiquiditySide;
using offline::OrderEvent;
using offline::PriceTick;
using offline::Qty;

// A small hand-built trace with explicit iceberg (hidden > display) orders so
// the censoring bias is known exactly by construction.
std::vector<OrderEvent> hand_trace() {
    // Bids at 99,100; asks at 101,102. Several iceberg orders.
    return {
        // side, price, display, total
        {LiquiditySide::Bid, 100, 5, 5},    // fully displayed
        {LiquiditySide::Bid, 100, 2, 12},   // iceberg: hidden 10
        {LiquiditySide::Bid, 99, 7, 7},     // fully displayed
        {LiquiditySide::Ask, 101, 3, 3},    // fully displayed
        {LiquiditySide::Ask, 101, 1, 9},    // iceberg: hidden 8
        {LiquiditySide::Ask, 102, 4, 20},   // iceberg: hidden 16
    };
}

// Generate a seeded random trace where each order's hidden portion is >= 0 by
// construction, so the censoring-bias direction must always hold.
std::vector<OrderEvent> random_trace(std::uint64_t seed, std::size_t n) {
    std::mt19937_64 eng(seed);
    std::uniform_int_distribution<int> side_dist(0, 1);
    std::uniform_int_distribution<PriceTick> price_dist(90, 110);
    std::uniform_int_distribution<Qty> disp_dist(1, 50);
    std::uniform_int_distribution<Qty> hidden_dist(0, 200);

    std::vector<OrderEvent> trace;
    trace.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto side =
            side_dist(eng) == 0 ? LiquiditySide::Bid : LiquiditySide::Ask;
        const PriceTick price = price_dist(eng);
        const Qty display = disp_dist(eng);
        const Qty total = display + hidden_dist(eng); // total >= display
        trace.push_back({side, price, display, total});
    }
    return trace;
}

} // namespace

// (c) Compile-time read-only / non-feedback guarantees mirrored in the test so a
// regression that makes the diagnostic mutating fails the test TU's compile.
static_assert(
    std::is_invocable_r_v<LiquidityReconstruction,
                          decltype(&LiquidityAdmissibilityReplay::reconstruct),
                          const LiquidityAdmissibilityReplay&>,
    "reconstruct() must be callable on a const engine (read-only)");
static_assert(
    std::is_invocable_r_v<FillOutcome,
                          decltype(&LiquidityAdmissibilityReplay::simulate_fill),
                          const LiquidityAdmissibilityReplay&, LiquiditySide, Qty>,
    "simulate_fill() must be callable on a const engine (read-only)");

TEST_CASE("offline liquidity replay: hand-built iceberg reconstruction is exact",
          "[offline][liquidity][censoring]") {
    const LiquidityAdmissibilityReplay engine(hand_trace());
    const LiquidityReconstruction r = engine.reconstruct();

    // Bid side: price 99 -> obs 7 / true 7; price 100 -> obs 7 / true 17.
    REQUIRE(r.bid_levels.size() == 2);
    REQUIRE(r.bid_levels[0].price == 99);
    REQUIRE(r.bid_levels[0].observed_qty == 7);
    REQUIRE(r.bid_levels[0].true_qty == 7);
    REQUIRE(r.bid_levels[0].hidden_qty() == 0);
    REQUIRE(r.bid_levels[1].price == 100);
    REQUIRE(r.bid_levels[1].observed_qty == 7);  // 5 + 2
    REQUIRE(r.bid_levels[1].true_qty == 17);     // 5 + 12
    REQUIRE(r.bid_levels[1].hidden_qty() == 10);

    // Ask side: price 101 -> obs 4 / true 12; price 102 -> obs 4 / true 20.
    REQUIRE(r.ask_levels.size() == 2);
    REQUIRE(r.ask_levels[0].price == 101);
    REQUIRE(r.ask_levels[0].observed_qty == 4);  // 3 + 1
    REQUIRE(r.ask_levels[0].true_qty == 12);     // 3 + 9
    REQUIRE(r.ask_levels[1].price == 102);
    REQUIRE(r.ask_levels[1].observed_qty == 4);
    REQUIRE(r.ask_levels[1].true_qty == 20);

    // Aggregate: observed 7+7+4+4 = 22, true 7+17+12+20 = 56, bias = 34.
    REQUIRE(r.observed_total == 22);
    REQUIRE(r.true_total == 56);
    REQUIRE(r.censoring_bias() == 34);
}

TEST_CASE("offline liquidity replay: determinism / replay-hash invariance",
          "[offline][liquidity][determinism]") {
    // Same trace -> identical reconstruction and hash, every time.
    const LiquidityAdmissibilityReplay a(hand_trace());
    const LiquidityAdmissibilityReplay b(hand_trace());
    const std::uint64_t h1 = a.reconstruct().replay_hash;
    const std::uint64_t h2 = a.reconstruct().replay_hash; // idempotent
    const std::uint64_t h3 = b.reconstruct().replay_hash;
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // Insertion order of same-price orders must not change the canonical hash
    // (levels are canonicalized by price).
    std::vector<OrderEvent> shuffled = hand_trace();
    std::mt19937_64 eng(0xFEEDFACEULL);
    std::shuffle(shuffled.begin(), shuffled.end(), eng);
    const LiquidityAdmissibilityReplay c(shuffled);
    REQUIRE(c.reconstruct().replay_hash == h1);

    // A genuinely different trace (extra hidden size) must change the hash.
    std::vector<OrderEvent> more = hand_trace();
    more.push_back({LiquiditySide::Ask, 102, 0, 5}); // pure hidden add
    const LiquidityAdmissibilityReplay d(more);
    REQUIRE(d.reconstruct().replay_hash != h1);
}

TEST_CASE("offline liquidity replay: censoring-bias direction is correct",
          "[offline][liquidity][censoring]") {
    // Across many seeded traces, observed <= true ALWAYS, per level and in
    // aggregate; bias equals the summed hidden quantity exactly.
    const std::uint64_t seeds[] = {1, 7, 42, 1337, 0xABCDEF, 0xC0FFEE};
    const std::size_t sizes[] = {1, 5, 25, 200, 1000};
    std::size_t checked = 0;

    for (std::uint64_t seed : seeds) {
        for (std::size_t n : sizes) {
            const LiquidityAdmissibilityReplay engine(random_trace(seed, n));
            const LiquidityReconstruction r = engine.reconstruct();

            Qty summed_hidden = 0;
            for (const auto& lvl : r.bid_levels) {
                REQUIRE(lvl.observed_qty <= lvl.true_qty); // direction
                REQUIRE(lvl.hidden_qty() >= 0);
                summed_hidden += lvl.hidden_qty();
            }
            for (const auto& lvl : r.ask_levels) {
                REQUIRE(lvl.observed_qty <= lvl.true_qty);
                REQUIRE(lvl.hidden_qty() >= 0);
                summed_hidden += lvl.hidden_qty();
            }

            // Displayed-only estimator is biased LOW by exactly hidden size.
            REQUIRE(r.observed_total <= r.true_total);
            REQUIRE(r.censoring_bias() >= 0);
            REQUIRE(r.censoring_bias() == summed_hidden);
            REQUIRE(r.observed_total + r.censoring_bias() == r.true_total);
            ++checked;
        }
    }
    REQUIRE(checked == 30);
}

TEST_CASE("offline liquidity replay: a fill is a right-censored observation",
          "[offline][liquidity][censoring][fill]") {
    const LiquidityAdmissibilityReplay engine(hand_trace());

    // Ask-side true depth = 12 + 20 = 32; observed (displayed) depth = 8.
    // A taker Bid of 10 lifts the ask side.
    const FillOutcome fo = engine.simulate_fill(LiquiditySide::Bid, /*qty=*/10);

    // Observer sizing from the displayed book under-predicts the fill: it sees
    // only 8 of displayed depth, so it predicts at most 8 fillable.
    REQUIRE(fo.observer_predicted_fill == 8);
    // Reality fills the full 10 against true (display + hidden) depth.
    REQUIRE(fo.realized_fill == 10);
    REQUIRE(fo.observer_predicted_fill < fo.realized_fill); // censoring direction
    // Latent (hidden) liquidity remains after the fill: 32 - 10 = 22 -> the
    // print is a right-censored observation of total executable volume.
    REQUIRE(fo.latent_remaining == 22);
    REQUIRE(fo.right_censored);

    // A taker that requests >= the full true depth exhausts the book: no latent
    // remainder, so the observation is NOT right-censored.
    const FillOutcome full = engine.simulate_fill(LiquiditySide::Bid, /*qty=*/100);
    REQUIRE(full.realized_fill == 32);
    REQUIRE(full.latent_remaining == 0);
    REQUIRE_FALSE(full.right_censored);
}

TEST_CASE("offline liquidity replay: empty trace is well-defined and unbiased",
          "[offline][liquidity][determinism]") {
    const LiquidityAdmissibilityReplay engine{};
    REQUIRE(engine.empty());
    const LiquidityReconstruction r = engine.reconstruct();
    REQUIRE(r.bid_levels.empty());
    REQUIRE(r.ask_levels.empty());
    REQUIRE(r.observed_total == 0);
    REQUIRE(r.true_total == 0);
    REQUIRE(r.censoring_bias() == 0); // no hidden liquidity, no bias

    const FillOutcome fo = engine.simulate_fill(LiquiditySide::Bid, 50);
    REQUIRE(fo.realized_fill == 0);
    REQUIRE(fo.observer_predicted_fill == 0);
    REQUIRE_FALSE(fo.right_censored);
}
