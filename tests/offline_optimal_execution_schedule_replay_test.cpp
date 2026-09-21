#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/optimal_execution_schedule_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// (the channel asked for an offline optimal-execution-schedule computation and a
// post-hoc TCA-metric fixture, all carried by a replayable hash -- supplied here).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash over a fixed synthetic fill stream is deterministic AND
//       order-sensitive: the same ordered stream -> the same hash; a reordering
//       of the SAME multiset of fills -> a different hash (even though every
//       TCA aggregate is order-invariant);
//   (2) the Almgren-Chriss optimal-execution trajectory is correct on hand-built
//       cases: the closed form satisfies its boundary conditions and second-order
//       recurrence, collapses to the exact LINEAR/TWAP schedule as risk-aversion
//       -> 0, and front-loads monotonically as risk-aversion increases;
//   (3) the post-hoc TCA metric (implementation shortfall / VWAP slippage) on a
//       KNOWN fill set matches hand-checked expected values;
//   (4) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// HONEST SCOPE: this validates the OFFLINE schedule-computation + post-hoc TCA
// measurement discipline (planning/measurement statistics), NOT that the
// Almgren-Chriss linear-impact model (or any impact model) is the true market
// mechanism, and the schedule NEVER feeds back into matching/replay.
//
// Depends only on the self-contained header
// `offline/optimal_execution_schedule_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::compute_execution_schedule;
using offline::compute_tca;
using offline::ExecutionFill;
using offline::ExecutionSchedule;
using offline::ExecutionScheduleParams;
using offline::OptimalExecutionScheduleReplay;
using offline::TcaInput;
using offline::TcaResult;
using offline::Tick;

// Build a fill stream from (quantity, price) pairs for a single side; seq and
// child_id are the position, is_maker alternates so the stream is non-trivial.
std::vector<ExecutionFill> make_fills(const std::vector<std::pair<std::int64_t, Tick>>& qp,
                                      int side) {
    std::vector<ExecutionFill> out;
    out.reserve(qp.size());
    std::uint64_t seq = 0;
    for (const auto& [q, p] : qp) {
        out.push_back(ExecutionFill{seq, seq + 1, side, q, p, (seq % 2) == 1});
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

// (4) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<TcaResult>,
              "TcaResult must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ExecutionSchedule>,
              "ExecutionSchedule must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ExecutionFill>,
              "ExecutionFill must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<ExecutionSchedule,
                          decltype(&OptimalExecutionScheduleReplay::schedule),
                          const OptimalExecutionScheduleReplay&>,
    "schedule() must be callable on a const engine (read-only)");
static_assert(
    std::is_invocable_r_v<TcaResult,
                          decltype(&OptimalExecutionScheduleReplay::tca),
                          const OptimalExecutionScheduleReplay&>,
    "tca() must be callable on a const engine (read-only)");

TEST_CASE("offline execution: risk-neutral schedule collapses to the exact "
          "LINEAR/TWAP trajectory (kappa -> 0)",
          "[offline][execution][twap]") {
    // X=100 over N=4 intervals, kappa=0 (risk-neutral limit). The optimal
    // trajectory is the linear schedule x_j = X*(N-j)/N = {100,75,50,25,0} with
    // equal child trades X/N = 25 each.
    const ExecutionScheduleParams params{100, 4, 0.0};
    const ExecutionSchedule s = compute_execution_schedule(params);

    REQUIRE(s.intervals == 4);
    REQUIRE(s.is_twap);
    REQUIRE_FALSE(s.front_loaded);  // TWAP is flat, not front-loaded
    REQUIRE(s.holdings.size() == 5);
    REQUIRE(s.trades.size() == 4);

    REQUIRE(s.holdings[0] == Approx(100.0));
    REQUIRE(s.holdings[1] == Approx(75.0));
    REQUIRE(s.holdings[2] == Approx(50.0));
    REQUIRE(s.holdings[3] == Approx(25.0));
    REQUIRE(s.holdings[4] == Approx(0.0));
    for (double n : s.trades) REQUIRE(n == Approx(25.0));

    // Child trades sum back to X (mass conservation).
    double sum = 0.0;
    for (double n : s.trades) sum += n;
    REQUIRE(sum == Approx(100.0));
}

TEST_CASE("offline execution: risk-averse Almgren-Chriss trajectory satisfies its "
          "boundary conditions and second-order recurrence",
          "[offline][execution][almgren-chriss]") {
    // X=100, N=4, per-interval decay kappa=0.5 (risk-averse). The closed form
    //   x_j = X*sinh(kappa*(N-j))/sinh(kappa*N)
    // must satisfy x_0=X, x_N=0 exactly, the recurrence
    //   x_{j-1} + x_{j+1} = 2*cosh(kappa)*x_j,
    // and produce strictly decreasing (front-loaded) child trades summing to X.
    const double kappa = 0.5;
    const ExecutionScheduleParams params{100, 4, kappa};
    const ExecutionSchedule s = compute_execution_schedule(params);

    REQUIRE_FALSE(s.is_twap);
    REQUIRE(s.front_loaded);

    // Exact boundary conditions.
    REQUIRE(s.holdings.front() == Approx(100.0));
    REQUIRE(s.holdings.back() == Approx(0.0));

    // First holding matches the hand-computed sinh ratio:
    //   x_1 = 100*sinh(1.5)/sinh(2.0) = 100*2.12927946/3.62686041 = 58.708613...
    REQUIRE(s.holdings[1] == Approx(58.708613).epsilon(1e-6));

    // Second-order recurrence holds at every interior node.
    const double two_cosh = 2.0 * offline::analytic::ac_recurrence_factor(kappa);
    for (std::size_t j = 1; j + 1 < s.holdings.size(); ++j) {
        REQUIRE(s.holdings[j - 1] + s.holdings[j + 1] ==
                Approx(two_cosh * s.holdings[j]).epsilon(1e-9));
    }

    // Strictly decreasing child trades, summing to X.
    double sum = 0.0;
    for (std::size_t j = 0; j < s.trades.size(); ++j) {
        sum += s.trades[j];
        REQUIRE(s.trades[j] > 0.0);
        if (j > 0) REQUIRE(s.trades[j] < s.trades[j - 1]);
    }
    REQUIRE(sum == Approx(100.0));
}

TEST_CASE("offline execution: front-loading increases monotonically with "
          "risk-aversion, and small kappa converges to TWAP",
          "[offline][execution][almgren-chriss][front-loading]") {
    const std::int64_t X = 100;
    const std::size_t N = 4;

    const ExecutionSchedule lo = compute_execution_schedule({X, N, 0.5});
    const ExecutionSchedule hi = compute_execution_schedule({X, N, 1.0});

    // More risk-averse (larger kappa) => more front-loaded => every interior
    // holding is strictly smaller (sells faster early), and the first child
    // trade is strictly larger.
    for (std::size_t j = 1; j + 1 < lo.holdings.size(); ++j) {
        REQUIRE(hi.holdings[j] < lo.holdings[j]);
    }
    REQUIRE(hi.trades.front() > lo.trades.front());
    REQUIRE(lo.front_loaded);
    REQUIRE(hi.front_loaded);

    // As kappa -> 0 the closed form converges to the exact TWAP trajectory.
    const ExecutionSchedule tiny = compute_execution_schedule({X, N, 1e-5});
    const ExecutionSchedule twap = compute_execution_schedule({X, N, 0.0});
    for (std::size_t j = 0; j < twap.holdings.size(); ++j) {
        REQUIRE(tiny.holdings[j] == Approx(twap.holdings[j]).margin(1e-2));
    }
}

TEST_CASE("offline execution: implementation shortfall / VWAP slippage match "
          "hand-checked values (buy and sell)",
          "[offline][execution][tca]") {
    // BUY of 20 shares, arrival 100, benchmark VWAP 101. Fills 10@101 and 10@103.
    //   notional = 10*101 + 10*103 = 2040; avg = 102.
    //   IS/share = +(102 - 100) = +2 (paid 2 ticks above arrival -> cost).
    //   IS total = 2 * 20 = 40.
    //   VWAP slippage = +(102 - 101) = +1 (1 tick worse than benchmark).
    TcaInput buy;
    buy.side = 1;
    buy.arrival_price = 100;
    buy.benchmark_vwap = 101;
    buy.fills = make_fills({{10, 101}, {10, 103}}, /*side=*/1);
    const TcaResult rb = compute_tca(buy);

    REQUIRE(rb.executed_quantity == 20);
    REQUIRE(rb.notional == 2040);
    REQUIRE(rb.avg_exec_price == Approx(102.0));
    REQUIRE(rb.implementation_shortfall_per_share == Approx(2.0));
    REQUIRE(rb.implementation_shortfall_total == Approx(40.0));
    REQUIRE(rb.vwap_slippage_per_share == Approx(1.0));

    // SELL of 20 shares, arrival 100, benchmark VWAP 99. Fills 10@99 and 10@97.
    //   avg = 98. IS/share = -(98 - 100) = +2 (sold 2 below arrival -> cost).
    //   VWAP slippage = -(98 - 99) = +1 (1 tick worse than benchmark).
    TcaInput sell;
    sell.side = -1;
    sell.arrival_price = 100;
    sell.benchmark_vwap = 99;
    sell.fills = make_fills({{10, 99}, {10, 97}}, /*side=*/-1);
    const TcaResult rs = compute_tca(sell);

    REQUIRE(rs.executed_quantity == 20);
    REQUIRE(rs.avg_exec_price == Approx(98.0));
    REQUIRE(rs.implementation_shortfall_per_share == Approx(2.0));
    REQUIRE(rs.implementation_shortfall_total == Approx(40.0));
    REQUIRE(rs.vwap_slippage_per_share == Approx(1.0));

    // A favorable fill: buy filled BELOW arrival -> negative shortfall (a saving).
    TcaInput good;
    good.side = 1;
    good.arrival_price = 100;
    good.benchmark_vwap = 100;
    good.fills = make_fills({{10, 98}, {10, 98}}, /*side=*/1);
    const TcaResult rg = compute_tca(good);
    REQUIRE(rg.avg_exec_price == Approx(98.0));
    REQUIRE(rg.implementation_shortfall_per_share == Approx(-2.0));
}

TEST_CASE("offline execution: fill-stream replay hash is deterministic",
          "[offline][execution][determinism]") {
    const ExecutionScheduleParams params{100, 4, 0.5};
    TcaInput in;
    in.side = 1;
    in.arrival_price = 100;
    in.benchmark_vwap = 101;
    in.fills = make_fills({{10, 101}, {5, 102}, {10, 103}}, 1);

    const OptimalExecutionScheduleReplay a(params, in);
    const OptimalExecutionScheduleReplay b(params, in);
    const std::uint64_t h1 = a.tca().fill_stream_hash;
    const std::uint64_t h2 = a.tca().fill_stream_hash;  // idempotent
    const std::uint64_t h3 = b.fill_replay_hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // One extra fill changes the hash; the count fold prevents prefix collisions.
    TcaInput extended = in;
    extended.fills = make_fills({{10, 101}, {5, 102}, {10, 103}, {1, 104}}, 1);
    REQUIRE(compute_tca(extended).fill_stream_hash != h1);

    // The schedule hash is deterministic too, and a parameter change perturbs it.
    REQUIRE(a.schedule_replay_hash() == b.schedule_replay_hash());
    REQUIRE(compute_execution_schedule({100, 4, 0.6}).schedule_hash !=
            a.schedule_replay_hash());
}

TEST_CASE("offline execution: fill-stream hash is order-sensitive while the TCA "
          "aggregates are order-invariant",
          "[offline][execution][determinism][order]") {
    // The SAME multiset of three fills in two different orders.
    TcaInput forward;
    forward.side = 1;
    forward.arrival_price = 100;
    forward.benchmark_vwap = 101;
    forward.fills = make_fills({{10, 101}, {5, 102}, {10, 103}}, 1);

    TcaInput shuffled = forward;
    shuffled.fills = make_fills({{10, 103}, {10, 101}, {5, 102}}, 1);

    const TcaResult fa = compute_tca(forward);
    const TcaResult fb = compute_tca(shuffled);

    // TCA aggregates are order-INVARIANT.
    REQUIRE(fa.executed_quantity == fb.executed_quantity);
    REQUIRE(fa.notional == fb.notional);
    REQUIRE(fa.avg_exec_price == Approx(fb.avg_exec_price));
    REQUIRE(fa.implementation_shortfall_total ==
            Approx(fb.implementation_shortfall_total));
    REQUIRE(fa.vwap_slippage_per_share == Approx(fb.vwap_slippage_per_share));

    // Yet the order-SENSITIVE replay hash differs: a reordering is detectable.
    REQUIRE(fa.fill_stream_hash != fb.fill_stream_hash);
}

TEST_CASE("offline execution: degenerate and empty inputs are well-defined",
          "[offline][execution][determinism]") {
    // Empty schedule: no holdings/trades, but a stable deterministic hash.
    const ExecutionSchedule e0 = compute_execution_schedule({100, 0, 0.5});
    REQUIRE(e0.intervals == 0);
    REQUIRE(e0.holdings.empty());
    REQUIRE(e0.trades.empty());
    REQUIRE(e0.schedule_hash != 0);

    // Single-interval schedule liquidates everything in one trade.
    const ExecutionSchedule one = compute_execution_schedule({100, 1, 0.5});
    REQUIRE(one.holdings.size() == 2);
    REQUIRE(one.holdings[0] == Approx(100.0));
    REQUIRE(one.holdings[1] == Approx(0.0));
    REQUIRE(one.trades.size() == 1);
    REQUIRE(one.trades[0] == Approx(100.0));
    REQUIRE_FALSE(one.front_loaded);  // needs >= 2 intervals to claim front-loading

    // Empty fill stream: zero aggregates, but a stable deterministic hash.
    TcaInput empty;
    empty.side = 1;
    empty.arrival_price = 100;
    empty.benchmark_vwap = 100;
    const TcaResult re = compute_tca(empty);
    REQUIRE(re.executed_quantity == 0);
    REQUIRE(re.notional == 0);
    REQUIRE(re.avg_exec_price == Approx(0.0));
    REQUIRE(re.implementation_shortfall_total == Approx(0.0));
    REQUIRE(re.fill_stream_hash != 0);

    const ExecutionScheduleParams params{100, 4, 0.5};
    const OptimalExecutionScheduleReplay wrap(params, empty);
    REQUIRE(wrap.empty());
    REQUIRE(wrap.fill_count() == 0);
    REQUIRE(wrap.fill_replay_hash() == re.fill_stream_hash);  // deterministic
}
