#pragma once

// =============================================================================
// Offline optimal-execution-SCHEDULE + post-hoc TCA REPLAY fixture.
//
// by providing the missing deterministic offline fixture: an Almgren-Chriss
// optimal-execution-trajectory computation (closed form + its TWAP limit and
// front-loading monotonicity) and a post-hoc transaction-cost-analysis metric
// (implementation shortfall / VWAP slippage), all carried by an order-sensitive
// replay hash over a fixed synthetic fill stream.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and event-driven*; everything computed
//   here (the parent-order liquidation trajectory and the trade schedule, and
//   the post-trade implementation-shortfall / VWAP-slippage statistics over an
//   already-recorded fill stream) is a PURE OFFLINE planning/measurement
//   statistic. Data flows ONE WAY, OUT of a recorded replay and into derived
//   schedules and TCA numbers; a computed schedule or TCA value may NEVER be fed
//   back as a matching input, a hot-path branch, a price-process mechanic, a
//   fill rule, or production state. The schedule NEVER feeds back into
//   matching/replay. This header is the deterministic proof of that one-way,
//   kernel-free contract.
//
// WHAT THIS FIXTURE PROVES (and, just as importantly, what it does NOT):
//   It validates the OFFLINE schedule-COMPUTATION and the post-hoc TCA
//   measurement DISCIPLINE -- that a parent-order trajectory and shortfall/VWAP
//   statistics can be computed deterministically, off the hot path, with a
//   replayable order-sensitive hash. It is IMPLEMENTATION EVIDENCE ONLY
//   (boundary-constructibility): it is NOT a proof that the Almgren-Chriss
//   linear temporary/permanent-impact model (or ANY impact model) is the true
//   market mechanism, and it is NOT engine adoption. The closed form is a
//   PLANNING statistic; implementation shortfall and VWAP slippage are post-hoc
//   MEASUREMENT statistics. Neither asserts a causal impact law and neither ever
//   touches matching.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   the computations are PURE functions of their (immutable) inputs: identical
//   schedule parameters yield an identical schedule + hash, and an identical
//   ordered fill stream yields an identical TCA result + hash; reordering the
//   SAME multiset of fills leaves the order-INVARIANT TCA aggregates (executed
//   quantity, notional, average price, shortfall, slippage) unchanged yet
//   changes the order-SENSITIVE fill-stream hash (replay integrity). Every
//   computation takes its input by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Execution / TCA background (kept self-contained; no equation imported from any
// rendered source image -- these are the textbook relations re-stated here for
// the fixture):
//   * Almgren-Chriss: liquidating X shares over N equal intervals (per-interval
//     decay kappa = kappa_phys * tau) under LINEAR temporary/permanent impact and
//     a mean-variance objective gives the optimal remaining-holdings trajectory
//       x_j = X * sinh(kappa*(N - j)) / sinh(kappa*N),    j = 0..N,
//     with boundary conditions x_0 = X, x_N = 0. It satisfies the second-order
//     recurrence  x_{j-1} + x_{j+1} = 2*cosh(kappa)*x_j  (a property of sinh). As
//     the risk-aversion -> 0 (kappa -> 0) it collapses to the LINEAR / TWAP
//     schedule x_j = X*(N - j)/N (equal child trades X/N); as risk-aversion
//     increases (kappa up) the schedule FRONT-LOADS (strictly decreasing child
//     trades, interior holdings pushed below the TWAP line).
//   * Implementation shortfall is the signed cost of an execution against an
//     arrival/decision reference price; VWAP slippage is the signed gap of the
//     achieved average price against a benchmark VWAP. Both are post-hoc
//     measurement statistics over a recorded fill stream, signed by trade
//     direction so that a positive value is always an adverse cost.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price / price-reference unit

// One replayable synthetic fill. Pure integer/enum data: no pointer, no engine
// handle, no vtable. `seq` is the deterministic event ordinal/time, `child_id`
// is the child-order id, `side` is the parent direction (+1 buy, -1 sell),
// `quantity` is shares filled (>= 0), `price` is the execution price in ticks,
// and `is_maker` is TCA metadata (maker/taker) folded into the replay hash.
struct ExecutionFill {
    std::uint64_t seq = 0;
    std::uint64_t child_id = 0;
    int side = 1;
    std::int64_t quantity = 0;
    Tick price = 0;
    bool is_maker = false;
};

// FIXED Almgren-Chriss schedule parameters. Chosen OFFLINE; `kappa` is the
// per-interval urgency/decay (kappa_phys * tau). `kappa <= 0` selects the
// risk-neutral TWAP/linear limit. Pure value type.
struct ExecutionScheduleParams {
    std::int64_t total_quantity = 0;  // X, parent shares to liquidate (>= 0)
    std::size_t intervals = 0;        // N, number of trading intervals
    double kappa = 0.0;               // per-interval AC decay; <= 0 => TWAP limit
};

// Post-hoc TCA input: an execution direction, an arrival/decision reference
// price, a benchmark VWAP, and the recorded fill stream. Pure value type.
struct TcaInput {
    int side = 1;            // +1 buy, -1 sell
    Tick arrival_price = 0;  // arrival/decision reference price (ticks)
    Tick benchmark_vwap = 0;  // market VWAP benchmark over the interval (ticks)
    std::vector<ExecutionFill> fills;
};

// Derived optimal-execution schedule plus an order-sensitive replay hash. Pure
// value type: copyable, holds no reference to any external (matching) state.
struct ExecutionSchedule {
    std::size_t intervals = 0;

    // Remaining-holdings trajectory x_0..x_N (length N+1); x_0 = X, x_N = 0.
    std::vector<double> holdings;
    // Child trades n_1..n_N (length N); n_j = x_{j-1} - x_j, all >= 0, sum = X.
    std::vector<double> trades;

    bool is_twap = false;       // true iff the risk-neutral linear/TWAP limit
    bool front_loaded = false;  // true iff trades strictly decreasing (urgency>0)

    ReplayHash schedule_hash = 0;
};

// Derived TCA statistics plus the order-sensitive fill-stream replay hash. Pure
// value type. Aggregates are order-INVARIANT; the hash is order-SENSITIVE.
struct TcaResult {
    std::int64_t executed_quantity = 0;  // sum q_i (exact)
    std::int64_t notional = 0;           // sum q_i * p_i (exact, ticks*shares)

    double avg_exec_price = 0.0;  // notional / executed_quantity

    // Implementation shortfall vs the arrival price, signed by side so a positive
    // value is an adverse cost (buy filled above / sell filled below arrival).
    double implementation_shortfall_total = 0.0;      // side*(avg-arrival)*qty
    double implementation_shortfall_per_share = 0.0;  // side*(avg-arrival)

    // VWAP slippage vs the benchmark VWAP, signed by side (positive = adverse).
    double vwap_slippage_per_share = 0.0;  // side*(avg - benchmark_vwap)

    ReplayHash fill_stream_hash = 0;
};

namespace detail {

inline constexpr ReplayHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr ReplayHash kFnvPrime = 1099511628211ULL;

inline constexpr ReplayHash fnv1a_byte(ReplayHash h, std::uint8_t b) noexcept {
    h ^= static_cast<ReplayHash>(b);
    h *= kFnvPrime;
    return h;
}

// Fold a fixed-width value in canonical little-endian order, so the hash does
// not depend on host endianness.
inline constexpr ReplayHash fnv1a_u64(ReplayHash h, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = fnv1a_byte(h, static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
    return h;
}

inline constexpr ReplayHash fnv1a_i64(ReplayHash h, std::int64_t v) noexcept {
    return fnv1a_u64(h, static_cast<std::uint64_t>(v));
}

// Quantize a real value to a fixed-point integer so it can be folded into the
// integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// Deterministic exp via range reduction (exp(x) = exp(x/2^k)^(2^k)) plus a
// fixed-length Taylor series on the reduced, tiny argument. Free of any <cmath>
// dependency and deterministic / accurate for the moderate arguments this
// fixture uses (per-interval kappa over a handful of intervals).
inline double det_exp(double x) noexcept {
    constexpr int kReductions = 12;       // 2^12 = 4096
    const double r = x / 4096.0;          // reduced, |r| tiny
    double term = 1.0;
    double sum = 1.0;
    for (int i = 1; i < 18; ++i) {        // fixed-length Taylor on the tiny arg
        term *= r / static_cast<double>(i);
        sum += term;
    }
    for (int i = 0; i < kReductions; ++i) {  // re-square back up
        sum *= sum;
    }
    return sum;
}

inline double det_sinh(double x) noexcept {
    return 0.5 * (det_exp(x) - det_exp(-x));
}

inline double det_cosh(double x) noexcept {
    return 0.5 * (det_exp(x) + det_exp(-x));
}

}  // namespace detail

// Analytic reference relations, used to validate the schedule on closed-form
// cases. Textbook relations, computed here so the fixture is self-checking and
// self-contained.
namespace analytic {

// TWAP / linear-limit remaining holding: x_j = X*(N - j)/N. Exact.
inline double twap_holding(double total, std::size_t intervals,
                           std::size_t j) noexcept {
    if (intervals == 0 || j >= intervals) return 0.0;
    return total * (static_cast<double>(intervals - j) /
                    static_cast<double>(intervals));
}

// Almgren-Chriss closed-form remaining holding under per-interval decay kappa:
//   x_j = X * sinh(kappa*(N - j)) / sinh(kappa*N),  with x_0 = X, x_N = 0.
// kappa <= 0 selects the exact TWAP/linear limit. Boundary nodes are returned
// exactly to keep x_0 == X and x_N == 0 free of floating-point residue.
inline double ac_holding(double total, std::size_t intervals, std::size_t j,
                         double kappa) noexcept {
    if (intervals == 0 || j >= intervals) return 0.0;  // x_N = 0
    if (j == 0) return total;                           // x_0 = X
    if (kappa <= 0.0) return twap_holding(total, intervals, j);
    const double num = detail::det_sinh(kappa * static_cast<double>(intervals - j));
    const double den = detail::det_sinh(kappa * static_cast<double>(intervals));
    if (den <= 0.0) return twap_holding(total, intervals, j);  // degenerate guard
    return total * (num / den);
}

// The cosh factor of the Almgren-Chriss second-order recurrence
//   x_{j-1} + x_{j+1} = 2*cosh(kappa)*x_j.
inline double ac_recurrence_factor(double kappa) noexcept {
    return detail::det_cosh(kappa);
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered fill stream.
// Pure, read-only: folds each fill's ORDINAL with its bytes so reordering the
// input changes the hash, and folds the count so truncation/extension cannot
// collide with a prefix.
inline ReplayHash fill_stream_hash(const std::vector<ExecutionFill>& fills) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    std::uint64_t ordinal = 0;
    for (const ExecutionFill& f : fills) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, f.seq);
        h = detail::fnv1a_u64(h, f.child_id);
        h = detail::fnv1a_i64(h, static_cast<std::int64_t>(f.side));
        h = detail::fnv1a_i64(h, f.quantity);
        h = detail::fnv1a_i64(h, f.price);
        h = detail::fnv1a_byte(h, f.is_maker ? 1u : 0u);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the fill count
    return h;
}

// Deterministic FNV-1a-64 hash over the FIXED schedule parameters and the
// derived trajectory. Pure, read-only.
inline ReplayHash schedule_hash(const ExecutionScheduleParams& params,
                                const ExecutionSchedule& sched) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    h = detail::fnv1a_i64(h, params.total_quantity);
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(params.intervals));
    h = detail::fnv1a_i64(h, detail::quantize(params.kappa));

    std::uint64_t ordinal = 0;
    for (double x : sched.holdings) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_i64(h, detail::quantize(x));
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the holdings count

    std::uint64_t tordinal = 0;
    for (double n : sched.trades) {
        h = detail::fnv1a_u64(h, tordinal);
        h = detail::fnv1a_i64(h, detail::quantize(n));
        ++tordinal;
    }
    h = detail::fnv1a_u64(h, tordinal);  // fold the trades count

    h = detail::fnv1a_byte(h, sched.is_twap ? 1u : 0u);
    h = detail::fnv1a_byte(h, sched.front_loaded ? 1u : 0u);
    return h;
}

// Compute the offline Almgren-Chriss optimal-execution schedule for FIXED
// parameters. PURE, READ-ONLY, OFFLINE: the parameters are taken by const
// reference and never modified; the result is a plain value. The work is FIXED
// per interval (a constant number of closed-form evaluations), so the
// computation is deterministic. This cannot mutate, and is not consumed by, any
// matching state. The trajectory is a PLANNING statistic only.
inline ExecutionSchedule compute_execution_schedule(
    const ExecutionScheduleParams& params) {
    ExecutionSchedule sched;
    sched.intervals = params.intervals;
    sched.is_twap = (params.kappa <= 0.0);
    if (params.intervals == 0) {
        sched.schedule_hash = schedule_hash(params, sched);
        return sched;
    }

    const double total = static_cast<double>(params.total_quantity);
    const std::size_t n = params.intervals;

    sched.holdings.reserve(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        sched.holdings.push_back(analytic::ac_holding(total, n, j, params.kappa));
    }

    sched.trades.reserve(n);
    bool strictly_decreasing = true;
    double prev_trade = 0.0;
    for (std::size_t j = 1; j <= n; ++j) {
        const double trade = sched.holdings[j - 1] - sched.holdings[j];
        sched.trades.push_back(trade);
        if (j >= 2) {
            constexpr double kTol = 1e-9;
            if (!(prev_trade > trade + kTol)) strictly_decreasing = false;
        }
        prev_trade = trade;
    }
    // Front-loaded only makes sense (and is only claimed) for the urgent case.
    sched.front_loaded = (!sched.is_twap) && (n >= 2) && strictly_decreasing;

    sched.schedule_hash = schedule_hash(params, sched);
    return sched;
}

// Compute the post-hoc TCA statistics for a recorded fill stream. PURE,
// READ-ONLY, OFFLINE: the input is taken by const reference; the result is a
// plain value. The aggregates are exact integer sums plus deterministic double
// arithmetic; they are MEASUREMENT statistics over already-recorded fills and
// can never rewrite a fill, a price, cash, or book state.
inline TcaResult compute_tca(const TcaInput& in) {
    TcaResult r;
    for (const ExecutionFill& f : in.fills) {
        r.executed_quantity += f.quantity;
        r.notional += f.quantity * static_cast<std::int64_t>(f.price);
    }

    const int side = (in.side >= 0) ? 1 : -1;
    if (r.executed_quantity > 0) {
        r.avg_exec_price =
            static_cast<double>(r.notional) / static_cast<double>(r.executed_quantity);
        r.implementation_shortfall_per_share =
            static_cast<double>(side) *
            (r.avg_exec_price - static_cast<double>(in.arrival_price));
        r.implementation_shortfall_total =
            r.implementation_shortfall_per_share *
            static_cast<double>(r.executed_quantity);
        r.vwap_slippage_per_share =
            static_cast<double>(side) *
            (r.avg_exec_price - static_cast<double>(in.benchmark_vwap));
    }

    r.fill_stream_hash = fill_stream_hash(in.fills);
    return r;
}

// Read-only convenience wrapper. Holds a const view of an externally owned TCA
// input (with its fill stream) and FIXED schedule parameters, and exposes ONLY
// the computations; it has no mutating member and no matching-engine coupling,
// making the offline / one-way-data-flow boundary explicit at the call site. The
// schedule and statistics it returns can be read but never written back into
// matching/replay.
class OptimalExecutionScheduleReplay {
public:
    OptimalExecutionScheduleReplay(const ExecutionScheduleParams& params,
                                   const TcaInput& tca) noexcept
        : params_(params), tca_(&tca) {}

    ExecutionSchedule schedule() const {
        return compute_execution_schedule(params_);
    }
    TcaResult tca() const { return compute_tca(*tca_); }

    ReplayHash schedule_replay_hash() const { return schedule().schedule_hash; }
    ReplayHash fill_replay_hash() const { return fill_stream_hash(tca_->fills); }
    std::size_t fill_count() const noexcept { return tca_->fills.size(); }
    bool empty() const noexcept { return tca_->fills.empty(); }

private:
    ExecutionScheduleParams params_;
    const TcaInput* tca_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline schedule / TCA types do NOT,
// which is the type-level proof that a computed schedule or TCA value cannot be
// wired back into the kernel through these structs. A test supplies a POSITIVE
// CONTROL (a type that DOES expose such a member) to prove the trait actually
// discriminates and is not vacuously false.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// Static proof, evaluated at header-include time, that the offline types are pure
// data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => schedule/TCA values are one-way OUT;
//   * the computations are invocable on a CONST wrapper => read-only diagnostics.
static_assert(std::is_trivially_copyable_v<ExecutionFill>,
              "ExecutionFill must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ExecutionScheduleParams>,
              "ExecutionScheduleParams must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<TcaResult>,
              "TcaResult must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<ExecutionFill>,
              "ExecutionFill must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<TcaResult>,
              "TcaResult must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<ExecutionFill>,
              "ExecutionFill must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ExecutionSchedule>,
              "ExecutionSchedule must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<TcaResult>,
              "TcaResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<OptimalExecutionScheduleReplay>,
              "OptimalExecutionScheduleReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<ExecutionSchedule,
                          decltype(&OptimalExecutionScheduleReplay::schedule),
                          const OptimalExecutionScheduleReplay&>,
    "schedule() must be a read-only (const) diagnostic");
static_assert(
    std::is_invocable_r_v<TcaResult,
                          decltype(&OptimalExecutionScheduleReplay::tca),
                          const OptimalExecutionScheduleReplay&>,
    "tca() must be a read-only (const) diagnostic");

}  // namespace offline
