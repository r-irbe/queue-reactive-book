#pragma once

// =============================================================================
// Offline ADVERSE-SELECTION / EFFECTIVE-vs-REALIZED-SPREAD-DECOMPOSITION replay
// fixture.
//
// by supplying a deterministic, replay-neutral fixture in which a recorded
// (trade-direction, trade-price, mid-at-trade, mid-after-horizon, size) sample is
// reduced to the textbook effective / realized / adverse-selection spread
// decomposition plus a single-bucket order-flow-toxicity proxy -- all post-hoc
// TCA diagnostics -- WITHOUT ever touching matching. It is IMPLEMENTATION
// EVIDENCE ONLY (boundary-constructibility), NOT a proof of any adverse-selection
// or toxicity theory, and NOT engine adoption.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and price/size/time only*; every quantity
//   computed here (a signed effective/realized half-spread in ticks, an
//   adverse-selection (price-impact) component, an order-flow-toxicity ratio, a
//   sequential cumulative adverse-selection path peak) is a PURE OFFLINE
//   diagnostic LABEL over an already-recorded, replayable trade-and-quote sample.
//   Data flows ONE WAY, OUT of the sample and into derived statistics; a derived
//   value may NEVER be fed back as a matching input, a fill/priority/tie-break
//   rule, a fee/inventory/accounting mutation, a replay-hash perturbation of the
//   engine, or a hot-path branch. This header is the deterministic proof of that
//   one-way, kernel-free contract.
//
// HONEST SCOPE (keep visible):
//   The fixture validates the DIAGNOSTIC / DECOMPOSITION DISCIPLINE: the
//   effective half-spread d*(P - M0) splits EXACTLY into a realized half-spread
//   d*(P - M_tau) (the liquidity provider's captured spread net of post-trade
//   drift) plus an adverse-selection / price-impact component d*(M_tau - M0).
//   It proves the decomposition and the toxicity proxy are constructible,
//   deterministic, and replay-neutral, and that they recover hand-checked integer
//   adverse selection (Glosten-Milgrom) or order-flow toxicity (VPIN) is the TRUE
//   mechanism: those are empirically-motivated but contested measurement
//   conventions, not theorems. Nothing here is read back into matching/replay.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-adverse-selection-toxicity-diagnostics` keeps every adverse-selection /
//   flow-toxicity construct (effective/realized-spread decomposition, VPIN
//   bucketing, micro-price, markout drift) strictly in the offline TCA /
//   diagnostics layer, never inside `FlatLadderBook`. Its open
//   `needs-code-cross-validation` channel asked for a worked, replay-neutral
//   example where a (direction, price, mid, mid-after) sample is reduced to a
//   spread-decomposition / toxicity statistic under a diagnostics-on/off replay
//   hash. This fixture supplies exactly that. Given an ORDERED sample of fills, it
//     * counts EXACT integer buy-/sell-initiated splits, total and signed volume
//       -- the order-INVARIANT sample aggregates;
//     * computes the EXACT integer-tick decomposition sums
//       sum_effective_half = sum d*(P - M0),
//       sum_realized_half   = sum d*(P - M_tau),
//       sum_adverse_selection = sum d*(M_tau - M0),
//       which satisfy the per-observation and aggregate IDENTITY
//       effective == realized + adverse_selection EXACTLY in integers -- the
//       order-INVARIANT decomposition statistic, plus their double means;
//     * computes a single-bucket VPIN-style order-flow-toxicity ratio
//       |signed_volume| / total_volume in [0,1] -- the order-INVARIANT toxicity
//       proxy;
//     * tracks the SEQUENTIAL cumulative adverse-selection path and its peak
//       absolute excursion along the sample order -- the order-SENSITIVE path
//       metric (same multiset, different order => different peak, same sums);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED sample plus the derived statistics, so the diagnostic is
//       reproducible and order-sensitive.
//   Two closed-form OFFLINE diagnostic relations are provided as directly-testable
//   `analytic` helpers, faithful to the MOC's routed conventions:
//     * the three signed half-spread components themselves; and
//     * the exact algebraic IDENTITY effective == realized + adverse_selection,
//       which holds tick-for-tick by construction -- the hand-checkable
//       decomposition signature.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered sample: an identical
//   ordered sample yields an identical result and an identical hash; reordering
//   the SAME multiset of fills leaves the buy/sell split, the volumes, the spread
//   decomposition sums, and the toxicity ratio unchanged, yet changes the
//   sequential cumulative adverse-selection peak and the hash (order-sensitive
//   integrity). The computation takes the sample by const reference, returns a
//   plain value, and exposes no API that could mutate or be consumed by matching
//   state.
//
// Adverse-selection / toxicity background (kept self-contained; no equation
// imported from any rendered source image -- these are stylized textbook
// measurement conventions re-stated here for the fixture, and are empirically
// motivated yet contested, NOT theorems):
//   * The effective half-spread d*(P - M0) is the signed cost paid against the
//     quote midpoint at trade time (d = +1 buy-initiated, -1 sell-initiated).
//   * The realized half-spread d*(P - M_tau) measures the liquidity provider's
//     post-trade drift.
//   * Their difference is the adverse-selection / price-impact component
//     d*(M_tau - M0); the identity effective = realized + adverse_selection holds
//     by construction.
//   * VPIN-style order-flow toxicity proxies the fraction of flow that is
//     directionally imbalanced; the single-bucket form here is
//     |sum d*size| / sum size in [0,1].
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price / midpoint unit

// One replayable trade-and-quote observation. Pure integer data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal/time,
// `direction` is the trade-initiation sign (buy +1, sell -1, unknown 0),
// `trade_price` is the executed price P (ticks), `mid_at_trade` is the quote
// midpoint M0 at trade time (ticks), `mid_after` is the midpoint M_tau a fixed
// horizon later (ticks), and `size` is the executed quantity (shares, >= 0). The
// fields are an already-recorded sample; the fixture only reads them.
struct SpreadFill {
    std::uint64_t seq = 0;
    std::int64_t direction = 0;     // d  (buy +1, sell -1, unknown 0)
    std::int64_t trade_price = 0;   // P  (ticks)
    std::int64_t mid_at_trade = 0;  // M0 (ticks)
    std::int64_t mid_after = 0;     // M_tau (ticks)
    std::int64_t size = 0;          // shares (>= 0)
};

// Derived adverse-selection / spread-decomposition diagnostics plus the
// order-sensitive replay hash. Pure value type: copyable, holds no reference to
// any external (matching) state.
struct AdverseSelectionDiagnostics {
    std::size_t n = 0;

    // (1) EXACT integer sample aggregates (order-INVARIANT).
    std::int64_t num_buy_initiated = 0;   // count direction > 0
    std::int64_t num_sell_initiated = 0;  // count direction < 0
    std::int64_t total_volume = 0;        // sum size (over directional fills)
    std::int64_t signed_volume = 0;       // sum direction*size (>0 buy pressure)

    // (2) EXACT integer-tick spread decomposition sums (order-INVARIANT). The
    // aggregate identity sum_effective_half == sum_realized_half +
    // sum_adverse_selection holds tick-for-tick by construction.
    std::int64_t sum_effective_half = 0;     // sum d*(P - M0)
    std::int64_t sum_realized_half = 0;      // sum d*(P - M_tau)
    std::int64_t sum_adverse_selection = 0;  // sum d*(M_tau - M0)

    // Double means of the three components (order-INVARIANT; 0 when n == 0).
    double mean_effective_half = 0.0;
    double mean_realized_half = 0.0;
    double mean_adverse_selection = 0.0;

    // (3) Single-bucket VPIN-style order-flow-toxicity proxy (order-INVARIANT):
    // |signed_volume| / total_volume in [0, 1]. Valid iff total_volume > 0.
    bool toxicity_valid = false;
    double orderflow_toxicity = 0.0;

    // (4) SEQUENTIAL cumulative adverse-selection path peak (order-SENSITIVE):
    // the largest absolute running sum of per-fill adverse-selection components
    // seen while replaying the sample in order.
    std::int64_t peak_abs_cumulative_adverse_selection = 0;

    ReplayHash replay_hash = 0;
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

// Quantize a real statistic to a fixed-point integer so it can be folded into
// the integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// Branch-free integer absolute value over int64 (no overflow on the inputs this
// fixture is exercised with).
inline std::int64_t iabs(std::int64_t v) noexcept { return v < 0 ? -v : v; }

// Normalize a raw direction field to a strict sign in {-1, 0, +1}.
inline std::int64_t dir_sign(std::int64_t d) noexcept {
    return (d > 0) ? 1 : (d < 0 ? -1 : 0);
}

}  // namespace detail

// Analytic reference relations for the spread decomposition, used to validate the
// three signed half-spread components and their exact algebraic identity on
// closed-form cases. Stylized measurement conventions, computed here so the
// fixture is self-checking and self-contained.
namespace analytic {

// Signed effective half-spread: d * (P - M0). The cost paid against the midpoint
// at trade time. `direction` is normalized to a strict sign.
inline std::int64_t effective_half_spread(std::int64_t direction,
                                          std::int64_t trade_price,
                                          std::int64_t mid_at_trade) noexcept {
    return detail::dir_sign(direction) * (trade_price - mid_at_trade);
}

// Signed realized half-spread: d * (P - M_tau). The captured spread relative to
// the midpoint a horizon tau later, i.e. net of post-trade drift.
inline std::int64_t realized_half_spread(std::int64_t direction,
                                         std::int64_t trade_price,
                                         std::int64_t mid_after) noexcept {
    return detail::dir_sign(direction) * (trade_price - mid_after);
}

// Signed adverse-selection / price-impact component: d * (M_tau - M0). The
// midpoint drift in the trade's direction.
inline std::int64_t adverse_selection(std::int64_t direction,
                                      std::int64_t mid_at_trade,
                                      std::int64_t mid_after) noexcept {
    return detail::dir_sign(direction) * (mid_after - mid_at_trade);
}

// The exact decomposition IDENTITY, returned as a bool. By construction
//   d*(P - M0) == d*(P - M_tau) + d*(M_tau - M0)
// holds tick-for-tick in integers; this helper makes that algebra directly
// testable for any (direction, P, M0, M_tau).
inline bool decomposition_identity_holds(std::int64_t direction,
                                         std::int64_t trade_price,
                                         std::int64_t mid_at_trade,
                                         std::int64_t mid_after) noexcept {
    return effective_half_spread(direction, trade_price, mid_at_trade) ==
           realized_half_spread(direction, trade_price, mid_after) +
               adverse_selection(direction, mid_at_trade, mid_after);
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered sample and the
// derived diagnostics. Pure, read-only: folds each observation's ORDINAL with its
// bytes so reordering the input changes the hash; folds the integer aggregates,
// the integer decomposition sums, the quantized means and toxicity ratio, the
// order-sensitive cumulative adverse-selection peak, and the sample count so
// truncation/extension cannot collide with a prefix.
inline ReplayHash adverse_selection_hash(
    const std::vector<SpreadFill>& sample,
    const AdverseSelectionDiagnostics& diag) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const SpreadFill& f : sample) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, f.seq);
        h = detail::fnv1a_i64(h, f.direction);
        h = detail::fnv1a_i64(h, f.trade_price);
        h = detail::fnv1a_i64(h, f.mid_at_trade);
        h = detail::fnv1a_i64(h, f.mid_after);
        h = detail::fnv1a_i64(h, f.size);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the sample count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.n));
    h = detail::fnv1a_i64(h, diag.num_buy_initiated);
    h = detail::fnv1a_i64(h, diag.num_sell_initiated);
    h = detail::fnv1a_i64(h, diag.total_volume);
    h = detail::fnv1a_i64(h, diag.signed_volume);
    h = detail::fnv1a_i64(h, diag.sum_effective_half);
    h = detail::fnv1a_i64(h, diag.sum_realized_half);
    h = detail::fnv1a_i64(h, diag.sum_adverse_selection);
    h = detail::fnv1a_i64(h, detail::quantize(diag.mean_effective_half));
    h = detail::fnv1a_i64(h, detail::quantize(diag.mean_realized_half));
    h = detail::fnv1a_i64(h, detail::quantize(diag.mean_adverse_selection));
    h = detail::fnv1a_byte(h, diag.toxicity_valid ? 1u : 0u);
    h = detail::fnv1a_i64(h, detail::quantize(diag.orderflow_toxicity));
    h = detail::fnv1a_i64(h, diag.peak_abs_cumulative_adverse_selection);
    return h;
}

// Compute the offline adverse-selection / spread-decomposition diagnostics for a
// replayable ordered sample of fills. PURE, READ-ONLY, OFFLINE: the sample is
// taken by const reference and never modified; the result is a plain value. The
// work is FIXED per observation (a single linear pass), so the computation is
// deterministic and never iterates a data-dependent number of times. This cannot
// mutate, and is not consumed by, any matching state.
inline AdverseSelectionDiagnostics compute_adverse_selection_diagnostics(
    const std::vector<SpreadFill>& sample) {
    AdverseSelectionDiagnostics diag;
    diag.n = sample.size();

    // Single pass: exact integer aggregates + decomposition sums + sequential
    // cumulative adverse-selection peak. The aggregates and sums are
    // order-invariant; the cumulative peak is order-sensitive.
    std::int64_t running_adverse = 0;
    for (const SpreadFill& f : sample) {
        const std::int64_t d = detail::dir_sign(f.direction);
        if (d > 0) {
            ++diag.num_buy_initiated;
        } else if (d < 0) {
            ++diag.num_sell_initiated;
        }
        if (d != 0) {
            diag.total_volume += f.size;
            diag.signed_volume += d * f.size;
        }

        const std::int64_t eff =
            analytic::effective_half_spread(d, f.trade_price, f.mid_at_trade);
        const std::int64_t real =
            analytic::realized_half_spread(d, f.trade_price, f.mid_after);
        const std::int64_t adv =
            analytic::adverse_selection(d, f.mid_at_trade, f.mid_after);
        diag.sum_effective_half += eff;
        diag.sum_realized_half += real;
        diag.sum_adverse_selection += adv;

        running_adverse += adv;
        const std::int64_t mag = detail::iabs(running_adverse);
        if (mag > diag.peak_abs_cumulative_adverse_selection) {
            diag.peak_abs_cumulative_adverse_selection = mag;
        }
    }

    if (diag.n > 0) {
        const double inv_n = 1.0 / static_cast<double>(diag.n);
        diag.mean_effective_half =
            static_cast<double>(diag.sum_effective_half) * inv_n;
        diag.mean_realized_half =
            static_cast<double>(diag.sum_realized_half) * inv_n;
        diag.mean_adverse_selection =
            static_cast<double>(diag.sum_adverse_selection) * inv_n;
    }

    if (diag.total_volume > 0) {
        diag.toxicity_valid = true;
        diag.orderflow_toxicity =
            static_cast<double>(detail::iabs(diag.signed_volume)) /
            static_cast<double>(diag.total_volume);
    }

    diag.replay_hash = adverse_selection_hash(sample, diag);
    return diag;
}

// Read-only convenience wrapper. Holds a const view of an externally owned fill
// sample and exposes ONLY the computation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The diagnostics it returns can be read but never
// written back into matching.
class AdverseSelectionReplay {
public:
    explicit AdverseSelectionReplay(
        const std::vector<SpreadFill>& sample) noexcept
        : sample_(&sample) {}

    AdverseSelectionDiagnostics compute() const {
        return compute_adverse_selection_diagnostics(*sample_);
    }
    ReplayHash hash() const { return compute().replay_hash; }
    std::size_t size() const noexcept { return sample_->size(); }
    bool empty() const noexcept { return sample_->empty(); }

private:
    const std::vector<SpreadFill>* sample_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline diagnostic types do NOT,
// which is the type-level proof that an adverse-selection / toxicity value cannot
// be wired back into the kernel through these structs. A test supplies a POSITIVE
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
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<SpreadFill>,
              "SpreadFill must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<AdverseSelectionDiagnostics>,
              "AdverseSelectionDiagnostics must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<SpreadFill>,
              "SpreadFill must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<AdverseSelectionDiagnostics>,
              "AdverseSelectionDiagnostics must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<SpreadFill>,
              "SpreadFill must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<AdverseSelectionDiagnostics>,
              "AdverseSelectionDiagnostics must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<AdverseSelectionReplay>,
              "AdverseSelectionReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<AdverseSelectionDiagnostics,
                          decltype(&AdverseSelectionReplay::compute),
                          const AdverseSelectionReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
