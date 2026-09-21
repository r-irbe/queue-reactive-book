#pragma once

#include "risk_guard.hpp"  // StrategyIntent, RiskLimits, StrategyIntentKind

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace sim {

// --- Guarded market-making quote adapters (Tier-2 PROTOTYPE) -------------------
//
// PROTOTYPE. Avellaneda-Stoikov (AS) and related quote models computed as
// GUARDED adapters that emit ordinary tick-valid SubmitLimit intents ABOVE the
// replay path. The seductive-wrong-takeaway this design refuses (roadmap T4):
// AS/GLFT/CJP assume a fill-intensity law lambda(delta)=A*exp(-k*delta); if the
// model self-reports fills through that intensity it fabricates PnL. Here the
// adapter ONLY proposes quotes -- it has no fill, intensity, or PnL output, and
// fills are decided solely by the frozen kernel/replay. The header never touches
// `flat_ladder_book.*` and uses no RNG/wall-clock/mutable global state, so it is
// replay-hash invariant.
//
// Units (a deliberate guard against the classic AS unit bug):
//   * `mid_x2` is a DOUBLED-tick coordinate (int64); converted once to a tick
//     price double `mid = 0.5*mid_x2`. All AS math is in tick-price doubles, and
//     only the final bid/ask are snapped to integer ticks.
//   * `sigma2_per_second` is variance per second; tau is seconds, derived from
//     event-time ns. The product sigma2*tau is therefore unit-consistent.

enum class QuoteModelKind : std::uint8_t {
    AvellanedaStoikov = 0,
    // Deferred until calibration data + exact closed forms. Implementing their
    // constants from memory would overclaim correctness, so they are honest
    // non-fabricating placeholders: `make_guarded_quote` returns model_supported
    // = false and suppresses both sides (zero intents -> cannot fabricate fills).
    GlftHeuristic = 1,
    CjpHeuristic = 2,
};

struct QuoteModelParams {
    double risk_aversion = 0.0;       // gamma > 0
    double sigma2_per_second = 0.0;   // variance per second, >= 0
    double fill_k = 0.0;              // intensity decay k > 0 (spread parameter only)
    Qty quote_qty = 0;                // > 0
};

struct QuoteInputs {
    std::int64_t mid_x2 = 0;  // doubled-tick mid (int64 to avoid Tick overflow)
    Qty inventory = 0;        // signed; long > 0, short < 0
    TimeNs now = 0;
    TimeNs horizon_ts = 0;    // tau = max(0, horizon_ts - now) in seconds
};

struct GuardedQuote {
    bool model_supported = false;  // false: model can't quote (bad params / placeholder)
    bool has_bid = false;
    bool has_ask = false;
    Tick bid_px = 0;
    Tick ask_px = 0;
    Qty bid_qty = 0;
    Qty ask_qty = 0;
    bool bid_suppressed = false;  // a guard removed an otherwise-computed bid
    bool ask_suppressed = false;
};

namespace detail {

inline bool finite_positive(double x) noexcept { return std::isfinite(x) && x > 0.0; }

inline bool tick_in_range(double v) noexcept {
    return std::isfinite(v) &&
           v >= static_cast<double>(std::numeric_limits<Tick>::min()) &&
           v <= static_cast<double>(std::numeric_limits<Tick>::max());
}

inline std::optional<Tick> snap_floor(double v) noexcept {
    const double f = std::floor(v);
    if (!tick_in_range(f)) return std::nullopt;
    return static_cast<Tick>(f);
}

inline std::optional<Tick> snap_ceil(double v) noexcept {
    const double c = std::ceil(v);
    if (!tick_in_range(c)) return std::nullopt;
    return static_cast<Tick>(c);
}

inline double tau_seconds(TimeNs now, TimeNs horizon_ts) noexcept {
    if (horizon_ts <= now) return 0.0;  // branch before subtracting (no overflow)
    return static_cast<double>(horizon_ts - now) * 1e-9;
}

// True iff a [min,max] band is meaningfully configured (default RiskLimits has
// min==max==0, which means "no band").
inline bool band_active(const RiskLimits& limits) noexcept {
    return limits.max_price > limits.min_price;
}

} // namespace detail

// Avellaneda-Stoikov, computed exactly. reservation r = mid - q*gamma*sigma2*tau;
// half-spread delta = 0.5*gamma*sigma2*tau + (1/gamma)*ln(1 + gamma/k). Long
// inventory (q>0) lowers r, shifting BOTH quotes down (eager to sell, reluctant
// to buy); short inventory raises them. Guards suppress rather than distort.
inline GuardedQuote make_as_quote(const QuoteModelParams& params,
                                  const QuoteInputs& inputs,
                                  const RiskLimits& limits = {}) noexcept {
    GuardedQuote out{};

    // Parameter validity: an invalid model cannot quote at all.
    if (!detail::finite_positive(params.risk_aversion) ||
        !detail::finite_positive(params.fill_k) ||
        !(std::isfinite(params.sigma2_per_second) && params.sigma2_per_second >= 0.0) ||
        params.quote_qty <= 0) {
        return out;  // model_supported stays false
    }
    out.model_supported = true;

    const double mid = 0.5 * static_cast<double>(inputs.mid_x2);
    const double tau = detail::tau_seconds(inputs.now, inputs.horizon_ts);
    const double gamma = params.risk_aversion;
    const double sigma2 = params.sigma2_per_second;

    const double r = mid - static_cast<double>(inputs.inventory) * gamma * sigma2 * tau;
    const double delta =
        0.5 * gamma * sigma2 * tau + (1.0 / gamma) * std::log1p(gamma / params.fill_k);
    const double bid_d = r - delta;
    const double ask_d = r + delta;
    if (!std::isfinite(bid_d) || !std::isfinite(ask_d)) {
        out.model_supported = false;  // numerics blew up -> no quote
        return out;
    }

    const auto bid_snapped = detail::snap_floor(bid_d);  // never bid tighter
    const auto ask_snapped = detail::snap_ceil(ask_d);   // never ask tighter
    if (!bid_snapped || !ask_snapped) return out;        // out of Tick range
    Tick bid_px = *bid_snapped;
    Tick ask_px = *ask_snapped;

    // Degenerate / crossed after snapping: suppress rather than invent a spread.
    if (ask_px <= bid_px) return out;

    bool keep_bid = true;
    bool keep_ask = true;

    // Price band: suppress out-of-band sides (never clamp toward mid, which would
    // quote MORE aggressively than the model intended).
    if (detail::band_active(limits)) {
        if (bid_px < limits.min_price) keep_bid = false;
        if (ask_px > limits.max_price) keep_ask = false;
    }

    // Inventory cap: do not add to an exposure already at the limit.
    if (limits.max_abs_inventory > 0) {
        if (inputs.inventory >= limits.max_abs_inventory) keep_bid = false;   // long: stop buying
        if (inputs.inventory <= -limits.max_abs_inventory) keep_ask = false;  // short: stop selling
    }

    // Order-size guards (0 = unlimited), per side for notional.
    if (limits.max_order_qty > 0 && params.quote_qty > limits.max_order_qty) {
        keep_bid = false;
        keep_ask = false;
    }
    if (limits.max_order_notional > 0) {
        const std::int64_t qty = params.quote_qty;
        if (static_cast<std::int64_t>(bid_px) * qty > limits.max_order_notional)
            keep_bid = false;
        if (static_cast<std::int64_t>(ask_px) * qty > limits.max_order_notional)
            keep_ask = false;
    }

    out.has_bid = keep_bid;
    out.has_ask = keep_ask;
    out.bid_suppressed = !keep_bid;
    out.ask_suppressed = !keep_ask;
    if (keep_bid) {
        out.bid_px = bid_px;
        out.bid_qty = params.quote_qty;
    }
    if (keep_ask) {
        out.ask_px = ask_px;
        out.ask_qty = params.quote_qty;
    }
    return out;
}

// Dispatch by model kind. Only AS is implemented; GLFT/CJP are honest
// non-fabricating placeholders (model_supported = false, no quote) pending
// calibration data and exact closed forms.
inline GuardedQuote make_guarded_quote(QuoteModelKind kind,
                                       const QuoteModelParams& params,
                                       const QuoteInputs& inputs,
                                       const RiskLimits& limits = {}) noexcept {
    switch (kind) {
    case QuoteModelKind::AvellanedaStoikov:
        return make_as_quote(params, inputs, limits);
    case QuoteModelKind::GlftHeuristic:
    case QuoteModelKind::CjpHeuristic:
        return GuardedQuote{};  // deferred: never fabricates a quote or a fill
    }
    return GuardedQuote{};
}

struct QuoteIntentIds {
    EventSeq first_seq = 0;
    OrderId first_order_id = 0;
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;
};

// Lower a guarded quote to ordinary SubmitLimit intents. NEVER emits a fill or a
// market order: execution is the kernel/replay's job, not the model's.
inline std::vector<StrategyIntent> to_quote_intents(const GuardedQuote& quote,
                                                    TimeNs ts,
                                                    QuoteIntentIds ids = {}) {
    std::vector<StrategyIntent> out{};
    if (!quote.model_supported) return out;
    std::uint64_t offset = 0;
    const auto emit = [&](Side side, Tick price, Qty qty) {
        out.push_back(StrategyIntent{
            .seq = static_cast<EventSeq>(ids.first_seq + offset),
            .ts = ts,
            .kind = StrategyIntentKind::SubmitLimit,
            .order_id = static_cast<OrderId>(ids.first_order_id + offset),
            .parent_id = ids.parent_id,
            .agent = ids.agent,
            .source = ids.source,
            .side = side,
            .price = price,
            .qty = qty});
        ++offset;
    };
    if (quote.has_bid) emit(Side::Buy, quote.bid_px, quote.bid_qty);
    if (quote.has_ask) emit(Side::Sell, quote.ask_px, quote.ask_qty);
    return out;
}

} // namespace sim
