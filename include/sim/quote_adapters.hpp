// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

// Guarded market-making quote adapters (`AvellanedaStoikov`, `GlftHeuristic`,
// `CjpHeuristic`) that compute tick-snapped reservation prices and half-spreads
// from `QuoteInputs` and lower valid quotes strictly to `SubmitLimit` intents.

#include "risk_guard.hpp"  // StrategyIntent, RiskLimits
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace sim {

enum class QuoteModelKind : std::uint8_t {
    AvellanedaStoikov = 0,
    // Closed-form Gueant-Lehalle-Fernandez-Tapia (2013) infinite-horizon asymptotic quotes.
    GlftHeuristic = 1,
    // Closed-form Cartea-Jaimungal-Penalva (2015) short-term alpha / OFI-skewed quotes.
    CjpHeuristic = 2,
};

struct QuoteModelParams {
    double risk_aversion = 0.0;       // gamma > 0
    double sigma2_per_second = 0.0;   // variance per second, >= 0
    double fill_k = 0.0;              // intensity decay k > 0 (spread parameter)
    Qty quote_qty = 0;                // > 0
    // Optional GLFT (2013) / CJP (2015) parameters with sensible defaults:
    double arrival_a = 1.5;           // baseline arrival intensity A > 0 (GLFT)
    double glft_tau_eff = 1.0;        // effective asymptotic horizon scale tau_eff >= 0
    Qty q_max = 100;                  // maximum inventory clipping bound for GLFT
    double phi_inventory = 0.05;      // CJP running inventory penalty phi >= 0
    double ofi_alpha_weight = 0.50;   // CJP short-term alpha / OFI sensitivity weight
};

struct QuoteInputs {
    std::int64_t mid_x2 = 0;  // doubled-tick mid (int64 to avoid Tick overflow)
    Qty inventory = 0;        // signed; long > 0, short < 0
    TimeNs now = 0;
    TimeNs horizon_ts = 0;    // tau = max(0, horizon_ts - now) in seconds
    double alpha_ofi = 0.0;   // normalized order-flow imbalance / short-term drift signal in [-1, 1]
};

struct GuardedQuote {
    bool model_supported = false;  // false: model can't quote (bad params)
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

inline GuardedQuote finalize_guarded_quote(double bid_d, double ask_d,
                                           const QuoteModelParams& params,
                                           const QuoteInputs& inputs,
                                           const RiskLimits& limits) noexcept {
    GuardedQuote out{};
    out.model_supported = true;
    if (!std::isfinite(bid_d) || !std::isfinite(ask_d)) {
        out.model_supported = false;
        return out;
    }

    const auto bid_snapped = snap_floor(bid_d);
    const auto ask_snapped = snap_ceil(ask_d);
    if (!bid_snapped || !ask_snapped) return out;
    Tick bid_px = *bid_snapped;
    Tick ask_px = *ask_snapped;

    if (ask_px <= bid_px) return out;

    bool keep_bid = true;
    bool keep_ask = true;

    if (band_active(limits)) {
        if (bid_px < limits.min_price) keep_bid = false;
        if (ask_px > limits.max_price) keep_ask = false;
    }

    if (limits.max_abs_inventory > 0) {
        if (inputs.inventory >= limits.max_abs_inventory) keep_bid = false;
        if (inputs.inventory <= -limits.max_abs_inventory) keep_ask = false;
    }

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

} // namespace detail

// Avellaneda-Stoikov, computed exactly. reservation r = mid - q*gamma*sigma2*tau;
// half-spread delta = 0.5*gamma*sigma2*tau + (1/gamma)*ln(1 + gamma/k). Long
// inventory (q>0) lowers r, shifting BOTH quotes down (eager to sell, reluctant
// to buy); short inventory raises them. Guards suppress rather than distort.
inline GuardedQuote make_as_quote(const QuoteModelParams& params,
                                  const QuoteInputs& inputs,
                                  const RiskLimits& limits = {}) noexcept {
    if (!detail::finite_positive(params.risk_aversion) ||
        !detail::finite_positive(params.fill_k) ||
        !(std::isfinite(params.sigma2_per_second) && params.sigma2_per_second >= 0.0) ||
        params.quote_qty <= 0) {
        return GuardedQuote{};
    }

    const double mid = 0.5 * static_cast<double>(inputs.mid_x2);
    const double tau = detail::tau_seconds(inputs.now, inputs.horizon_ts);
    const double gamma = params.risk_aversion;
    const double sigma2 = params.sigma2_per_second;

    const double r = mid - static_cast<double>(inputs.inventory) * gamma * sigma2 * tau;
    const double delta =
        0.5 * gamma * sigma2 * tau + (1.0 / gamma) * std::log1p(gamma / params.fill_k);
    return detail::finalize_guarded_quote(r - delta, r + delta, params, inputs, limits);
}

// Gueant-Lehalle-Fernandez-Tapia (2013) infinite-horizon asymptotic closed-form quotes:
//   r(s, q) = s - q * gamma * sigma^2 * tau_eff
//   delta^{bid}(q) = (1/gamma) * ln(1 + gamma/k) + ((2*q + 1)/2) * c_glft
//   delta^{ask}(q) = (1/gamma) * ln(1 + gamma/k) - ((2*q - 1)/2) * c_glft
// where c_glft = sqrt( (sigma^2 * gamma) / (2 * A * k) * (1 + gamma/k)^(1 + k/gamma) )
// and inventory `q` is clipped to `[-q_max, +q_max]`.
inline GuardedQuote make_glft_quote(const QuoteModelParams& params,
                                    const QuoteInputs& inputs,
                                    const RiskLimits& limits = {}) noexcept {
    if (!detail::finite_positive(params.risk_aversion) ||
        !detail::finite_positive(params.fill_k) ||
        !(std::isfinite(params.sigma2_per_second) && params.sigma2_per_second >= 0.0) ||
        !detail::finite_positive(params.arrival_a) ||
        params.quote_qty <= 0) {
        return GuardedQuote{};
    }

    const double mid = 0.5 * static_cast<double>(inputs.mid_x2);
    const double gamma = params.risk_aversion;
    const double kappa = params.fill_k;
    const double sigma2 = params.sigma2_per_second;
    const double cap_A = params.arrival_a;
    const double tau_eff = (params.glft_tau_eff > 0.0 && std::isfinite(params.glft_tau_eff))
                               ? params.glft_tau_eff
                               : std::max(0.1, detail::tau_seconds(inputs.now, inputs.horizon_ts));

    const Qty q_bound = (params.q_max > 0) ? params.q_max : 100;
    const double q_clipped = static_cast<double>(
        std::clamp<Qty>(inputs.inventory, -q_bound, q_bound));

    const double base_spread = (1.0 / gamma) * std::log1p(gamma / kappa);
    const double ratio = 1.0 + (gamma / kappa);
    const double exp_term = std::pow(ratio, 1.0 + (kappa / gamma));
    const double c_glft = (std::isfinite(exp_term) && sigma2 > 0.0)
                              ? std::sqrt((sigma2 * gamma / (2.0 * cap_A * kappa)) * exp_term)
                              : 0.0;

    const double r = mid - q_clipped * gamma * sigma2 * tau_eff;
    const double delta_bid = std::max(0.25 * base_spread,
                                      base_spread + 0.5 * (2.0 * q_clipped + 1.0) * c_glft);
    const double delta_ask = std::max(0.25 * base_spread,
                                      base_spread - 0.5 * (2.0 * q_clipped - 1.0) * c_glft);

    return detail::finalize_guarded_quote(r - delta_bid, r + delta_ask, params, inputs, limits);
}

// Cartea-Jaimungal-Penalva (2015) short-term alpha / OFI-skewed market-making quotes:
// Incorporates short-term drift `alpha_ofi` and running inventory penalty `phi * q`
// into the reservation price:
//   r_cjp(s, q, alpha) = s + w_alpha * alpha_ofi - (phi + gamma * sigma^2 * tau) * q
//   delta_cjp = (1 / k) + 0.5 * (1/gamma) * ln(1 + gamma/k)
inline GuardedQuote make_cjp_quote(const QuoteModelParams& params,
                                   const QuoteInputs& inputs,
                                   const RiskLimits& limits = {}) noexcept {
    if (!detail::finite_positive(params.risk_aversion) ||
        !detail::finite_positive(params.fill_k) ||
        !(std::isfinite(params.sigma2_per_second) && params.sigma2_per_second >= 0.0) ||
        !(std::isfinite(params.phi_inventory) && params.phi_inventory >= 0.0) ||
        params.quote_qty <= 0) {
        return GuardedQuote{};
    }

    const double mid = 0.5 * static_cast<double>(inputs.mid_x2);
    const double tau = std::max(0.1, detail::tau_seconds(inputs.now, inputs.horizon_ts));
    const double gamma = params.risk_aversion;
    const double kappa = params.fill_k;
    const double sigma2 = params.sigma2_per_second;
    const double phi = params.phi_inventory;
    const double alpha_sig = std::isfinite(inputs.alpha_ofi) ? inputs.alpha_ofi : 0.0;
    const double alpha_w = std::isfinite(params.ofi_alpha_weight) ? params.ofi_alpha_weight : 0.5;

    const double inv_skew = (phi + gamma * sigma2 * tau) * static_cast<double>(inputs.inventory);
    const double r = mid + alpha_w * alpha_sig - inv_skew;
    const double delta = (1.0 / gamma) * std::log1p(gamma / kappa) + 0.25 * gamma * sigma2 * tau;

    return detail::finalize_guarded_quote(r - delta, r + delta, params, inputs, limits);
}

// Dispatch by model kind across Avellaneda-Stoikov (2008), GLFT (2013), and CJP (2015).
inline GuardedQuote make_guarded_quote(QuoteModelKind kind,
                                       const QuoteModelParams& params,
                                       const QuoteInputs& inputs,
                                       const RiskLimits& limits = {}) noexcept {
    switch (kind) {
    case QuoteModelKind::AvellanedaStoikov:
        return make_as_quote(params, inputs, limits);
    case QuoteModelKind::GlftHeuristic:
        return make_glft_quote(params, inputs, limits);
    case QuoteModelKind::CjpHeuristic:
        return make_cjp_quote(params, inputs, limits);
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
