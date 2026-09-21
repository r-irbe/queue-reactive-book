#pragma once

#include "risk_guard.hpp"
#include "tca.hpp"  // TcaConventions for the one-harness comparability contract

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string_view>
#include <vector>

namespace sim {

struct ParentExecutionOrder {
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;
    Side side = Side::Buy;
    Qty qty = 0;
};

struct ScheduleIds {
    EventSeq first_seq = 0;
    OrderId first_order_id = 0;
};

inline std::vector<StrategyIntent> make_no_trade_schedule() {
    return {};
}

inline StrategyIntent make_market_liquidation_intent(const ParentExecutionOrder& order,
                                                     TimeNs ts,
                                                     ScheduleIds ids = {}) noexcept {
    return StrategyIntent{.seq = ids.first_seq,
                          .ts = ts,
                          .kind = StrategyIntentKind::SubmitMarket,
                          .order_id = ids.first_order_id,
                          .parent_id = order.parent_id,
                          .agent = order.agent,
                          .source = order.source,
                          .side = order.side,
                          .price = 0,
                          .qty = order.qty};
}

inline TimeNs interpolate_schedule_ts(TimeNs start_ts,
                                      TimeNs end_ts,
                                      std::uint32_t index,
                                      std::uint32_t count) noexcept {
    if (count <= 1) return start_ts;
    const auto span = static_cast<std::int64_t>(end_ts - start_ts);
    return static_cast<TimeNs>(
        start_ts + (span * static_cast<std::int64_t>(index)) /
                       static_cast<std::int64_t>(count - 1));
}

inline Qty schedule_slice_qty(Qty total_qty,
                              std::uint32_t index,
                              std::uint32_t count) noexcept {
    if (count == 0) return 0;
    const Qty base = static_cast<Qty>(total_qty / static_cast<Qty>(count));
    const Qty remainder = static_cast<Qty>(total_qty % static_cast<Qty>(count));
    return static_cast<Qty>(base + (index < static_cast<std::uint32_t>(remainder) ? 1 : 0));
}

inline std::vector<StrategyIntent> make_twap_market_schedule(
    const ParentExecutionOrder& order,
    TimeNs start_ts,
    TimeNs end_ts,
    std::uint32_t slices,
    ScheduleIds ids = {}) {
    std::vector<StrategyIntent> out{};
    if (order.qty <= 0 || slices == 0) return out;

    const auto max_slices = static_cast<std::uint32_t>(order.qty);
    const std::uint32_t actual_slices = slices > max_slices ? max_slices : slices;
    out.reserve(actual_slices);

    for (std::uint32_t i = 0; i < actual_slices; ++i) {
        const Qty qty = schedule_slice_qty(order.qty, i, actual_slices);
        if (qty <= 0) continue;
        out.push_back(StrategyIntent{.seq = static_cast<EventSeq>(ids.first_seq + i),
                                     .ts = interpolate_schedule_ts(start_ts, end_ts, i,
                                                                   actual_slices),
                                     .kind = StrategyIntentKind::SubmitMarket,
                                     .order_id = static_cast<OrderId>(ids.first_order_id + i),
                                     .parent_id = order.parent_id,
                                     .agent = order.agent,
                                     .source = order.source,
                                     .side = order.side,
                                     .price = 0,
                                     .qty = qty});
    }
    return out;
}

inline std::vector<StrategyIntent> make_twap_limit_schedule(
    const ParentExecutionOrder& order,
    Tick limit_price,
    TimeNs start_ts,
    TimeNs end_ts,
    std::uint32_t slices,
    ScheduleIds ids = {}) {
    auto out = make_twap_market_schedule(order, start_ts, end_ts, slices, ids);
    for (auto& intent : out) {
        intent.kind = StrategyIntentKind::SubmitLimit;
        intent.price = limit_price;
    }
    return out;
}

// --- Execution-baseline taxonomy under one harness (Tier-2 PROTOTYPE) ---------
//
// PROTOTYPE. The point of a baseline taxonomy is that execution baselines are
// only meaningful when compared under the SAME fees, latency, arrival/terminal
// benchmark, and residual convention -- i.e. the same `TcaConventions`. This
// taxonomy labels each baseline family and binds a run to the conventions it was
// evaluated under, and `baselines_comparable()` enforces that two baseline
// results sit on one footing before they are ranked. Every generator emits
// ordinary tick-valid `StrategyIntent`s through the frozen kernel/replay path --
// never model-implied (phantom) fills.

enum class ExecutionBaselineKind : std::uint8_t {
    NoTrade = 0,
    MarketLiquidation = 1,
    Twap = 2,
    TwapLimit = 3,
    Vwap = 4,
    Pov = 5,  // percentage-of-volume; may under-fill if volume runs out
};

inline constexpr std::string_view baseline_kind_name(ExecutionBaselineKind k) noexcept {
    switch (k) {
    case ExecutionBaselineKind::NoTrade: return "no_trade";
    case ExecutionBaselineKind::MarketLiquidation: return "market_liquidation";
    case ExecutionBaselineKind::Twap: return "twap";
    case ExecutionBaselineKind::TwapLimit: return "twap_limit";
    case ExecutionBaselineKind::Vwap: return "vwap";
    case ExecutionBaselineKind::Pov: return "pov";
    }
    return "unknown";
}

// Binds a baseline run to the parent order it executes and the conventions it is
// scored under. Two runs are comparable iff they share the same conventions and
// the same parent economics (side, qty, source); the `kind` may differ -- that
// is exactly what a baseline horse-race compares.
struct ExecutionBaselineSpec {
    ExecutionBaselineKind kind = ExecutionBaselineKind::NoTrade;
    ParentExecutionOrder order{};
    TcaConventions conventions{};
};

inline bool baselines_comparable(const ExecutionBaselineSpec& a,
                                 const ExecutionBaselineSpec& b) noexcept {
    return a.order.side == b.order.side && a.order.qty == b.order.qty &&
           a.order.source == b.order.source &&
           conventions_comparable(a.conventions, b.conventions);
}

// Deterministic integer allocation of `total_qty` across `weights` using the
// largest-remainder method, so slices are non-negative integers that sum exactly
// to `total_qty`. Zero/empty weights fall back to an equal split. Ties break by
// lowest index for determinism.
inline std::vector<Qty> allocate_by_weight(Qty total_qty,
                                           const std::vector<Qty>& weights) {
    const std::size_t n = weights.size();
    std::vector<Qty> out(n, 0);
    if (total_qty <= 0 || n == 0) return out;

    std::int64_t total_weight = 0;
    for (const Qty w : weights) total_weight += w > 0 ? w : 0;

    if (total_weight == 0) {  // degenerate weights -> equal split
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = schedule_slice_qty(total_qty, static_cast<std::uint32_t>(i),
                                        static_cast<std::uint32_t>(n));
        }
        return out;
    }

    std::vector<std::int64_t> remainder(n, 0);
    std::int64_t allocated = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t w = weights[i] > 0 ? weights[i] : 0;
        const std::int64_t num = static_cast<std::int64_t>(total_qty) * w;
        out[i] = static_cast<Qty>(num / total_weight);
        remainder[i] = num % total_weight;
        allocated += out[i];
    }

    std::int64_t leftover = static_cast<std::int64_t>(total_qty) - allocated;
    std::vector<std::size_t> order_idx(n);
    std::iota(order_idx.begin(), order_idx.end(), std::size_t{0});
    std::sort(order_idx.begin(), order_idx.end(), [&](std::size_t l, std::size_t r) {
        if (remainder[l] != remainder[r]) return remainder[l] > remainder[r];
        return l < r;
    });
    for (std::size_t k = 0; k < order_idx.size() && leftover > 0; ++k, --leftover) {
        out[order_idx[k]] = static_cast<Qty>(out[order_idx[k]] + 1);
    }
    return out;
}

// VWAP: allocate the parent across slices proportional to an exogenous,
// caller-supplied volume profile, completing the parent exactly (largest
// remainder). Profile is treated as relative weights; it is NOT realized in
// real time and does not peek at future fills (no lookahead -- the profile is an
// input assumption, recorded as such).
inline std::vector<StrategyIntent> make_vwap_market_schedule(
    const ParentExecutionOrder& order,
    const std::vector<Qty>& volume_profile,
    TimeNs start_ts,
    TimeNs end_ts,
    ScheduleIds ids = {}) {
    std::vector<StrategyIntent> out{};
    if (order.qty <= 0 || volume_profile.empty()) return out;

    const auto slice_qty = allocate_by_weight(order.qty, volume_profile);
    const auto count = static_cast<std::uint32_t>(volume_profile.size());
    std::uint32_t emitted = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (slice_qty[i] <= 0) continue;
        out.push_back(StrategyIntent{
            .seq = static_cast<EventSeq>(ids.first_seq + emitted),
            .ts = interpolate_schedule_ts(start_ts, end_ts, i, count),
            .kind = StrategyIntentKind::SubmitMarket,
            .order_id = static_cast<OrderId>(ids.first_order_id + emitted),
            .parent_id = order.parent_id,
            .agent = order.agent,
            .source = order.source,
            .side = order.side,
            .price = 0,
            .qty = slice_qty[i]});
        ++emitted;
    }
    return out;
}

// POV: target a fixed participation fraction (participation_num/participation_den)
// of each interval's realized volume. Unlike VWAP this may UNDER-FILL the parent
// if interval volume runs out -- that is a true property of percentage-of-volume,
// not a bug, so the schedule is allowed to sum to less than order.qty. Cumulative
// quantity is capped at the parent size.
inline std::vector<StrategyIntent> make_pov_market_schedule(
    const ParentExecutionOrder& order,
    const std::vector<Qty>& interval_volume,
    std::uint32_t participation_num,
    std::uint32_t participation_den,
    TimeNs start_ts,
    TimeNs end_ts,
    ScheduleIds ids = {}) {
    std::vector<StrategyIntent> out{};
    if (order.qty <= 0 || interval_volume.empty() || participation_den == 0 ||
        participation_num == 0) {
        return out;
    }

    const auto count = static_cast<std::uint32_t>(interval_volume.size());
    Qty remaining = order.qty;
    std::uint32_t emitted = 0;
    for (std::uint32_t i = 0; i < count && remaining > 0; ++i) {
        const std::int64_t vol = interval_volume[i] > 0 ? interval_volume[i] : 0;
        Qty target = static_cast<Qty>((vol * participation_num) / participation_den);
        if (target <= 0) continue;
        if (target > remaining) target = remaining;
        out.push_back(StrategyIntent{
            .seq = static_cast<EventSeq>(ids.first_seq + emitted),
            .ts = interpolate_schedule_ts(start_ts, end_ts, i, count),
            .kind = StrategyIntentKind::SubmitMarket,
            .order_id = static_cast<OrderId>(ids.first_order_id + emitted),
            .parent_id = order.parent_id,
            .agent = order.agent,
            .source = order.source,
            .side = order.side,
            .price = 0,
            .qty = target});
        remaining = static_cast<Qty>(remaining - target);
        ++emitted;
    }
    return out;
}

// Total quantity a schedule submits (for harness checks: VWAP/TWAP complete the
// parent; POV may legitimately under-fill).
inline Qty schedule_total_qty(const std::vector<StrategyIntent>& schedule) noexcept {
    Qty total = 0;
    for (const auto& intent : schedule) total = static_cast<Qty>(total + intent.qty);
    return total;
}

} // namespace sim
