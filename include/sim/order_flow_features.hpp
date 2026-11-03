// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "export_schema.hpp"
#include "replay_driver.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sim {

inline constexpr std::size_t kOrderFlowFeatureCount = 18;
inline constexpr std::array<ExportColumn, kOrderFlowFeatureCount>
    kOrderFlowFeatureSchema = {{{"seq"},
                                {"ts"},
                                {"accepted"},
                                {"kind"},
                                {"side"},
                                {"source"},
                                {"agent"},
                                {"limit_buy_qty"},
                                {"limit_sell_qty"},
                                {"market_buy_qty"},
                                {"market_sell_qty"},
                                {"cancel_buy_count"},
                                {"cancel_sell_count"},
                                {"trade_buy_qty"},
                                {"trade_sell_qty"},
                                {"signed_submit_qty"},
                                {"signed_trade_qty"},
                                {"fill_count"}}};

struct OrderFlowDelta {
    EventSeq seq = 0;
    TimeNs ts = 0;
    bool accepted = false;
    TraceEventKind kind = TraceEventKind::SubmitLimit;
    ScenarioId source = 0;
    AgentId agent = 0;
    Side side = Side::Buy;
    Qty limit_buy_qty = 0;
    Qty limit_sell_qty = 0;
    Qty market_buy_qty = 0;
    Qty market_sell_qty = 0;
    std::uint64_t cancel_buy_count = 0;
    std::uint64_t cancel_sell_count = 0;
    Qty trade_buy_qty = 0;
    Qty trade_sell_qty = 0;
    Qty filled_qty = 0;
    std::uint64_t fill_count = 0;
};

inline Qty non_negative_qty(Qty qty) noexcept {
    return std::max<Qty>(qty, 0);
}

inline Qty signed_qty(Side side, Qty qty) noexcept {
    const Qty sign = 1 - 2 * static_cast<Qty>(static_cast<std::uint8_t>(side) & 1);
    return qty * sign;
}

inline OrderFlowDelta make_order_flow_delta(const TraceEvent& event,
                                            bool accepted = true) noexcept {
    OrderFlowDelta out{.seq = event.seq,
                       .ts = event.ts,
                       .accepted = accepted,
                       .kind = event.kind,
                       .source = event.source,
                       .agent = event.agent,
                       .side = event.side};
    if (!accepted) return out;

    const Qty qty = non_negative_qty(event.qty);
    switch (event.kind) {
    case TraceEventKind::SubmitLimit:
        if (is_buy(event.side)) out.limit_buy_qty = qty;
        else out.limit_sell_qty = qty;
        break;
    case TraceEventKind::SubmitMarket:
        if (is_buy(event.side)) out.market_buy_qty = qty;
        else out.market_sell_qty = qty;
        break;
    case TraceEventKind::Cancel:
        if (is_buy(event.side)) ++out.cancel_buy_count;
        else ++out.cancel_sell_count;
        break;
    case TraceEventKind::Trade:
        if (is_buy(event.side)) out.trade_buy_qty = qty;
        else out.trade_sell_qty = qty;
        out.filled_qty = qty;
        out.fill_count = qty > 0 ? 1 : 0;
        break;
    case TraceEventKind::Snapshot:
    case TraceEventKind::Reject:
        break;
    }
    return out;
}

inline void add_trade_event(OrderFlowDelta& delta, const TraceEvent& trade) noexcept {
    const Qty qty = non_negative_qty(trade.qty);
    if (is_buy(trade.side)) {
        delta.trade_buy_qty = static_cast<Qty>(delta.trade_buy_qty + qty);
    } else {
        delta.trade_sell_qty = static_cast<Qty>(delta.trade_sell_qty + qty);
    }
    delta.filled_qty = static_cast<Qty>(delta.filled_qty + qty);
    if (qty > 0) ++delta.fill_count;
}

inline OrderFlowDelta make_order_flow_delta(const ReplayStep& step) noexcept {
    OrderFlowDelta out = make_order_flow_delta(step.input, step.accepted);
    out.filled_qty = 0;
    out.fill_count = 0;
    out.trade_buy_qty = 0;
    out.trade_sell_qty = 0;
    for (const TraceEvent& trade : step.trade_events) {
        add_trade_event(out, trade);
    }
    return out;
}

inline Qty signed_submit_qty(const OrderFlowDelta& delta) noexcept {
    return static_cast<Qty>(delta.limit_buy_qty + delta.market_buy_qty -
                            delta.limit_sell_qty - delta.market_sell_qty);
}

inline Qty signed_trade_qty(const OrderFlowDelta& delta) noexcept {
    return static_cast<Qty>(delta.trade_buy_qty - delta.trade_sell_qty);
}

inline std::array<double, kOrderFlowFeatureCount>
export_order_flow_features(const OrderFlowDelta& delta) noexcept {
    return {static_cast<double>(delta.seq),
            static_cast<double>(delta.ts),
            delta.accepted ? 1.0 : 0.0,
            static_cast<double>(delta.kind),
            delta.side == Side::Buy ? 1.0 : -1.0,
            static_cast<double>(delta.source),
            static_cast<double>(delta.agent),
            static_cast<double>(delta.limit_buy_qty),
            static_cast<double>(delta.limit_sell_qty),
            static_cast<double>(delta.market_buy_qty),
            static_cast<double>(delta.market_sell_qty),
            static_cast<double>(delta.cancel_buy_count),
            static_cast<double>(delta.cancel_sell_count),
            static_cast<double>(delta.trade_buy_qty),
            static_cast<double>(delta.trade_sell_qty),
            static_cast<double>(signed_submit_qty(delta)),
            static_cast<double>(signed_trade_qty(delta)),
            static_cast<double>(delta.fill_count)};
}

} // namespace sim
