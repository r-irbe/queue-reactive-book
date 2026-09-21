#pragma once

#include "fill_attribution.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace sim {

struct ReplayStep {
    TraceEvent input{};
    bool accepted = false;
    bool cancel_accepted = false;
    BookObservables before{};
    BookObservables after{};
    ExecutionResult execution{};
    std::vector<FillRecord> fills{};
    std::vector<TraceEvent> trade_events{};
};

class ReplayDriver {
public:
    explicit ReplayDriver(Tick min_tick,
                          Tick max_tick,
                          std::size_t expected_orders = 4096,
                          bool enable_compaction = true,
                          CompactionConfig compaction_cfg = {},
                          std::size_t arena_bytes = 0)
        : book_(min_tick, max_tick, expected_orders, enable_compaction,
                compaction_cfg, arena_bytes) {}

    ReplayStep apply(const TraceEvent& event) {
        ReplayStep step{};
        step.input = event;
        step.before = make_book_observables(book_, event.seq);

        switch (event.kind) {
        case TraceEventKind::SubmitLimit:
            apply_submit(event, OrderType::Limit, step);
            break;
        case TraceEventKind::SubmitMarket:
            apply_submit(event, OrderType::Market, step);
            break;
        case TraceEventKind::Cancel:
            apply_cancel(event, step);
            break;
        case TraceEventKind::Snapshot:
            step.accepted = true;
            book_.set_time(event.ts);
            break;
        case TraceEventKind::Trade:
        case TraceEventKind::Reject:
            step.accepted = false;
            break;
        }

        step.after = make_book_observables(book_, event.seq);
        return step;
    }

    const FlatLadderBook& book() const noexcept { return book_; }
    FlatLadderBook& book() noexcept { return book_; }

    bool internal_order_id(OrderId external_id, OrderId& internal_id) const {
        const auto it = external_to_internal_.find(external_id);
        if (it == external_to_internal_.end()) return false;
        internal_id = it->second;
        return true;
    }

private:
    FlatLadderBook book_;
    std::unordered_map<OrderId, OrderId> external_to_internal_{};
    std::unordered_map<OrderId, ParentOrderId> internal_to_parent_{};
    OrderId next_internal_order_id_ = 1;

    void apply_submit(const TraceEvent& event,
                      OrderType type,
                      ReplayStep& step) {
        Order order{};
        order.agent = event.agent;
        order.qty = event.qty;
        order.orig_qty = event.qty;
        order.price = event.price;
        order.side = event.side;
        order.type = type;

        const OrderId assigned_internal_id = next_internal_order_id_++;
        if (event.order_id != 0) {
            external_to_internal_[event.order_id] = assigned_internal_id;
        }
        internal_to_parent_[assigned_internal_id] = event.parent_id;

        step.execution = book_.add(order, event.ts);
        step.accepted = true;
        step.fills.reserve(step.execution.trades.size());
        step.trade_events.reserve(step.execution.trades.size());
        for (const Trade& trade : step.execution.trades) {
            step.fills.push_back(make_fill_record(
                trade, step.before, event.seq, parent_id_for(trade.maker_id),
                parent_id_for(trade.taker_id), event.source));
            step.trade_events.push_back(make_trade_event(event, trade));
        }
    }

    void apply_cancel(const TraceEvent& event, ReplayStep& step) {
        OrderId internal_id = event.order_id;
        if (event.order_id != 0) {
            const auto it = external_to_internal_.find(event.order_id);
            if (it != external_to_internal_.end()) {
                internal_id = it->second;
            }
        }

        step.cancel_accepted = book_.cancel(internal_id, event.ts);
        step.accepted = step.cancel_accepted;
    }

    ParentOrderId parent_id_for(OrderId internal_id) const {
        const auto it = internal_to_parent_.find(internal_id);
        if (it == internal_to_parent_.end()) return 0;
        return it->second;
    }

    static TraceEvent make_trade_event(const TraceEvent& source,
                                       const Trade& trade) noexcept {
        return TraceEvent{.seq = source.seq,
                          .ts = trade.ts,
                          .kind = TraceEventKind::Trade,
                          .order_id = trade.taker_id,
                          .maker_id = trade.maker_id,
                          .taker_id = trade.taker_id,
                          .parent_id = source.parent_id,
                          .agent = trade.taker,
                          .counterparty = trade.maker,
                          .source = source.source,
                          .side = trade.taker_side,
                          .price = trade.price,
                          .qty = trade.qty,
                          .liquidity = LiquidityFlag::Aggressive};
    }
};

} // namespace sim
