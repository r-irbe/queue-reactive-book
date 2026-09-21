#pragma once

#include "book_observables.hpp"

namespace sim {

struct FillRecord {
    EventSeq seq = 0;
    TimeNs ts = 0;
    OrderId maker_id = 0;
    OrderId taker_id = 0;
    ParentOrderId maker_parent_id = 0;
    ParentOrderId taker_parent_id = 0;
    ScenarioId source = 0;
    AgentId maker = 0;
    AgentId taker = 0;
    Side maker_side = Side::Buy;
    Side taker_side = Side::Buy;
    Tick price = 0;
    Qty qty = 0;
    bool has_mid_at_fill = false;
    Tick mid_x2_at_fill = 0;
};

struct MarkoutRecord {
    FillRecord fill{};
    EventSeq horizon_events = 0;
    TimeNs horizon_ns = 0;
    EventSeq eval_seq = 0;
    TimeNs eval_ts = 0;
    bool has_mid_after = false;
    Tick mid_x2_after = 0;
    // Signed from each participant's economic side. Positive is favorable.
    Tick maker_markout_x2 = 0;
    Tick taker_markout_x2 = 0;
};

inline Side maker_side_from_taker(Side taker_side) noexcept {
    return opposite(taker_side);
}

inline Tick signed_markout_x2(Side side, Tick fill_price, Tick future_mid_x2) noexcept {
    const Tick fill_x2 = static_cast<Tick>(fill_price * 2);
    return is_buy(side) ? static_cast<Tick>(future_mid_x2 - fill_x2)
                        : static_cast<Tick>(fill_x2 - future_mid_x2);
}

inline FillRecord make_fill_record(const Trade& trade,
                                   const BookObservables& at_fill,
                                   EventSeq seq = 0,
                                   ParentOrderId maker_parent_id = 0,
                                   ParentOrderId taker_parent_id = 0,
                                   ScenarioId source = 0) noexcept {
    return FillRecord{.seq = seq,
                      .ts = trade.ts,
                      .maker_id = trade.maker_id,
                      .taker_id = trade.taker_id,
                      .maker_parent_id = maker_parent_id,
                      .taker_parent_id = taker_parent_id,
                      .source = source,
                      .maker = trade.maker,
                      .taker = trade.taker,
                      .maker_side = maker_side_from_taker(trade.taker_side),
                      .taker_side = trade.taker_side,
                      .price = trade.price,
                      .qty = trade.qty,
                      .has_mid_at_fill = at_fill.has_mid,
                      .mid_x2_at_fill = at_fill.has_mid ? at_fill.mid_x2 : 0};
}

inline MarkoutRecord make_markout_record(const FillRecord& fill,
                                          const BookObservables& after,
                                          EventSeq eval_seq = 0,
                                          EventSeq horizon_events = 0,
                                          TimeNs horizon_ns = 0) noexcept {
    MarkoutRecord out{.fill = fill,
                      .horizon_events = horizon_events,
                      .horizon_ns = horizon_ns,
                      .eval_seq = eval_seq,
                      .eval_ts = after.ts,
                      .has_mid_after = after.has_mid,
                      .mid_x2_after = after.has_mid ? after.mid_x2 : 0,
                      .maker_markout_x2 = 0,
                      .taker_markout_x2 = 0};
    if (!after.has_mid) return out;

    out.maker_markout_x2 = signed_markout_x2(fill.maker_side, fill.price, after.mid_x2);
    out.taker_markout_x2 = signed_markout_x2(fill.taker_side, fill.price, after.mid_x2);
    return out;
}

} // namespace sim
