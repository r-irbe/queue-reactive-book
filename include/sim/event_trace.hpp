#pragma once

#include "types.hpp"

#include <cstdint>

namespace sim {
// ParentOrderId, ScenarioId, and EventSeq are defined in types.hpp

enum class TraceEventKind : std::uint8_t {
    SubmitLimit = 0,
    SubmitMarket = 1,
    Cancel = 2,
    Trade = 3,
    Snapshot = 4,
    Reject = 5,
};

enum class LiquidityFlag : std::uint8_t {
    Unknown = 0,
    Passive = 1,
    Aggressive = 2,
};

// Durable replay/TCA record. This intentionally lives outside Order so parent
// ids, source ids, and attribution metadata do not enlarge the matching hot path.
struct TraceEvent {
    EventSeq       seq = 0;
    TimeNs         ts = 0;
    TraceEventKind kind = TraceEventKind::SubmitLimit;
    OrderId        order_id = 0;
    OrderId        maker_id = 0;
    OrderId        taker_id = 0;
    ParentOrderId  parent_id = 0;
    AgentId        agent = 0;
    AgentId        counterparty = 0;
    ScenarioId     source = 0;
    Side           side = Side::Buy;
    Tick           price = 0;
    Qty            qty = 0;
    LiquidityFlag  liquidity = LiquidityFlag::Unknown;
};

} // namespace sim
