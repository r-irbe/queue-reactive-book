// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "types.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory_resource>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sim {

struct BookLevelView {
    Tick        price   = 0;
    Qty         volume  = 0;
    std::size_t orders  = 0;  // live order count
};

struct ExecutionResult {
    std::vector<Trade> trades;
    Qty filled        = 0;
    Qty remaining     = 0;
    OrderStatus status = OrderStatus::RejectedInvalid;
    OrderId id        = 0;
};

struct MatchSummary {
    OrderId     id          = 0;
    Qty         filled      = 0;
    Qty         remaining   = 0;
    std::size_t trade_count = 0;
    OrderStatus status      = OrderStatus::RejectedInvalid;
    std::uint8_t stp_cancels = 0;
};

struct CompactionConfig {
    std::size_t dead_min  = 1024;
    std::size_t total_min = 2048;
    std::size_t dead_num  = 2; // dead/total >= dead_num/dead_den
    std::size_t dead_den  = 3;
};

// Monotonic arena sizing for PMR-backed book containers and order pools.
inline constexpr std::size_t suggest_arena_bytes(std::size_t expected_orders) noexcept {
    constexpr std::size_t kMinArena = std::size_t{512} * std::size_t{1024};
    constexpr std::size_t kMaxArena = std::size_t{16} * std::size_t{1024} * std::size_t{1024};
    constexpr std::size_t kPerOrderBytes = sizeof(Order) * std::size_t{5};

    const std::size_t estimated = expected_orders * kPerOrderBytes;
    return std::clamp(estimated, kMinArena, kMaxArena);
}

// One price level: flat FIFO queue. `head` advances past consumed or tombstoned
// entries; `volume` tracks live quantity inline.
struct Level {
    std::pmr::vector<Order> orders;
    std::uint32_t      head   = 0;
    std::uint32_t      live_orders = 0;
    Qty                volume = 0;

    explicit Level(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
        : orders(mr) {}

    bool empty()  const noexcept { return volume == 0; }
    void push(const Order& o)    { orders.push_back(o); ++live_orders; }
};

// Tree-backed price-time priority limit order book (`std::map` price levels +
// flat per-level FIFO vectors with tombstone compaction).
//
// Complexity:
//   add (no cross):  O(log P)      P = distinct price levels
//   add (full cross):O(log P + F)  F = number of fills
//   cancel:          O(log P)      direct index lookup + tombstone
//   best_bid/ask:    O(1)
//   volume_at:       O(log P)
class LimitOrderBook {
public:
    explicit LimitOrderBook(std::size_t expected_orders = 4096,
                            bool enable_compaction = true,
                            CompactionConfig compaction_cfg = {},
                            std::size_t arena_bytes = 0);

    ExecutionResult add(Order order, StpMode stp = StpMode::None);

    MatchOutcome add_outcome_into(const Order& order,
                                  Trade* out,
                                  std::size_t max_trades,
                                  StpMode stp = StpMode::None) noexcept;

    MatchOutcome add_outcome(const Order& order,
                             StpMode stp = StpMode::None) noexcept {
        return add_outcome_into(order, nullptr, 0, stp);
    }

    MatchSummary add_into(Order order,
                          Trade* trade_buf,
                          std::size_t trade_cap,
                          StpMode stp = StpMode::None) noexcept {
        const Qty orig_qty = order.qty;
        const MatchOutcome out = add_outcome_into(order, trade_buf, trade_cap, stp);
        MatchSummary summary{};
        summary.id = out.id;
        summary.filled = out.filled;
        summary.remaining = out.remaining(orig_qty);
        summary.trade_count = out.trade_count;
        summary.status = out.status;
        summary.stp_cancels = out.stp_cancels;
        return summary;
    }

    CancelOutcome cancel_outcome(OrderId id) noexcept;

    bool cancel(OrderId id) noexcept {
        return static_cast<bool>(cancel_outcome(id));
    }

    void set_default_stp_mode(StpMode mode) noexcept { default_stp_mode_ = mode; }
    StpMode default_stp_mode() const noexcept { return default_stp_mode_; }

    std::optional<Qty> queue_ahead(OrderId id) const noexcept;

    void   set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    std::optional<Tick> best_bid() const noexcept;
    std::optional<Tick> best_ask() const noexcept;

    Qty volume_at(Side side, Tick price) const noexcept;

    std::size_t levels_into(Side side, BookLevelView* out, std::size_t max_levels) const noexcept;
    std::vector<BookLevelView> levels(Side side, std::size_t depth = 0) const;

    bool        empty()       const noexcept;
    std::size_t order_count() const noexcept { return id_index_.size(); }

    void set_compaction_enabled(bool enabled) noexcept { enable_compaction_ = enabled; }
    bool compaction_enabled() const noexcept { return enable_compaction_; }
    void set_compaction_config(const CompactionConfig& cfg) noexcept { compaction_cfg_ = cfg; }
    const CompactionConfig& compaction_config() const noexcept { return compaction_cfg_; }

private:
    using BidPair = std::pair<const Tick, Level>;
    using AskPair = std::pair<const Tick, Level>;
    using BidMap = std::map<Tick, Level, std::greater<Tick>, std::pmr::polymorphic_allocator<BidPair>>;
    using AskMap = std::map<Tick, Level, std::less<Tick>, std::pmr::polymorphic_allocator<AskPair>>;

    using IdIndexValue = std::pair<const OrderId, OrderRef>;
    using IdIndex = std::pmr::unordered_map<OrderId, OrderRef>;

    std::pmr::monotonic_buffer_resource arena_;
    std::pmr::unsynchronized_pool_resource pool_;

    BidMap bids_;
    AskMap asks_;
    IdIndex id_index_;

    OrderId next_id_   = 1;
    TimeNs  book_time_ = 0;
    bool    enable_compaction_ = true;
    CompactionConfig compaction_cfg_{};
    StpMode default_stp_mode_ = StpMode::None;

    std::pmr::memory_resource* resource() noexcept { return &pool_; }
    const std::pmr::memory_resource* resource() const noexcept { return &pool_; }

    void match_against_asks(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp);
    void match_against_bids(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp);
    void maybe_compact_level(Side side, Tick price, Level& level);
    void rest(Order& order);
};

} // namespace sim
