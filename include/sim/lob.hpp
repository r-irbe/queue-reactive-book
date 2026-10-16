#pragma once

#include "types.hpp"
#include <algorithm>
#include <functional>
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
    Qty filled    = 0;
    Qty remaining = 0;
};

struct CompactionConfig {
    std::size_t dead_min  = 1024;
    std::size_t total_min = 2048;
    std::size_t dead_num  = 2; // dead/total >= dead_num/dead_den
    std::size_t dead_den  = 3;
};

// Heuristic arena sizing for PMR-backed LOB allocations.
// Scales with expected live order count and is clamped to a practical range.
inline constexpr std::size_t suggest_arena_bytes(std::size_t expected_orders) noexcept {
    constexpr std::size_t kMinArena = std::size_t{256} * std::size_t{1024};
    constexpr std::size_t kMaxArena = std::size_t{8} * std::size_t{1024} * std::size_t{1024};
    constexpr std::size_t kPerOrderBytes = sizeof(Order) * std::size_t{2}; // order storage + container/index overhead

    const std::size_t estimated = expected_orders * kPerOrderBytes;
    return std::clamp(estimated, kMinArena, kMaxArena);
}

// One price level: flat FIFO queue.  `head` advances past consumed/tombstoned
// entries; `volume` tracks live qty inline (no separate map needed).
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

// Price-time priority limit-order book.
//
// Complexity:
//   add (no cross):  O(log P)      P = distinct price levels
//   add (full cross):O(log P + F)  F = number of fills
//   cancel:          O(log P)      direct index -- no per-level scan
//   best_bid/ask:    O(1)
//   volume_at:       O(log P)
//
// Invariants after every public mutation:
//   - Live resting orders have qty > 0.
//   - No crossed book when both sides are non-empty.
//   - level.volume == sum of live qty in that level.
//
// Supports Self-Trade Prevention (STP) policies and ground-truth L3 queue position.
class LimitOrderBook {
public:
    // Pre-reserve id-index for estimated peak live order count.
    explicit LimitOrderBook(std::size_t expected_orders = 4096,
                            bool enable_compaction = true,
                            CompactionConfig compaction_cfg = {},
                            std::size_t arena_bytes = 0);

    // Submit an order. Book assigns OrderId. Market orders drop unfilled
    // remainder; limit orders rest if not fully filled.
    // Supports explicit or per-order Self-Trade Prevention (STP).
    ExecutionResult add(Order order, StpMode stp = StpMode::None);

    // Cancel by id. O(log P) -- direct tombstone, no level scan.
    bool cancel(OrderId id);

    void set_default_stp_mode(StpMode mode) noexcept { default_stp_mode_ = mode; }
    StpMode default_stp_mode() const noexcept { return default_stp_mode_; }

    // Ground-truth L3 FIFO queue position: returns resting volume ahead of `id` at its price level.
    // Returns std::nullopt if the order is not found, not live, or already fully filled/cancelled.
    std::optional<Qty> queue_ahead(OrderId id) const noexcept;

    void   set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    std::optional<Tick> best_bid() const noexcept;
    std::optional<Tick> best_ask() const noexcept;

    Qty volume_at(Side side, Tick price) const noexcept;

    // Depth snapshot. depth=0 returns all levels.
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

    template <typename BookMap>
    void match_with_side(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, BookMap& book, Side incoming_side, StpMode stp);
    
    void match_against_asks(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp);
    void match_against_bids(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp);
    void maybe_compact_level(Side side, Tick price, Level& level);
    void rest(Order& order);
};

} // namespace sim
