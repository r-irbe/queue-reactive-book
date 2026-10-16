#pragma once

#include "lob.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace sim {

// Sentinel for "no slot / invalid slot" in intrusive slotmap.
inline constexpr std::uint32_t kNullSlot = std::numeric_limits<std::uint32_t>::max();

// Intrusive node layout: 32 bytes aligned (2 nodes per 64-byte cache line).
// Stores resting order data and bidirectional slot indices for true O(1) unlinking.
struct alignas(32) IntrusiveOrderNode {
    OrderId       id         = 0;          // offset  0
    TimeNs        enqueue_ts = 0;          // offset  8
    Qty           qty        = 0;          // offset 16
    AgentId       agent      = 0;          // offset 20
    std::uint32_t prev_slot  = kNullSlot;  // offset 24
    std::uint32_t next_slot  = kNullSlot;  // offset 28
};
static_assert(sizeof(IntrusiveOrderNode) == 32,
              "IntrusiveOrderNode must be 32 bytes for cache line packing");

// Level representation: 16 bytes. Holds head and tail slot indices of the FIFO queue.
struct IntrusiveLevel {
    std::uint32_t head_slot   = kNullSlot;
    std::uint32_t tail_slot   = kNullSlot;
    std::uint32_t live_orders = 0;
    Qty           volume      = 0;

    bool empty() const noexcept { return volume == 0; }
};

// Flat price-ladder book with intrusive doubly-linked slotmap for zero-tombstone O(1) unlinking.
//
// Key properties:
//   - Zero tombstones: order cancellation unlinks nodes in O(1) and recycles slots immediately.
//   - Zero compaction pauses: eliminates latency jitter and GC-like compaction passes.
//   - Zero exceptions (-fno-exceptions clean): all hot paths and construction are noexcept.
//   - Extended order types: supports Limit, Market, PostOnly, IOC, and FOK.
//   - Self-Trade Prevention (STP): CancelPassive, CancelAggressive, DecrementAndCancel.
//   - L3 FIFO position tracking: queue_ahead(id) queries resting volume ahead in O(K) where K = queue depth.
class FlatLadderBook {
public:
    // Construct the flat ladder book with bounded tick interval. Never throws.
    explicit FlatLadderBook(Tick min_tick,
                            Tick max_tick,
                            std::size_t expected_orders = 4096,
                            bool enable_compaction = false,
                            CompactionConfig compaction_cfg = {},
                            std::size_t arena_bytes = 0) noexcept;

    // Static factory for optional construction without exceptions.
    static std::unique_ptr<FlatLadderBook> create(Tick min_tick,
                                                  Tick max_tick,
                                                  std::size_t expected_orders = 4096,
                                                  std::size_t arena_bytes = 0) noexcept;

    bool is_valid() const noexcept { return is_valid_; }

    // Insert a new order; crosses match against the opposite side, residual rests if Limit and in-band.
    // Supports Limit, Market, PostOnly, IOC, FOK, and Self-Trade Prevention (STP).
    [[gnu::hot]] ExecutionResult add(Order order, TimeNs ts, StpMode stp = StpMode::None);

    // Cancel by id; returns true if a live order was removed in deterministic O(1). Never throws.
    [[gnu::hot]] bool cancel(OrderId id, TimeNs ts);

    void set_default_stp_mode(StpMode mode) noexcept { default_stp_mode_ = mode; }
    StpMode default_stp_mode() const noexcept { return default_stp_mode_; }

    // Ground-truth L3 FIFO queue position: returns resting volume ahead of `id` at its price level.
    // Returns std::nullopt if the order is not found, not live, or already fully filled/cancelled.
    std::optional<Qty> queue_ahead(OrderId id) const noexcept;

    // Direct clock control for callers that drive `ts` externally.
    void  set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    bool best_bid(Tick& price, Qty& volume) const noexcept;
    bool best_ask(Tick& price, Qty& volume) const noexcept;

    // Floating mid for stats/plots. Convention only.
    bool mid_price(double& price) const noexcept;

    // Integer mid x 2 (sum of best bid + best ask). Lossless, deterministic, integer-only.
    [[gnu::hot]] bool mid_tick_x2(Tick& sum) const noexcept;

    // Displayed aggregate volume at the price level.
    Qty volume_at(Side side, Tick price) const noexcept;
    std::vector<BookLevelView> levels(Side side, std::size_t depth = 0) const;

    // True when both sides are empty.
    bool        is_empty()    const noexcept;
    std::size_t order_count() const noexcept { return live_order_count_; }

    // Compaction is obsolete with zero-tombstone intrusive slotmap, kept for API compatibility.
    void set_compaction_enabled(bool) noexcept {}
    bool compaction_enabled() const noexcept { return false; }
    void set_compaction_config(const CompactionConfig&) noexcept {}
    CompactionConfig compaction_config() const noexcept { return {}; }

    Tick min_tick() const noexcept { return min_tick_; }
    Tick max_tick() const noexcept { return max_tick_; }
    std::size_t tick_count() const noexcept { return tick_count_; }

    // O(1) branchless: book maintains running side totals on rest/match/cancel.
    Qty total_volume(Side s) const noexcept {
        return total_vol_[static_cast<std::uint8_t>(s) & 1];
    }
    std::uint64_t event_count() const noexcept { return event_count_; }

    // Comprehensive invariant audit. Never throws.
    InvariantReport check_invariants_report() const noexcept;
    void check_invariants() const noexcept;

private:
    std::pmr::monotonic_buffer_resource arena_;
    std::pmr::unsynchronized_pool_resource pool_;

    std::pmr::vector<IntrusiveLevel> levels_;
    std::pmr::vector<IntrusiveOrderNode> node_pool_;
    std::uint32_t free_head_ = kNullSlot;

    // 0 = none, 1 = buy, 2 = sell. PMR-allocated for arena locality with levels_.
    std::pmr::vector<std::uint8_t> level_side_;
    // One bit per ladder level (1 = occupied). Enables O(1) bitwise vEB search.
    std::pmr::vector<std::uint64_t> level_bitmap_;
    // 2-level Bitwise van Emde Boas summary mask: 1 bit per 64-level word in level_bitmap_.
    std::pmr::vector<std::uint64_t> summary_bitmap_;
    std::pmr::vector<OrderRef> id_refs_;
    std::pmr::vector<std::uint8_t> id_live_;
    std::size_t live_order_count_ = 0;
    Qty total_vol_[2] = {0, 0}; // [0] = Buy, [1] = Sell

    Tick min_tick_ = 0;
    Tick max_tick_ = -1;
    std::size_t tick_count_ = 0;
    std::uint64_t event_count_ = 0;
    std::size_t best_bid_idx_ = std::numeric_limits<std::size_t>::max();
    std::size_t best_ask_idx_ = std::numeric_limits<std::size_t>::max();
    OrderId next_id_ = 1;
    TimeNs book_time_ = 0;
    bool is_valid_ = true;

    std::pmr::memory_resource* resource() noexcept { return &pool_; }
    const std::pmr::memory_resource* resource() const noexcept { return &pool_; }

    bool in_range(Tick price) const noexcept;
    std::size_t index_of(Tick price) const noexcept;
    Tick price_of(std::size_t idx) const noexcept;

    static constexpr std::uint8_t kNoSide = 0;
    static constexpr std::uint8_t kBuySide = 1;
    static constexpr std::uint8_t kSellSide = 2;
    static constexpr std::uint8_t side_tag(Side side) noexcept {
        return static_cast<std::uint8_t>(static_cast<std::uint8_t>(side) + 1);
    }

    bool has_order_ref(OrderId id) const noexcept;
    void set_order_ref(OrderId id, const OrderRef& ref);
    void clear_order_ref(OrderId id) noexcept;
    OrderRef get_order_ref(OrderId id) const noexcept;

    IntrusiveLevel& level_at(std::size_t idx) noexcept;
    const IntrusiveLevel& level_at(std::size_t idx) const noexcept;

    std::uint32_t alloc_slot();
    void free_slot(std::uint32_t slot) noexcept;

    void rest(Order& order);

    // Pre-flight liquidity probe for Fill-Or-Kill (FOK).
    bool can_fill_fok(Side side, Tick price, Qty qty, bool is_market) const noexcept;

    // Single matching path; side AND limit-vs-market resolved at compile time.
    template <Side TakerSide, bool IsLimit>
    void match_incoming(Order& incoming, ExecutionResult& out,
                        Tick limit_price, TimeNs ts, StpMode stp);

    void mark_level_occupied(std::size_t idx) noexcept;
    void clear_level_occupied(std::size_t idx) noexcept;
    void refresh_best_bid_from(std::size_t start) noexcept;
    void refresh_best_ask_from(std::size_t start) noexcept;

    StpMode default_stp_mode_ = StpMode::None;
};

} // namespace sim
