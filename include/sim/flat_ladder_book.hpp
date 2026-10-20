#pragma once

#include "lob.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace sim {

inline constexpr std::uint32_t kNullSlot = std::numeric_limits<std::uint32_t>::max();

// 32-byte intrusive order node (2 nodes per 64-byte cache line).
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

// 16-byte level descriptor holding head and tail slot indices of the FIFO queue.
struct IntrusiveLevel {
    std::uint32_t head_slot   = kNullSlot;
    std::uint32_t tail_slot   = kNullSlot;
    std::uint32_t live_orders = 0;
    Qty           volume      = 0;

    bool empty() const noexcept { return volume == 0; }
};

// Bounded tick-ladder order book backed by a pre-allocated PMR arena, a
// doubly-linked intrusive slot pool for O(1) cancel unlinking, and a two-level
// 64-ary bitmap summary for O(1) best-price maintenance across wide ladders.
class FlatLadderBook {
public:
    explicit FlatLadderBook(Tick min_tick,
                            Tick max_tick,
                            std::size_t expected_orders = 4096,
                            bool enable_compaction = false,
                            CompactionConfig compaction_cfg = {},
                            std::size_t arena_bytes = 0) noexcept;

    static std::unique_ptr<FlatLadderBook> create(Tick min_tick,
                                                  Tick max_tick,
                                                  std::size_t expected_orders = 4096,
                                                  std::size_t arena_bytes = 0) noexcept;

    bool is_valid() const noexcept { return is_valid_; }

    [[gnu::hot]] ExecutionResult add(Order order, TimeNs ts, StpMode stp = StpMode::None);

    // Zero-allocation hot path: writes up to `trade_cap` fills into `trade_buf`
    // (when non-null) and returns total fill counts without heap allocation.
    [[gnu::hot]] MatchSummary add_into(Order order,
                                       TimeNs ts,
                                       Trade* trade_buf,
                                       std::size_t trade_cap,
                                       StpMode stp = StpMode::None) noexcept;

    [[gnu::hot]] bool cancel(OrderId id, TimeNs ts) noexcept;

    void set_default_stp_mode(StpMode mode) noexcept { default_stp_mode_ = mode; }
    StpMode default_stp_mode() const noexcept { return default_stp_mode_; }

    std::optional<Qty> queue_ahead(OrderId id) const noexcept;

    void   set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    bool best_bid(Tick& price, Qty& volume) const noexcept;
    bool best_ask(Tick& price, Qty& volume) const noexcept;

    bool mid_price(double& price) const noexcept;

    [[gnu::hot]] bool mid_tick_x2(Tick& sum) const noexcept;

    Qty volume_at(Side side, Tick price) const noexcept;
    std::vector<BookLevelView> levels(Side side, std::size_t depth = 0) const;

    bool        is_empty()    const noexcept;
    std::size_t order_count() const noexcept { return live_order_count_; }

    void set_compaction_enabled(bool) noexcept {}
    bool compaction_enabled() const noexcept { return false; }
    void set_compaction_config(const CompactionConfig&) noexcept {}
    CompactionConfig compaction_config() const noexcept { return {}; }

    Tick min_tick() const noexcept { return min_tick_; }
    Tick max_tick() const noexcept { return max_tick_; }
    std::size_t tick_count() const noexcept { return tick_count_; }

    Qty total_volume(Side s) const noexcept {
        return total_vol_[static_cast<std::uint8_t>(s) & 1];
    }
    std::uint64_t event_count() const noexcept { return event_count_; }

    InvariantReport check_invariants_report() const noexcept;
    void check_invariants() const noexcept;

private:
    std::pmr::monotonic_buffer_resource arena_;
    std::pmr::unsynchronized_pool_resource pool_;

    std::pmr::vector<IntrusiveLevel> levels_;
    std::pmr::vector<IntrusiveOrderNode> node_pool_;
    std::uint32_t free_head_ = kNullSlot;

    std::pmr::vector<std::uint8_t> level_side_;
    std::pmr::vector<std::uint64_t> level_bitmap_;
    std::pmr::vector<std::uint64_t> summary_bitmap_;
    std::pmr::vector<OrderRef> id_refs_;
    std::pmr::vector<std::uint8_t> id_live_;
    std::size_t live_order_count_ = 0;
    Qty total_vol_[2] = {0, 0};

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

    bool can_fill_fok(Side side, Tick price, Qty qty, bool is_market,
                      AgentId taker_agent = 0,
                      StpMode stp = StpMode::None) const noexcept;

    template <Side TakerSide, bool IsLimit, typename TradeSink>
    void match_incoming(Order& incoming, TradeSink&& on_trade,
                        Tick limit_price, TimeNs ts, StpMode stp);

    void mark_level_occupied(std::size_t idx) noexcept;
    void clear_level_occupied(std::size_t idx) noexcept;
    void refresh_best_bid_from(std::size_t start) noexcept;
    void refresh_best_ask_from(std::size_t start) noexcept;

    StpMode default_stp_mode_ = StpMode::None;
};

} // namespace sim
