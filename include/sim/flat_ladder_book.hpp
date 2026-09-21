#pragma once

#include "lob.hpp"

#include <cstdint>
#include <limits>

namespace sim {

// Flat price-ladder book for bounded tick ranges.
//
// Layout:
//   - contiguous vector of price levels indexed by (price - min_tick)
//   - each level keeps a FIFO queue of resting orders
//   - best bid/ask are cached indices into the ladder
//
// Favors tight price bands where the ladder fits comfortably in cache.
//
// Time discipline:
//   - `ts` parameters on add()/cancel() drive the book clock and trade timestamps
//     for simulation/replay; the alternative (caller pre-sets via set_time) leaks
//     ordering invariants to callers, so we keep `ts` on the mutating API.
class FlatLadderBook {
public:
    // Construct the flat ladder book with bounded tick interval.
    // ladder_size = max_tick - min_tick + 1; memory is O(ladder_size).
    explicit FlatLadderBook(Tick min_tick,
                            Tick max_tick,
                            std::size_t expected_orders = 4096,
                            bool enable_compaction = true,
                            CompactionConfig compaction_cfg = {},
                            std::size_t arena_bytes = 0);

    // Insert a new order; crosses match against the opposite side, residual rests if Limit and in-band.
    // Exceptions: never throws on the hot path; out-of-band Limits silently drop their residual.
    [[gnu::hot]] ExecutionResult add(Order order, TimeNs ts);
    // Cancel by id; returns true if a live order was removed. Never throws.
    [[gnu::hot]] bool cancel(OrderId id, TimeNs ts);

    // Direct clock control for callers that drive `ts` externally.
    void  set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    bool best_bid(Tick& price, Qty& volume) const noexcept;
    bool best_ask(Tick& price, Qty& volume) const noexcept;

    // Floating mid for stats/plots. Convention only — never use on the hot
    // path: prefer mid_tick_x2() to avoid FP determinism / latency cost.
    bool mid_price(double& price) const;

    // Integer mid × 2 (sum of best bid + best ask). Lossless, deterministic,
    // and integer-only. Returns false if either side is empty. The caller can
    // shift right by 1 if even-spread is guaranteed; otherwise compare twice.
    [[gnu::hot]] bool mid_tick_x2(Tick& sum) const noexcept;

    // Displayed aggregate volume at the price level.
    Qty volume_at(Side side, Tick price) const noexcept;
    std::vector<BookLevelView> levels(Side side, std::size_t depth = 0) const;
    // True when both sides are empty.
    bool        is_empty()    const noexcept;
    std::size_t order_count() const noexcept { return live_order_count_; }

    void set_compaction_enabled(bool enabled) noexcept { enable_compaction_ = enabled; }
    bool compaction_enabled() const noexcept { return enable_compaction_; }
    void set_compaction_config(const CompactionConfig& cfg) noexcept { compaction_cfg_ = cfg; }
    const CompactionConfig& compaction_config() const noexcept { return compaction_cfg_; }

    Tick min_tick() const noexcept { return min_tick_; }
    Tick max_tick() const noexcept { return max_tick_; }
    std::size_t tick_count() const noexcept { return tick_count_; }

    // O(1): book maintains running side totals on rest/match/cancel.
    Qty total_volume(Side s) const noexcept {
        return is_buy(s) ? total_bid_vol_ : total_ask_vol_;
    }
    // Cheap accepted-event counter (add+cancel). Useful for sim health checks.
    std::uint64_t event_count() const noexcept { return event_count_; }
    // Throws std::logic_error on first detected inconsistency. Test/debug only.
    void check_invariants() const;

private:
    using Ladder = std::pmr::vector<Level>;

    std::pmr::monotonic_buffer_resource arena_;
    std::pmr::unsynchronized_pool_resource pool_;
    Ladder levels_;
    // 0 = none, 1 = buy, 2 = sell. PMR-allocated for arena locality with levels_.
    std::pmr::vector<std::uint8_t> level_side_;
    // One bit per ladder level (1 = occupied). Enables O(N/64) refresh of the
    // cached best-price indices on level drain via __builtin_ctzll/clzll.
    std::pmr::vector<std::uint64_t> level_bitmap_;
    std::pmr::vector<OrderRef> id_refs_;
    std::pmr::vector<std::uint8_t> id_live_;
    std::size_t live_order_count_ = 0;
    Qty total_bid_vol_ = 0;
    Qty total_ask_vol_ = 0;

    Tick min_tick_ = 0;
    Tick max_tick_ = -1;
    std::size_t tick_count_ = 0;
    std::uint64_t event_count_ = 0;
    std::size_t best_bid_idx_ = std::numeric_limits<std::size_t>::max();
    std::size_t best_ask_idx_ = std::numeric_limits<std::size_t>::max();
    OrderId next_id_ = 1;
    TimeNs book_time_ = 0;
    bool enable_compaction_ = true;
    CompactionConfig compaction_cfg_{};

    std::size_t ladder_size() const noexcept { return tick_count_; }

    std::pmr::memory_resource* resource() noexcept { return &pool_; }
    const std::pmr::memory_resource* resource() const noexcept { return &pool_; }

    bool in_range(Tick price) const noexcept;
    std::size_t index_of(Tick price) const noexcept;
    Tick price_of(std::size_t idx) const noexcept;
    static constexpr std::uint8_t kNoSide = 0;
    static constexpr std::uint8_t kBuySide = 1;
    static constexpr std::uint8_t kSellSide = 2;
    static constexpr std::uint8_t side_tag(Side side) noexcept {
        return side == Side::Buy ? kBuySide : kSellSide;
    }
    bool has_order_ref(OrderId id) const noexcept;
    void set_order_ref(OrderId id, const OrderRef& ref);
    void clear_order_ref(OrderId id) noexcept;
    OrderRef get_order_ref(OrderId id) const noexcept;
    Level& level_at(std::size_t idx) noexcept;
    const Level& level_at(std::size_t idx) const noexcept;
    void rest(Order& order);
    void maybe_compact_level(std::size_t idx, Level& level);
    // Single matching path; side AND limit-vs-market resolved at compile time.
    // 4 explicit instantiations (Buy/Sell × Limit/Market) eliminate the per-iter
    // `if (is_limit)` and side branches entirely.
    template <Side TakerSide, bool IsLimit>
    void match_incoming(Order& incoming, ExecutionResult& out,
                        Tick limit_price, TimeNs ts);

    // Bitmap-accelerated best-price refresh — O(ladder_size/64) worst case.
    // Maintains `level_bitmap_` so deep ladders avoid linear scans.
    void mark_level_occupied(std::size_t idx) noexcept;
    void clear_level_occupied(std::size_t idx) noexcept;
    void refresh_best_bid_from(std::size_t start) noexcept;
    void refresh_best_ask_from(std::size_t start) noexcept;
};

} // namespace sim
