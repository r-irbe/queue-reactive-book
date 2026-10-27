#pragma once

#include "lob.hpp"

#include <array>
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

class LadderUndoLog;
class ReversibleSubgameScope;

enum class LadderDeltaKind : std::uint8_t {
    InsertRest  = 0,
    UnlinkNode  = 1,
    PartialFill = 2
};

// Self-inverse F_2 XOR / compact delta record for O(1) zero-allocation
// counterfactual rollout rollback (`old ^ delta ^ delta == old`).
struct LadderDeltaEntry {
    LadderDeltaKind kind         = LadderDeltaKind::InsertRest;
    std::uint8_t    side_xor     = 0;
    std::uint8_t    id_live_xor  = 0;
    Side            order_side   = Side::Buy;
    std::uint32_t   level_idx    = 0;
    std::uint32_t   slot         = kNullSlot;
    std::uint32_t   head_xor     = 0;
    std::uint32_t   tail_xor     = 0;
    std::uint32_t   live_xor     = 0;
    std::uint32_t   vol_xor      = 0;
    std::uint32_t   tot_vol_xor  = 0;
    std::uint32_t   free_head_xor = 0;
    std::uint32_t   prev_next_xor = 0;
    std::uint32_t   next_prev_xor = 0;
    std::uint64_t   l_bitmap_xor = 0ULL;
    std::uint64_t   s_bitmap_xor = 0ULL;
    OrderId         order_id     = 0;
    OrderRef        saved_ref{};
    IntrusiveOrderNode saved_node{};
};

// Fixed-capacity stack/arena-backed undo log for zero-allocation game-tree rollouts.
class LadderUndoLog {
public:
    static constexpr std::size_t kDefaultCapacity = 4096U;

    LadderUndoLog() noexcept = default;

    void bind_external_buffer(LadderDeltaEntry* buf, std::size_t cap) noexcept {
        ext_buf_ = buf;
        ext_cap_ = cap;
        count_ = 0U;
        overflow_ = false;
    }

    void reset() noexcept {
        count_ = 0U;
        overflow_ = false;
    }

    bool push(const LadderDeltaEntry& entry) noexcept {
        LadderDeltaEntry* storage = ext_buf_ ? ext_buf_ : inline_entries_.data();
        const std::size_t cap = ext_buf_ ? ext_cap_ : kDefaultCapacity;
        if (count_ >= cap) [[unlikely]] {
            overflow_ = true;
            return false;
        }
        storage[count_++] = entry;
        return true;
    }

    const LadderDeltaEntry& pop() noexcept {
        const LadderDeltaEntry* storage = ext_buf_ ? ext_buf_ : inline_entries_.data();
        return storage[--count_];
    }

    const LadderDeltaEntry& entry_at(std::size_t idx) const noexcept {
        const LadderDeltaEntry* storage = ext_buf_ ? ext_buf_ : inline_entries_.data();
        return storage[idx];
    }

    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0U; }
    bool overflowed() const noexcept { return overflow_; }

    std::size_t saved_best_bid_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_best_ask_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_live_order_count = 0;
    std::uint64_t saved_event_count = 0;
    OrderId saved_next_id = 1;
    TimeNs saved_book_time = 0;

private:
    std::array<LadderDeltaEntry, kDefaultCapacity> inline_entries_{};
    LadderDeltaEntry* ext_buf_ = nullptr;
    std::size_t ext_cap_ = 0U;
    std::size_t count_ = 0U;
    bool overflow_ = false;
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

    // Zero-allocation counterfactual undo log attachment & exact state digest
    void attach_undo_log(LadderUndoLog* log) noexcept;
    void detach_undo_log() noexcept { active_undo_log_ = nullptr; }
    LadderUndoLog* active_undo_log() const noexcept { return active_undo_log_; }
    void rollback_undo_log(LadderUndoLog& log) noexcept;
    void rollback_undo_log_to(LadderUndoLog& log,
                              std::size_t target_count,
                              std::size_t saved_best_bid,
                              std::size_t saved_best_ask,
                              std::size_t saved_live_orders,
                              std::uint64_t saved_events,
                              OrderId saved_next,
                              TimeNs saved_time) noexcept;
    std::uint64_t state_digest() const noexcept;
    bool state_equals(const FlatLadderBook& other) const noexcept;

private:
    friend class ReversibleSubgameScope;

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

    LadderUndoLog* active_undo_log_ = nullptr;

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
    void unlink_maker_node(std::size_t idx, std::uint32_t slot, Side maker_side) noexcept;

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

// RAII reversible scope for zero-allocation counterfactual subgame rollouts
// (`CFR+`, subgame resolving, best-response lookahead).
class ReversibleSubgameScope {
public:
    explicit ReversibleSubgameScope(FlatLadderBook& book) noexcept
        : book_(book), prev_log_(book.active_undo_log()) {
        arm_scope();
    }

    ReversibleSubgameScope(FlatLadderBook& book, LadderUndoLog& external_log) noexcept
        : book_(book), ext_log_(&external_log), prev_log_(book.active_undo_log()) {
        arm_scope();
    }

    ~ReversibleSubgameScope() noexcept {
        if (!committed_) {
            rollback();
        } else if (!rolled_back_) {
            commit_merge();
        }
    }

    ReversibleSubgameScope(const ReversibleSubgameScope&) = delete;
    ReversibleSubgameScope& operator=(const ReversibleSubgameScope&) = delete;

    MatchSummary add(Order order, TimeNs ts,
                     Trade* trade_buf = nullptr,
                     std::size_t trade_cap = 0,
                     StpMode stp = StpMode::None) noexcept {
        return book_.add_into(order, ts, trade_buf, trade_cap, stp);
    }

    bool cancel(OrderId id, TimeNs ts) noexcept {
        return book_.cancel(id, ts);
    }

    MatchSummary execute(Side taker_side, Qty qty, TimeNs ts,
                         AgentId agent = 0,
                         Trade* trade_buf = nullptr,
                         std::size_t trade_cap = 0,
                         StpMode stp = StpMode::None) noexcept {
        Order mkt{};
        mkt.qty = qty;
        mkt.orig_qty = qty;
        mkt.agent = agent;
        mkt.side = taker_side;
        mkt.type = OrderType::Market;
        mkt.stp = stp;
        return book_.add_into(mkt, ts, trade_buf, trade_cap, stp);
    }

    void rollback() noexcept {
        if (rolled_back_) return;
        book_.rollback_undo_log_to(*active_log_, start_count_,
                                   saved_best_bid_idx_, saved_best_ask_idx_,
                                   saved_live_order_count_, saved_event_count_,
                                   saved_next_id_, saved_book_time_);
        book_.active_undo_log_ = prev_log_;
        rolled_back_ = true;
    }

    // Re-arm the scope after a rollback so multiple branches can reuse the same scope.
    void reset() noexcept {
        if (!rolled_back_ && !committed_) {
            rollback();
        } else if (committed_) {
            commit_merge();
        }
        arm_scope();
        rolled_back_ = false;
        committed_ = false;
    }

    void commit() noexcept {
        if (!committed_ && !rolled_back_) {
            commit_merge();
            committed_ = true;
        }
    }

    std::size_t delta_count() const noexcept {
        return active_log_->size() - start_count_;
    }

    bool overflowed() const noexcept {
        return active_log_->overflowed();
    }

private:
    void arm_scope() noexcept {
        saved_best_bid_idx_ = book_.best_bid_idx_;
        saved_best_ask_idx_ = book_.best_ask_idx_;
        saved_live_order_count_ = book_.live_order_count_;
        saved_event_count_ = book_.event_count_;
        saved_next_id_ = book_.next_id_;
        saved_book_time_ = book_.book_time_;

        if (ext_log_ != nullptr) {
            active_log_ = ext_log_;
            if (prev_log_ == ext_log_) {
                start_count_ = ext_log_->size();
            } else {
                ext_log_->reset();
                ext_log_->saved_best_bid_idx = saved_best_bid_idx_;
                ext_log_->saved_best_ask_idx = saved_best_ask_idx_;
                ext_log_->saved_live_order_count = saved_live_order_count_;
                ext_log_->saved_event_count = saved_event_count_;
                ext_log_->saved_next_id = saved_next_id_;
                ext_log_->saved_book_time = saved_book_time_;
                start_count_ = 0U;
            }
            book_.active_undo_log_ = ext_log_;
        } else if (prev_log_ != nullptr) {
            active_log_ = prev_log_;
            start_count_ = prev_log_->size();
            book_.active_undo_log_ = prev_log_;
        } else {
            active_log_ = &local_log_;
            local_log_.reset();
            local_log_.saved_best_bid_idx = saved_best_bid_idx_;
            local_log_.saved_best_ask_idx = saved_best_ask_idx_;
            local_log_.saved_live_order_count = saved_live_order_count_;
            local_log_.saved_event_count = saved_event_count_;
            local_log_.saved_next_id = saved_next_id_;
            local_log_.saved_book_time = saved_book_time_;
            start_count_ = 0U;
            book_.active_undo_log_ = &local_log_;
        }
    }

    void commit_merge() noexcept {
        if (prev_log_ != nullptr && active_log_ != prev_log_) {
            const std::size_t n = active_log_->size();
            for (std::size_t i = start_count_; i < n; ++i) {
                prev_log_->push(active_log_->entry_at(i));
            }
        }
        book_.active_undo_log_ = prev_log_;
    }

    FlatLadderBook& book_;
    LadderUndoLog local_log_{};
    LadderUndoLog* ext_log_ = nullptr;
    LadderUndoLog* prev_log_ = nullptr;
    LadderUndoLog* active_log_ = nullptr;
    std::size_t start_count_ = 0U;
    std::size_t saved_best_bid_idx_ = std::numeric_limits<std::size_t>::max();
    std::size_t saved_best_ask_idx_ = std::numeric_limits<std::size_t>::max();
    std::size_t saved_live_order_count_ = 0;
    std::uint64_t saved_event_count_ = 0;
    OrderId saved_next_id_ = 1;
    TimeNs saved_book_time_ = 0;
    bool rolled_back_ = false;
    bool committed_ = false;
};

} // namespace sim
