// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

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

// Compact 32-byte delta token (half a 64B cache line) for O(1) F_2 XOR rollback.
// Structural InsertRest/UnlinkNode operations store node/ref snapshots in the
// side-table indexed by `aux_idx`, so PartialFill touches only 32 bytes.
struct alignas(32) CompactDeltaToken {
    LadderDeltaKind kind         = LadderDeltaKind::InsertRest; // 1B
    Side            order_side   = Side::Buy;                   // 1B
    std::uint8_t    side_xor     = 0;                           // 1B
    std::uint8_t    id_live_xor  = 0;                           // 1B
    std::uint32_t   level_idx    = 0;                           // 4B
    std::uint32_t   slot         = kNullSlot;                   // 4B
    std::uint32_t   vol_xor      = 0;                           // 4B
    std::uint32_t   tot_vol_xor  = 0;                           // 4B
    std::uint32_t   live_xor     = 0;                           // 4B
    std::uint32_t   aux_idx      = 0;                           // 4B
    std::uint32_t   head_xor     = 0;                           // 4B
};
static_assert(sizeof(CompactDeltaToken) == 32,
              "CompactDeltaToken must be 32 bytes (2 tokens per cache line)");

struct StructuralUndoPayload {
    std::uint32_t      tail_xor      = 0;
    std::uint32_t      free_head_xor = 0;
    std::uint32_t      prev_next_xor = 0;
    std::uint32_t      next_prev_xor = 0;
    std::uint64_t      l_bitmap_xor  = 0ULL;
    std::uint64_t      s_bitmap_xor  = 0ULL;
    OrderId            order_id      = 0;
    OrderRef           saved_ref{};
    IntrusiveOrderNode saved_node{};
};

// Self-inverse F_2 XOR / full delta record for external buffer binding compatibility.
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

// Fixed-capacity undo log with uninitialized inline storage (0-byte constructor memset)
// and 32B CompactDeltaToken fast path.
class LadderUndoLog {
public:
    static constexpr std::size_t kDefaultCapacity = 4096U;

    // Empty constructor leaves raw byte buffers uninitialized: 0 bytes of memset!
    LadderUndoLog() noexcept {}

    void bind_external_buffer(CompactDeltaToken* buf, std::size_t cap) noexcept {
        ext_tokens_ = buf;
        ext_cap_ = cap;
        count_ = 0U;
        aux_count_ = 0U;
        overflow_ = false;
    }

    void reset() noexcept {
        count_ = 0U;
        aux_count_ = 0U;
        overflow_ = false;
    }

    void clear() noexcept { reset(); }

    bool push_partial(Side order_side, std::uint32_t level_idx, std::uint32_t slot,
                      std::uint32_t vol_xor, std::uint32_t tot_vol_xor,
                      std::uint32_t live_xor) noexcept {
        CompactDeltaToken* storage = ext_tokens_ ? ext_tokens_ : token_ptr();
        const std::size_t cap = ext_tokens_ ? ext_cap_ : kDefaultCapacity;
        if (count_ >= cap) [[unlikely]] {
            overflow_ = true;
            return false;
        }
        CompactDeltaToken& tok = storage[count_++];
        tok.kind = LadderDeltaKind::PartialFill;
        tok.order_side = order_side;
        tok.side_xor = 0;
        tok.id_live_xor = 0;
        tok.level_idx = level_idx;
        tok.slot = slot;
        tok.vol_xor = vol_xor;
        tok.tot_vol_xor = tot_vol_xor;
        tok.live_xor = live_xor;
        tok.aux_idx = 0;
        tok.head_xor = 0;
        return true;
    }

    bool push_structural(const CompactDeltaToken& tok,
                         const StructuralUndoPayload& payload) noexcept {
        CompactDeltaToken* storage = ext_tokens_ ? ext_tokens_ : token_ptr();
        const std::size_t cap = ext_tokens_ ? ext_cap_ : kDefaultCapacity;
        if (count_ >= cap || aux_count_ >= kDefaultCapacity) [[unlikely]] {
            overflow_ = true;
            return false;
        }
        const auto a_idx = static_cast<std::uint32_t>(aux_count_++);
        aux_ptr()[a_idx] = payload;
        CompactDeltaToken& dst = storage[count_++];
        dst = tok;
        dst.aux_idx = a_idx;
        return true;
    }

    bool push(const LadderDeltaEntry& entry) noexcept {
        if (entry.kind == LadderDeltaKind::PartialFill) {
            return push_partial(entry.order_side, entry.level_idx, entry.slot,
                                entry.vol_xor, entry.tot_vol_xor, entry.live_xor);
        }
        CompactDeltaToken tok{};
        tok.kind = entry.kind;
        tok.order_side = entry.order_side;
        tok.side_xor = entry.side_xor;
        tok.id_live_xor = entry.id_live_xor;
        tok.level_idx = entry.level_idx;
        tok.slot = entry.slot;
        tok.vol_xor = entry.vol_xor;
        tok.tot_vol_xor = entry.tot_vol_xor;
        tok.live_xor = entry.live_xor;
        tok.head_xor = entry.head_xor;

        StructuralUndoPayload payload{};
        payload.tail_xor = entry.tail_xor;
        payload.free_head_xor = entry.free_head_xor;
        payload.prev_next_xor = entry.prev_next_xor;
        payload.next_prev_xor = entry.next_prev_xor;
        payload.l_bitmap_xor = entry.l_bitmap_xor;
        payload.s_bitmap_xor = entry.s_bitmap_xor;
        payload.order_id = entry.order_id;
        payload.saved_ref = entry.saved_ref;
        payload.saved_node = entry.saved_node;
        return push_structural(tok, payload);
    }

    const CompactDeltaToken& pop_token() noexcept {
        const CompactDeltaToken* storage = ext_tokens_ ? ext_tokens_ : token_ptr();
        const CompactDeltaToken& tok = storage[--count_];
        if (tok.kind != LadderDeltaKind::PartialFill && aux_count_ > 0U) {
            --aux_count_;
        }
        return tok;
    }

    const CompactDeltaToken& token_at(std::size_t idx) const noexcept {
        const CompactDeltaToken* storage = ext_tokens_ ? ext_tokens_ : token_ptr();
        return storage[idx];
    }

    const StructuralUndoPayload& payload_at(std::uint32_t aux_idx) const noexcept {
        return aux_ptr()[aux_idx];
    }

    bool push_from(const LadderUndoLog& other, std::size_t idx) noexcept {
        const CompactDeltaToken& tok = other.token_at(idx);
        if (tok.kind == LadderDeltaKind::PartialFill) {
            return push_partial(tok.order_side, tok.level_idx, tok.slot,
                                tok.vol_xor, tok.tot_vol_xor, tok.live_xor);
        }
        return push_structural(tok, other.payload_at(tok.aux_idx));
    }

    std::size_t size() const noexcept { return count_; }
    std::size_t aux_size() const noexcept { return aux_count_; }
    bool empty() const noexcept { return count_ == 0U; }
    bool overflowed() const noexcept { return overflow_; }

    static LadderUndoLog& default_tls_instance() noexcept {
        static thread_local LadderUndoLog tls_log;
        return tls_log;
    }

    std::size_t saved_best_bid_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_best_ask_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_live_order_count = 0;
    std::uint64_t saved_event_count = 0;
    OrderId saved_next_id = 1;
    TimeNs saved_book_time = 0;

private:
    CompactDeltaToken* token_ptr() noexcept {
        return reinterpret_cast<CompactDeltaToken*>(token_storage_);
    }
    const CompactDeltaToken* token_ptr() const noexcept {
        return reinterpret_cast<const CompactDeltaToken*>(token_storage_);
    }
    StructuralUndoPayload* aux_ptr() noexcept {
        return reinterpret_cast<StructuralUndoPayload*>(aux_storage_);
    }
    const StructuralUndoPayload* aux_ptr() const noexcept {
        return reinterpret_cast<const StructuralUndoPayload*>(aux_storage_);
    }

    alignas(32) unsigned char token_storage_[sizeof(CompactDeltaToken) * kDefaultCapacity];
    alignas(32) unsigned char aux_storage_[sizeof(StructuralUndoPayload) * kDefaultCapacity];
    CompactDeltaToken* ext_tokens_ = nullptr;
    std::size_t ext_cap_ = 0U;
    std::size_t count_ = 0U;
    std::size_t aux_count_ = 0U;
    bool overflow_ = false;
};

struct UndoScopeState {
    LadderUndoLog* prev_log = nullptr;
    std::size_t start_count = 0U;
    std::size_t saved_best_bid_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_best_ask_idx = std::numeric_limits<std::size_t>::max();
    std::size_t saved_live_order_count = 0;
    std::uint64_t saved_event_count = 0;
    OrderId saved_next_id = 1;
    TimeNs saved_book_time = 0;
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

    // Register-packed 16B MatchOutcome (rax:rdx) primary zero-allocation hot path.
    [[gnu::hot]] MatchOutcome add_outcome_into(const Order& order,
                                               TimeNs ts,
                                               Trade* out,
                                               std::size_t max_trades,
                                               StpMode stp = StpMode::None) noexcept;

    [[gnu::hot]] MatchOutcome add_outcome(const Order& order,
                                          TimeNs ts,
                                          StpMode stp = StpMode::None) noexcept {
        return add_outcome_into(order, ts, nullptr, 0, stp);
    }

    [[gnu::hot]] MatchOutcome add_outcome_into(const Order& order,
                                               Trade* out,
                                               std::size_t max_trades,
                                               StpMode stp = StpMode::None) noexcept {
        return add_outcome_into(order, book_time_, out, max_trades, stp);
    }

    [[gnu::hot]] MatchOutcome add_outcome(const Order& order,
                                          StpMode stp = StpMode::None) noexcept {
        return add_outcome_into(order, book_time_, nullptr, 0, stp);
    }

    [[gnu::hot]] ExecutionResult add(Order order, TimeNs ts, StpMode stp = StpMode::None);

    // Zero-allocation hot path adapter: writes up to `trade_cap` fills into `trade_buf`
    // (when non-null) and returns total fill counts without heap allocation.
    [[gnu::hot]] MatchSummary add_into(Order order,
                                       TimeNs ts,
                                       Trade* trade_buf,
                                       std::size_t trade_cap,
                                       StpMode stp = StpMode::None) noexcept {
        const Qty orig_qty = order.qty;
        const MatchOutcome out = add_outcome_into(order, ts, trade_buf, trade_cap, stp);
        MatchSummary summary{};
        summary.id = out.id;
        summary.filled = out.filled;
        summary.remaining = out.remaining(orig_qty);
        summary.trade_count = out.trade_count;
        summary.status = out.status;
        summary.stp_cancels = out.stp_cancels;
        return summary;
    }

    // Register-packed 8B CancelOutcome (rax) primary cancellation path.
    [[gnu::hot]] CancelOutcome cancel_outcome(OrderId id, TimeNs ts) noexcept;

    [[gnu::hot]] CancelOutcome cancel_outcome(OrderId id) noexcept {
        return cancel_outcome(id, book_time_);
    }

    [[gnu::hot]] bool cancel(OrderId id, TimeNs ts) noexcept {
        return static_cast<bool>(cancel_outcome(id, ts));
    }

    void set_default_stp_mode(StpMode mode) noexcept { default_stp_mode_ = mode; }
    StpMode default_stp_mode() const noexcept { return default_stp_mode_; }

    std::optional<Qty> queue_ahead(OrderId id) const noexcept;

    // O(1) head / tail / nth resting order ID lookup at a tick level (returns 0 if empty)
    OrderId peek_head_order_id(Side side, Tick price) const noexcept;
    OrderId peek_tail_order_id(Side side, Tick price) const noexcept;
    OrderId peek_nth_order_id(Side side, Tick price, std::size_t n) const noexcept;

    void   set_time(TimeNs t) noexcept { book_time_ = t; }
    TimeNs time()             const noexcept { return book_time_; }

    bool best_bid(Tick& price, Qty& volume) const noexcept;
    bool best_ask(Tick& price, Qty& volume) const noexcept;

    bool mid_price(double& price) const noexcept;

    [[gnu::hot]] bool mid_tick_x2(Tick& sum) const noexcept;

    Qty volume_at(Side side, Tick price) const noexcept;
    std::size_t levels_into(Side side, BookLevelView* out, std::size_t max_levels) const noexcept;
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
    static constexpr std::size_t kMaxScopeDepth = 16U;
    std::array<UndoScopeState, kMaxScopeDepth> scope_stack_{};
    std::uint16_t scope_depth_ = 0U;

    // O(1) queue-prefix acceleration (docs/39_QRLOB_LEAN_GPU_INTEGRATION_BLUEPRINT.md,
    // Area 2): per-level arrival-cumulus watermark, zero allocations at run
    // time (all state preallocated at construction).
    // - entry_cum_[slot] snapshot: cumulative live-chain volume up to and
    //   including the order at insertion time.
    // - Pure-FIFO + head-partial regime (the common case) answers in O(1):
    //       ahead = cum(target) - qty(target) - cum(head) + qty(head).
    // - The first interior cancel of a level permanently selects the exact
    //   O(k) chain walk for that level (conservative, never wrong).
    // - Rollbacks only set wm_dirty (O(1) per token); the next query rebuilds
    //   entry_cum_ from the restored chain, so no undo-token layout changes.
    struct LevelPrefix {
        bool interior_seen = false;
        bool wm_dirty = false;
    };
    std::pmr::vector<LevelPrefix> level_prefix_;
    std::pmr::vector<std::uint64_t> entry_cum_;  // slot -> cumulative arrival volume snapshot

    void watermark_rebuild(std::size_t level_idx) noexcept;
    void prefix_on_removal(std::size_t level_idx, bool interior) noexcept;

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

    template <bool RecordUndo>
    void rest_impl(Order& order);
    void rest(Order& order);

    template <bool RecordUndo>
    void unlink_maker_node_impl(std::size_t idx, std::uint32_t slot, Side maker_side) noexcept;
    void unlink_maker_node(std::size_t idx, std::uint32_t slot, Side maker_side) noexcept;

    bool can_fill_fok(Side side, Tick price, Qty qty, bool is_market,
                      AgentId taker_agent = 0,
                      StpMode stp = StpMode::None) const noexcept;

    template <Side TakerSide, bool IsLimit, bool RecordUndo, typename TradeSink>
    void match_incoming(Order& incoming, TradeSink&& on_trade,
                        Tick limit_price, TimeNs ts, StpMode stp,
                        std::uint8_t& stp_cancels, bool& stp_cancelled_taker);

    std::size_t find_next_occupied_from(std::size_t start) const noexcept;
    std::size_t find_prev_occupied_from(std::size_t start) const noexcept;

    void mark_level_occupied(std::size_t idx) noexcept;
    void clear_level_occupied(std::size_t idx) noexcept;
    void refresh_best_bid_from(std::size_t start) noexcept;
    void refresh_best_ask_from(std::size_t start) noexcept;

    StpMode default_stp_mode_ = StpMode::None;
};

// Lightweight 24-byte RAII reversible scope for zero-allocation counterfactual subgame
// rollouts (`CFR+`, subgame resolving, best-response lookahead).
// Eliminates the 384 KiB stack zero-initialization bomb via TLS / external log binding.
class ReversibleSubgameScope {
public:
    explicit ReversibleSubgameScope(FlatLadderBook& book) noexcept
        : book_(&book), ext_log_(nullptr) {
        arm_scope();
    }

    ReversibleSubgameScope(FlatLadderBook& book, LadderUndoLog& external_log) noexcept
        : book_(&book), ext_log_(&external_log) {
        arm_scope();
    }

    ~ReversibleSubgameScope() noexcept {
        if (!committed_ && !rolled_back_) {
            rollback();
        }
    }

    ReversibleSubgameScope(const ReversibleSubgameScope&) = delete;
    ReversibleSubgameScope& operator=(const ReversibleSubgameScope&) = delete;

    MatchSummary add(Order order, TimeNs ts,
                     Trade* trade_buf = nullptr,
                     std::size_t trade_cap = 0,
                     StpMode stp = StpMode::None) noexcept {
        return book_->add_into(order, ts, trade_buf, trade_cap, stp);
    }

    MatchOutcome add_outcome(const Order& order, TimeNs ts,
                             Trade* trade_buf = nullptr,
                             std::size_t trade_cap = 0,
                             StpMode stp = StpMode::None) noexcept {
        return book_->add_outcome_into(order, ts, trade_buf, trade_cap, stp);
    }

    bool cancel(OrderId id, TimeNs ts) noexcept {
        return book_->cancel(id, ts);
    }

    CancelOutcome cancel_outcome(OrderId id, TimeNs ts) noexcept {
        return book_->cancel_outcome(id, ts);
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
        return book_->add_into(mkt, ts, trade_buf, trade_cap, stp);
    }

    void rollback() noexcept {
        if (rolled_back_) return;
        const UndoScopeState& st = book_->scope_stack_[slot_idx_];
        book_->rollback_undo_log_to(*active_log_, st.start_count,
                                    st.saved_best_bid_idx, st.saved_best_ask_idx,
                                    st.saved_live_order_count, st.saved_event_count,
                                    st.saved_next_id, st.saved_book_time);
        book_->active_undo_log_ = st.prev_log;
        if (book_->scope_depth_ > slot_idx_) {
            book_->scope_depth_ = slot_idx_;
        }
        rolled_back_ = true;
    }

    void reset() noexcept {
        if (!rolled_back_ && !committed_) {
            rollback();
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
        return active_log_->size() - book_->scope_stack_[slot_idx_].start_count;
    }

    bool overflowed() const noexcept {
        return active_log_->overflowed();
    }

private:
    void arm_scope() noexcept {
        slot_idx_ = (book_->scope_depth_ < FlatLadderBook::kMaxScopeDepth)
                        ? book_->scope_depth_++
                        : static_cast<std::uint16_t>(FlatLadderBook::kMaxScopeDepth - 1U);
        UndoScopeState& st = book_->scope_stack_[slot_idx_];
        st.prev_log = book_->active_undo_log_;
        st.saved_best_bid_idx = book_->best_bid_idx_;
        st.saved_best_ask_idx = book_->best_ask_idx_;
        st.saved_live_order_count = book_->live_order_count_;
        st.saved_event_count = book_->event_count_;
        st.saved_next_id = book_->next_id_;
        st.saved_book_time = book_->book_time_;

        if (ext_log_ != nullptr) {
            active_log_ = ext_log_;
            if (st.prev_log == ext_log_) {
                st.start_count = ext_log_->size();
            } else {
                ext_log_->reset();
                ext_log_->saved_best_bid_idx = st.saved_best_bid_idx;
                ext_log_->saved_best_ask_idx = st.saved_best_ask_idx;
                ext_log_->saved_live_order_count = st.saved_live_order_count;
                ext_log_->saved_event_count = st.saved_event_count;
                ext_log_->saved_next_id = st.saved_next_id;
                ext_log_->saved_book_time = st.saved_book_time;
                st.start_count = 0U;
            }
            book_->active_undo_log_ = ext_log_;
        } else if (st.prev_log != nullptr) {
            active_log_ = st.prev_log;
            st.start_count = st.prev_log->size();
            book_->active_undo_log_ = st.prev_log;
        } else {
            LadderUndoLog& tls_log = LadderUndoLog::default_tls_instance();
            active_log_ = &tls_log;
            tls_log.reset();
            tls_log.saved_best_bid_idx = st.saved_best_bid_idx;
            tls_log.saved_best_ask_idx = st.saved_best_ask_idx;
            tls_log.saved_live_order_count = st.saved_live_order_count;
            tls_log.saved_event_count = st.saved_event_count;
            tls_log.saved_next_id = st.saved_next_id;
            tls_log.saved_book_time = st.saved_book_time;
            st.start_count = 0U;
            book_->active_undo_log_ = &tls_log;
        }
    }

    void commit_merge() noexcept {
        const UndoScopeState& st = book_->scope_stack_[slot_idx_];
        if (st.prev_log != nullptr && active_log_ != st.prev_log) {
            const std::size_t n = active_log_->size();
            for (std::size_t i = st.start_count; i < n; ++i) {
                st.prev_log->push_from(*active_log_, i);
            }
        }
        book_->active_undo_log_ = st.prev_log;
        if (book_->scope_depth_ > slot_idx_) {
            book_->scope_depth_ = slot_idx_;
        }
    }

    FlatLadderBook* book_ = nullptr;
    LadderUndoLog* ext_log_ = nullptr;
    LadderUndoLog* active_log_ = nullptr;
    std::uint16_t slot_idx_ = 0U;
    bool rolled_back_ = false;
    bool committed_ = false;
};
static_assert(sizeof(ReversibleSubgameScope) <= 32,
              "ReversibleSubgameScope must be <= 32 bytes (no inline LadderUndoLog stack bomb)");

} // namespace sim
