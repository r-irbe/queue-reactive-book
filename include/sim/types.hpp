// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include <cstdint>
#include <cstddef>

namespace sim {
    using Tick          = std::int32_t;//oil was negative years ago ...
    using Qty           = std::int32_t;//can also be neg. position
    using OrderId       = std::uint64_t;//monotonically increasing
    using ParentOrderId = std::uint64_t;
    using ScenarioId    = std::uint32_t;
    using EventSeq      = std::uint64_t;
    using AgentId       = std::uint32_t;
    using TimeNs        = std::int64_t;//for example clock

    // Latency model constants & profiles
    struct LatencyConfig {
        TimeNs md_delay_ns        = 50'000;
        TimeNs decision_delay_ns  = 10'000;
        TimeNs oe_delay_ns        = 50'000;
        TimeNs jitter_ns          = 0;
    };

    struct LatencyProfile {
        LatencyConfig md;
        LatencyConfig dec;
        LatencyConfig oe;
    };
    // Reduced data storage requirements: 1B compact enums and 24B CompactOrder layout.
    enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

    inline constexpr Side opposite(Side s) noexcept {
        return static_cast<Side>(static_cast<std::uint8_t>(s) ^ 1);
    }

    inline constexpr const char* side_name(Side s) noexcept {
        constexpr const char* kSideNames[2] = {"BUY", "SELL"};
        return kSideNames[static_cast<std::uint8_t>(s) & 1];
    }

    inline constexpr bool is_buy(Side s) noexcept {
        return static_cast<std::uint8_t>(s) == 0;
    }
    inline constexpr bool is_sell(Side s) noexcept {
        return static_cast<std::uint8_t>(s) == 1;
    }

    inline constexpr std::uint8_t side_idx(Side s) noexcept {
        return static_cast<std::uint8_t>(s) & 1;
    }

    // RNG constants: data-driven seed & distribution parameters
    namespace rng_constants {
        inline constexpr std::uint64_t kDefaultSeed = 0x9e3779b97f4a7c15ULL;
        inline constexpr int kDoubleShift = 11;  // xoshiro output >> 11 for unit double
        inline constexpr int kDoubleMantissa = 53;  // bits of precision in [0,1)
    }

    // Shared across both LOB implementations for O(1) order ref lookup on cancel.
    // Layout: 1B side + 3B pad + 4B price + 4B level_idx + 4B slot = 16B.
    // `price` is dead in flat ladder, `level_idx` is dead in tree LOB; keeping a
    // single shared layout avoids virtual dispatch / template overhead in tests
    // and keeps OrderRef cache-line friendly (4 refs per 64B line).
    struct OrderRef {
        Side side;
        Tick price;           // for tree-based LOB; unused in flat ladder
        std::uint32_t level_idx;  // for flat ladder; unused in tree LOB
        std::uint32_t slot;
    };
    static_assert(sizeof(OrderRef) == 16, "OrderRef must be 16B for cache-line packing");

    struct InvariantReport {
        bool ok = true;
        const char* message = "ok";
        explicit operator bool() const noexcept { return ok; }
    };

    enum class OrderType : std::uint8_t {
        Limit    = 0,
        Cancel   = 1,
        Market   = 2,
        PostOnly = 3, // Passive maker only: rejected if crosses the book
        IOC      = 4, // Immediate-Or-Cancel: matches available depth; remainder cancelled
        FOK      = 5, // Fill-Or-Kill: fills 100% of order volume or cancels completely with 0 fills
    };

    enum class EventKind : std::uint8_t {
        Add    = 0,
        Cancel = 1,
        Trade  = 2,
    };

    // Self-Trade Prevention (STP) policies for high-frequency market participants.
    enum class StpMode : std::uint8_t {
        None               = 0, // Match normally (self-trades allowed)
        CancelPassive      = 1, // Cancel resting order (maker) on collision
        CancelAggressive   = 2, // Cancel incoming order (taker) on collision
        DecrementAndCancel = 3, // Decrement both orders by min(taker, maker) qty without trade
    };

    // Exhaustive FOOP order execution status (side-effects and errors as data).
    enum class OrderStatus : std::uint8_t {
        AcceptedRested     = 0,
        FilledComplete     = 1,
        FilledPartialRest  = 2,
        FilledPartialIoc   = 3,
        ExpiredIocZero     = 4,
        RejectedInvalid    = 5,
        RejectedOutOfRange = 6,
        RejectedPostOnly   = 7,
        RejectedFok        = 8,
        CancelledByStp     = 9,
    };

    // Register-packed 16-byte match outcome returned in (rax:rdx) under System V AMD64 ABI.
    struct alignas(16) MatchOutcome {
        OrderId       id          = 0;                             // offset  0 (8B)
        Qty           filled      = 0;                             // offset  8 (4B)
        std::uint16_t trade_count = 0;                             // offset 12 (2B)
        OrderStatus   status      = OrderStatus::RejectedInvalid;  // offset 14 (1B)
        std::uint8_t  stp_cancels = 0;                             // offset 15 (1B)

        constexpr bool is_accepted() const noexcept {
            return status == OrderStatus::AcceptedRested ||
                   status == OrderStatus::FilledComplete ||
                   status == OrderStatus::FilledPartialRest ||
                   status == OrderStatus::FilledPartialIoc ||
                   status == OrderStatus::ExpiredIocZero ||
                   status == OrderStatus::CancelledByStp;
        }

        constexpr bool is_rejected() const noexcept {
            return status == OrderStatus::RejectedInvalid ||
                   status == OrderStatus::RejectedOutOfRange ||
                   status == OrderStatus::RejectedPostOnly ||
                   status == OrderStatus::RejectedFok;
        }

        constexpr bool has_rested() const noexcept {
            return status == OrderStatus::AcceptedRested ||
                   status == OrderStatus::FilledPartialRest;
        }

        constexpr Qty remaining(Qty orig_qty) const noexcept {
            return has_rested() ? ((orig_qty > filled) ? (orig_qty - filled) : 0) : 0;
        }
    };
    static_assert(sizeof(MatchOutcome) == 16, "MatchOutcome must be 16B for rax:rdx register return");

    // Exhaustive FOOP cancellation status.
    enum class CancelStatus : std::uint8_t {
        Cancelled       = 0,
        NotFound        = 1,
        AlreadyInactive = 2,
    };

    // Register-packed 8-byte cancel outcome returned in a single GP register (rax).
    struct alignas(8) CancelOutcome {
        Qty          cancelled_qty = 0;                       // offset 0 (4B)
        std::int16_t price_offset  = 0;                       // offset 4 (2B)
        Side         side          = Side::Buy;               // offset 6 (1B)
        CancelStatus status        = CancelStatus::NotFound;  // offset 7 (1B)

        constexpr explicit operator bool() const noexcept {
            return status == CancelStatus::Cancelled;
        }
    };
    static_assert(sizeof(CancelOutcome) == 8, "CancelOutcome must be 8B for single-register return");

    // Hot matching fields first (id, qty, agent, price) fit in 24 bytes -- one cache line access.
    struct Order {
        OrderId   id         = 0;                // offset  0
        Qty       qty        = 0;                // offset  8  remaining, >0 while live
        AgentId   agent      = 0;                // offset 12
        Qty       orig_qty   = 0;                // offset 16  size at submission
        Tick      price      = 0;                // offset 20  ignored for Market
        Side      side       = Side::Buy;        // offset 24
        OrderType type       = OrderType::Limit; // offset 25
        StpMode   stp        = StpMode::None;    // offset 26  Self-Trade Prevention policy
                                                 // offset 27-31: 5B padding
        TimeNs    enqueue_ts = 0;                // offset 32  book time when queued
    };
    static_assert(sizeof(Order) == 40);

    // Reduced data storage: CompactOrder packed into 24 bytes for high-density transport and replay logging.
    struct alignas(8) CompactOrder {
        OrderId       id         = 0;                // offset  0 (8B)
        TimeNs        ts         = 0;                // offset  8 (8B)
        Tick          price      = 0;                // offset 16 (4B)
        Qty           qty        = 0;                // offset 20 (4B)
    };
    static_assert(sizeof(CompactOrder) == 24, "CompactOrder must be 24B for high-density storage");

    // Layout-stable: hot ids first, ts last so the matching loop only touches
    // the first cache line per Trade until trades are flushed downstream.
    struct Trade {
        OrderId   maker_id   = 0;   // offset  0
        OrderId   taker_id   = 0;   // offset  8
        AgentId   maker      = 0;   // offset 16
        AgentId   taker      = 0;   // offset 20
        Side      taker_side = Side::Buy; // offset 24 (1B + 3B pad)
        Tick      price      = 0;   // offset 28
        Qty       qty        = 0;   // offset 32 (4B + 4B pad to align ts)
        TimeNs    ts         = 0;   // offset 40
    };
    static_assert(sizeof(Trade) == 48, "Trade must be 48B (fits 4/cache-pair, 1.33/line)");

    struct Event {
        EventKind kind   = EventKind::Add;
        OrderId   id     = 0;
        AgentId   agent  = 0;
        Side      side   = Side::Buy;
        Tick      price  = 0;
        Qty       qty    = 0;
        TimeNs    ts     = 0;
    };
}