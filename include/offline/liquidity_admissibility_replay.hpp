#pragma once

// =============================================================================
// Offline liquidity-admissibility / censoring-bias replay diagnostic.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It has NO dependency on FlatLadderBook, the
//   matching kernel, or any `sim::` symbol. It is a self-contained, header-only
//   one-way diagnostic that reconstructs OBSERVED (displayed) vs TRUE
//   (displayed + hidden) liquidity from a synthetic order-flow trace and
//   quantifies the *censoring bias* an observer incurs when it treats the
//   displayed book size as a fact about available liquidity.
//
//   It is the code/replay falsifier for `moc-liquidity-epistemics`'s load-
//   bearing epistemic claims:
//     - "displayed book size is not a fact about true liquidity"
//     - hidden / iceberg liquidity is censored out of the visible feed
//     - a fill is a *censored* observation, not a measurement of all volume
//   It must NEVER be linked into the matching hot path, and its outputs must
//   NEVER be fed back into any book as matchable quantity. Reconstruction is a
//   strictly READ-ONLY (const) projection of an immutable trace.
//
// WHAT IT IS:
//   Given a synthetic order-flow trace where each resting order carries a
//   `display_qty` (what a public feed shows) and a `total_qty` (display +
//   hidden / iceberg reserve, with total_qty >= display_qty), the diagnostic:
//     1. reconstructs per-price OBSERVED depth (sum of display_qty) and TRUE
//        depth (sum of total_qty), with hidden = true - observed >= 0;
//     2. shows an observer using only displayed depth systematically
//        UNDER-estimates available liquidity by exactly the hidden amount
//        (one-directional censoring bias: observed <= true, always);
//     3. shows that a fill against true liquidity is a right-CENSORED
//        observation whenever latent (hidden) size remains after the fill,
//        so the realized print under-reports total executable volume.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   reconstruct() is a pure function of the (immutable) trace: the same trace
//   yields a byte-identical reconstruction and a stable FNV-1a replay hash,
//   independent of platform, padding, or insertion order of same-price orders
//   (levels are canonicalized by price). Censoring bias is non-negative by
//   construction; it is never written back into any book.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <vector>

namespace offline {

// Book side of a resting order. Offline-only; not a matching-kernel enum.
enum class LiquiditySide : std::uint8_t { Bid = 0, Ask = 1 };

using PriceTick = std::int64_t; // integer tick price (deterministic)
using Qty = std::int64_t;       // integer quantity

// A single resting order in the synthetic order-flow trace. `display_qty` is
// what a public feed exposes; `total_qty` is the true resting size (display +
// hidden/iceberg reserve). Invariant: total_qty >= display_qty >= 0.
struct OrderEvent {
    LiquiditySide side;
    PriceTick price;
    Qty display_qty;
    Qty total_qty;

    constexpr Qty hidden_qty() const {
        return total_qty - display_qty; // >= 0 by invariant
    }
};

// Per-price reconstruction of observed vs true depth at one book level.
struct LevelReconstruction {
    PriceTick price;
    Qty observed_qty; // sum of display_qty at this price (what a feed shows)
    Qty true_qty;     // sum of total_qty at this price (display + hidden)

    // Censoring bias at the level: hidden size invisible to a displayed-only
    // observer. Non-negative by construction (true_qty >= observed_qty).
    constexpr Qty hidden_qty() const { return true_qty - observed_qty; }
};

// Whole-book reconstruction plus the aggregate censoring-bias diagnostic and a
// deterministic replay hash. Pure value type: copyable, no references to any
// external (matching) state.
struct LiquidityReconstruction {
    std::vector<LevelReconstruction> bid_levels; // sorted by price ascending
    std::vector<LevelReconstruction> ask_levels; // sorted by price ascending

    Qty observed_total = 0; // displayed-only liquidity estimate
    Qty true_total = 0;     // true (display + hidden) available liquidity

    // Aggregate censoring bias = true_total - observed_total >= 0. This is the
    // systematic amount by which a displayed-only observer UNDER-estimates
    // available liquidity. Direction is fixed: never negative.
    constexpr Qty censoring_bias() const { return true_total - observed_total; }

    std::uint64_t replay_hash = 0;
};

// Outcome of replaying a marketable order of `requested_qty` against the
// reconstructed book. Demonstrates that a fill is a right-CENSORED observation:
// an observer that sized liquidity from the displayed book under-predicts the
// realized fill, and any latent (hidden) size remaining after the fill is never
// revealed by the print.
struct FillOutcome {
    Qty requested_qty = 0;
    Qty observer_predicted_fill = 0; // max fill an observer infers from display
    Qty realized_fill = 0;           // actual fill against true liquidity
    Qty latent_remaining = 0;        // hidden size still resting after the fill
    bool right_censored = false;     // true iff latent_remaining > 0
};

// Offline liquidity-admissibility replay engine. Build-once from an immutable
// trace; query-only. Every public method is const: the diagnostic can never
// mutate a book or feed reconstructed liquidity back into matching.
class LiquidityAdmissibilityReplay {
public:
    LiquidityAdmissibilityReplay() = default;

    explicit LiquidityAdmissibilityReplay(std::vector<OrderEvent> trace)
        : trace_(std::move(trace)) {}

    std::size_t size() const { return trace_.size(); }
    bool empty() const { return trace_.empty(); }
    const std::vector<OrderEvent>& trace() const { return trace_; }

    // Reconstruct observed vs true liquidity from the trace. Pure, const,
    // deterministic: identical traces yield identical reconstructions and hash.
    LiquidityReconstruction reconstruct() const {
        // Canonicalize by price (std::map => deterministic ascending order,
        // independent of trace insertion order for same-price orders).
        std::map<PriceTick, std::pair<Qty, Qty>> bids; // price -> (observed,true)
        std::map<PriceTick, std::pair<Qty, Qty>> asks;

        for (const OrderEvent& ev : trace_) {
            auto& book = (ev.side == LiquiditySide::Bid) ? bids : asks;
            auto& agg = book[ev.price];
            agg.first += ev.display_qty;
            agg.second += ev.total_qty;
        }

        LiquidityReconstruction out;
        Qty observed_total = 0;
        Qty true_total = 0;

        for (const auto& [price, qtys] : bids) {
            out.bid_levels.push_back({price, qtys.first, qtys.second});
            observed_total += qtys.first;
            true_total += qtys.second;
        }
        for (const auto& [price, qtys] : asks) {
            out.ask_levels.push_back({price, qtys.first, qtys.second});
            observed_total += qtys.first;
            true_total += qtys.second;
        }

        out.observed_total = observed_total;
        out.true_total = true_total;
        out.replay_hash = hash_reconstruction(out);
        return out;
    }

    // Replay a marketable order of `requested_qty` consuming the opposite side
    // (a Bid order lifts the ask side, an Ask order hits the bid side). Returns
    // the censored-observation diagnostic. Read-only: nothing is consumed from
    // the trace; the reconstruction is recomputed each call.
    FillOutcome simulate_fill(LiquiditySide taker_side, Qty requested_qty) const {
        const LiquidityReconstruction r = reconstruct();
        const std::vector<LevelReconstruction>& book =
            (taker_side == LiquiditySide::Bid) ? r.ask_levels : r.bid_levels;

        Qty observed_depth = 0;
        Qty true_depth = 0;
        for (const LevelReconstruction& lvl : book) {
            observed_depth += lvl.observed_qty;
            true_depth += lvl.true_qty;
        }

        FillOutcome fo;
        fo.requested_qty = requested_qty;
        // Observer can only "see" displayed depth, so it predicts at most that.
        fo.observer_predicted_fill = min_qty(requested_qty, observed_depth);
        // Reality matches against true (display + hidden) depth.
        fo.realized_fill = min_qty(requested_qty, true_depth);
        fo.latent_remaining = true_depth - fo.realized_fill; // >= 0
        fo.right_censored = fo.latent_remaining > 0;
        return fo;
    }

private:
    static constexpr Qty min_qty(Qty a, Qty b) { return a < b ? a : b; }

    // FNV-1a over the canonicalized reconstruction. Stable across platforms.
    static std::uint64_t hash_reconstruction(const LiquidityReconstruction& r) {
        std::uint64_t h = 1469598103934665603ULL; // FNV offset basis
        auto mix = [&h](std::uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                h ^= (v & 0xFFULL);
                h *= 1099511628211ULL; // FNV prime
                v >>= 8;
            }
        };
        auto mix_levels = [&mix](const std::vector<LevelReconstruction>& lv,
                                 std::uint64_t tag) {
            mix(tag);
            mix(static_cast<std::uint64_t>(lv.size()));
            for (const LevelReconstruction& l : lv) {
                mix(static_cast<std::uint64_t>(l.price));
                mix(static_cast<std::uint64_t>(l.observed_qty));
                mix(static_cast<std::uint64_t>(l.true_qty));
            }
        };
        mix_levels(r.bid_levels, 0xB1D5ULL);
        mix_levels(r.ask_levels, 0xA5C5ULL);
        mix(static_cast<std::uint64_t>(r.observed_total));
        mix(static_cast<std::uint64_t>(r.true_total));
        return h;
    }

    std::vector<OrderEvent> trace_;
};

// --- Compile-time non-feedback / read-only guarantees -----------------------
// These static_asserts are part of the falsifier: they fail to COMPILE if the
// diagnostic ever gains a mutating reconstruct path or stops being a pure
// value-producing projection (which is how a feedback-into-matching coupling
// would manifest).

// reconstruct() must be invocable on a CONST engine (read-only diagnostic).
static_assert(
    std::is_invocable_r_v<LiquidityReconstruction,
                          decltype(&LiquidityAdmissibilityReplay::reconstruct),
                          const LiquidityAdmissibilityReplay&>,
    "reconstruct() must be a read-only (const) diagnostic");

// simulate_fill() must be invocable on a CONST engine.
static_assert(
    std::is_invocable_r_v<FillOutcome,
                          decltype(&LiquidityAdmissibilityReplay::simulate_fill),
                          const LiquidityAdmissibilityReplay&, LiquiditySide, Qty>,
    "simulate_fill() must be a read-only (const) diagnostic");

// The reconstruction is a self-contained, copyable value: it holds no handle to
// any external (matching) state that could be written back.
static_assert(std::is_copy_constructible_v<LiquidityReconstruction>,
              "LiquidityReconstruction must be a pure copyable value type");
static_assert(std::is_copy_constructible_v<FillOutcome>,
              "FillOutcome must be a pure copyable value type");

} // namespace offline
