#pragma once

// =============================================================================
// Offline PnL-ATTRIBUTION / ACCOUNTING decomposition replay fixture.
//
// by supplying the missing deterministic, replay-neutral fixture in which an
// ORDERED stream of fills is decomposed into the MOC's separately-reconciled
// accounting fields -- realized cash, remaining position, terminal-mark
// unrealized, fee/rebate sign, and carry/financing -- as PURE OFFLINE labels,
// never touching matching.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and price/size/time only*; every quantity
//   computed here (mark-to-market net PnL, FIFO realized/unrealized split, a
//   fee/rebate ledger, a carry charge, a peak absolute inventory) is a PURE
//   OFFLINE accounting LABEL over an already-fixed sequence of fills. Data flows
//   ONE WAY, OUT of the fills and into derived accounting numbers; a computed
//   value may NEVER be fed back as a matching input, a fill/priority/tie-break
//   that one-way, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-accounting-schema` requires that realized cash, remaining position,
//   fee/rebate, cost basis, terminal mark source, and carry/financing reconcile
//   as SEPARATE fields, with golden rows covering flat, long, short,
//   partial-fill, and terminal-position cases. Its open `needs-code-validation`
//   channel asked for a worked, replay-neutral example of exactly that
//   decomposition. Given an ORDERED sequence of fills (signed quantity, integer
//   tick price, a non-negative taker fee, a non-negative maker rebate, a
//   non-negative carry/financing charge) plus an explicit terminal mark, it:
//     * computes the EXACT integer mark-to-market accounting identity
//         gross_mtm = accumulated_cash + final_position * mark
//       where accumulated_cash = sum(-qty*price). This total, the net fee ledger
//       (rebates - fees), and the carry total are ORDER-INVARIANT, so the net
//       PnL gross_mtm - fees + rebates - carry is ORDER-INVARIANT;
//     * computes the FIFO realized/unrealized SPLIT (close oldest lot first;
//       realized on a closing trade is sign(lot)*(fill_price-lot_price)*matched;
//       unrealized marks the remaining open lots to `mark`). The split is
//       ORDER-SENSITIVE -- a reordering of the SAME multiset of fills changes
//       realized and unrealized -- yet realized + unrealized ALWAYS reconciles
//       to the order-invariant gross_mtm (the accounting identity);
//     * tracks the peak absolute running inventory along the fill order -- the
//       order-SENSITIVE inventory-exposure path metric (same multiset, different
//       order => different peak, same final position);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED fills plus the derived accounting numbers, so the diagnostic is
//       reproducible and order-sensitive.
//   Closed-form OFFLINE accounting relations are provided as directly-testable
//   `analytic` helpers faithful to the MOC's routed fixture rows:
//     * a round-trip realized-PnL identity (exit-entry)*qty;
//     * the fee/rebate sign convention net_fee = rebate - fee (maker rebate
//       credits, taker fee debits);
//     * an explicit integer carry/financing charge rate*abs_position*periods
//       (no hidden zero-carry default);
//     * the net-after-costs fold gross - fees + rebates - carry.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered fills and mark: an
//   identical ordered input yields an identical result and hash; reordering the
//   SAME multiset of fills leaves the mark-to-market net PnL, the fee/rebate
//   ledger, the carry total, and the final position unchanged, yet changes the
//   FIFO realized/unrealized split, the peak absolute inventory, and the hash
//   (order-sensitive integrity). The computation takes the fills by const
//   reference, returns a plain value, and exposes no API that could mutate or be
//   consumed by matching state.
//
// Accounting background (kept self-contained; textbook relations re-stated here
// for the fixture, no value imported from any rendered source):
//   * The mark-to-market PnL of a position is the accumulated trade cash plus the
//     terminal-mark value of the remaining position; it is invariant to the order
//     in which the fills arrive.
//   * Any lot-matching convention (FIFO here) only PARTITIONS that total into a
//     realized part (closed lots) and an unrealized part (open lots marked to
//     terminal); the partition is convention/order-dependent but always sums to
//     the same mark-to-market total.
//   * Fees and rebates are post-trade accounting inputs with opposite sign: a
//     taker fee debits PnL, a maker rebate credits it.
//   * Inventory-bearing experiments carry an explicit financing/carry charge; a
//     missing charge is not silently treated as zero.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price / cash unit

// One replayable fill in the accounting stream. Pure integer data: no pointer,
// `quantity` is the trade direction (+ buy, - sell). `price` is an integer tick.
// `fee_paid` (taker debit) and `rebate_recv` (maker credit) are NON-NEGATIVE
// cash magnitudes carried separately so the fee/rebate sign convention is
// explicit; `carry_charge` is the NON-NEGATIVE explicit financing/carry debit
// for this step (never a hidden zero-carry default).
struct Fill {
    std::uint64_t seq = 0;
    std::int64_t quantity = 0;      // signed contracts (buy +, sell -)
    std::int64_t price = 0;         // integer tick price
    std::int64_t fee_paid = 0;      // >= 0 taker fee (cash debit)
    std::int64_t rebate_recv = 0;   // >= 0 maker rebate (cash credit)
    std::int64_t carry_charge = 0;  // >= 0 financing/carry debit for this step
};

// One open FIFO lot remaining after the stream is replayed: a signed residual
// quantity (sign = position direction) bought/sold at `price`. Pure value type;
// the vector of these is the cost-basis residue marked to terminal.
struct OpenLot {
    std::int64_t quantity = 0;  // signed residual quantity
    std::int64_t price = 0;     // integer tick cost basis
};

// Derived accounting decomposition plus the order-sensitive replay hash. Pure
// value type: copyable, holds no reference to any external (matching) state.
struct PnlAttribution {
    std::size_t n = 0;
    std::int64_t terminal_mark = 0;  // echoed terminal mark source/value

    // (1) ORDER-INVARIANT mark-to-market accounting identity.
    std::int64_t final_position = 0;       // sum of signed quantities
    std::int64_t accumulated_cash = 0;     // sum of -qty*price
    std::int64_t position_mark_value = 0;  // final_position * mark
    std::int64_t gross_mtm_pnl = 0;        // accumulated_cash + position_mark_value

    // (2) ORDER-INVARIANT fee / rebate / carry ledger and net PnL.
    std::int64_t total_fees = 0;      // sum fee_paid
    std::int64_t total_rebates = 0;   // sum rebate_recv
    std::int64_t total_carry = 0;     // sum carry_charge
    std::int64_t net_fee_pnl = 0;     // rebates - fees
    std::int64_t net_pnl = 0;         // gross_mtm - fees + rebates - carry

    // (3) ORDER-SENSITIVE FIFO realized / unrealized split (reconciles to (1)).
    std::int64_t realized_pnl = 0;    // closed-lot PnL (FIFO)
    std::int64_t unrealized_pnl = 0;  // open lots marked to terminal
    std::vector<OpenLot> open_lots;   // cost-basis residue, in FIFO order
    std::size_t open_lot_count = 0;

    // (4) ORDER-SENSITIVE inventory-exposure path metric.
    std::int64_t peak_abs_position = 0;

    ReplayHash replay_hash = 0;
};

namespace detail {

inline constexpr ReplayHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr ReplayHash kFnvPrime = 1099511628211ULL;

inline constexpr ReplayHash fnv1a_byte(ReplayHash h, std::uint8_t b) noexcept {
    h ^= static_cast<ReplayHash>(b);
    h *= kFnvPrime;
    return h;
}

// Fold a fixed-width value in canonical little-endian order, so the hash does
// not depend on host endianness.
inline constexpr ReplayHash fnv1a_u64(ReplayHash h, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = fnv1a_byte(h, static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
    return h;
}

inline constexpr ReplayHash fnv1a_i64(ReplayHash h, std::int64_t v) noexcept {
    return fnv1a_u64(h, static_cast<std::uint64_t>(v));
}

// Branch-free integer absolute value over int64 (no overflow on the inputs this
// fixture is exercised with).
inline std::int64_t iabs(std::int64_t v) noexcept { return v < 0 ? -v : v; }

// Sign of an int64 as {-1, 0, +1}.
inline std::int64_t isign(std::int64_t v) noexcept {
    return (v > 0) - (v < 0);
}

}  // namespace detail

// Analytic reference relations for the offline accounting layer, used to
// validate the decomposition on closed-form cases. Textbook relations, computed
// here so the fixture is self-checking and self-contained.
namespace analytic {

// Round-trip realized PnL of a fully closed position: closing `qty` units bought
// at `entry` and sold at `exit` realizes (exit - entry) * qty (exact integer).
inline std::int64_t round_trip_pnl(std::int64_t entry, std::int64_t exit,
                                   std::int64_t qty) noexcept {
    return (exit - entry) * qty;
}

// Fee/rebate sign convention: a maker rebate credits PnL, a taker fee debits it.
inline std::int64_t net_fee(std::int64_t rebate, std::int64_t fee) noexcept {
    return rebate - fee;
}

// Explicit integer carry/financing charge: rate per period * absolute position
// held * number of periods. No hidden zero-carry default; the caller must supply
// the rate and holding horizon.
inline std::int64_t carry_cost(std::int64_t rate_per_period,
                               std::int64_t abs_position,
                               std::int64_t periods) noexcept {
    return rate_per_period * detail::iabs(abs_position) * periods;
}

// Net PnL after costs: gross mark-to-market minus fees, plus rebates, minus
// carry. The single fold that ties the separate accounting fields together.
inline std::int64_t net_after_costs(std::int64_t gross, std::int64_t fees,
                                    std::int64_t rebates,
                                    std::int64_t carry) noexcept {
    return gross - fees + rebates - carry;
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered fills, the
// terminal mark, and the derived accounting numbers. Pure, read-only: folds each
// fill's ORDINAL with its bytes so reordering the input changes the hash; folds
// the order-invariant totals, the order-sensitive realized/unrealized split and
// peak inventory, and the open-lot residue; folds the fill count so a truncation
// cannot collide with a prefix.
inline ReplayHash pnl_attribution_hash(const std::vector<Fill>& fills,
                                       const PnlAttribution& acct) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const Fill& f : fills) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, f.seq);
        h = detail::fnv1a_i64(h, f.quantity);
        h = detail::fnv1a_i64(h, f.price);
        h = detail::fnv1a_i64(h, f.fee_paid);
        h = detail::fnv1a_i64(h, f.rebate_recv);
        h = detail::fnv1a_i64(h, f.carry_charge);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the fill count

    h = detail::fnv1a_i64(h, acct.terminal_mark);
    h = detail::fnv1a_i64(h, acct.final_position);
    h = detail::fnv1a_i64(h, acct.accumulated_cash);
    h = detail::fnv1a_i64(h, acct.gross_mtm_pnl);
    h = detail::fnv1a_i64(h, acct.total_fees);
    h = detail::fnv1a_i64(h, acct.total_rebates);
    h = detail::fnv1a_i64(h, acct.total_carry);
    h = detail::fnv1a_i64(h, acct.net_pnl);
    h = detail::fnv1a_i64(h, acct.realized_pnl);
    h = detail::fnv1a_i64(h, acct.unrealized_pnl);
    h = detail::fnv1a_i64(h, acct.peak_abs_position);

    std::uint64_t lot_ordinal = 0;
    for (const OpenLot& lot : acct.open_lots) {
        h = detail::fnv1a_u64(h, lot_ordinal);
        h = detail::fnv1a_i64(h, lot.quantity);
        h = detail::fnv1a_i64(h, lot.price);
        ++lot_ordinal;
    }
    h = detail::fnv1a_u64(h, lot_ordinal);  // fold the open-lot count
    return h;
}

// Compute the offline PnL attribution for an ordered stream of fills marked to
// `terminal_mark`. PURE, READ-ONLY, OFFLINE: the fills are taken by const
// reference and never modified; the result is a plain value. The work is a
// single ordered pass with FIFO lot matching (each fill closes oldest lots
// first, then any residual opens a new lot), so the computation is deterministic
// and never iterates a data-dependent number of times beyond the lot residue.
// This cannot mutate, and is not consumed by, any matching state.
inline PnlAttribution compute_pnl_attribution(const std::vector<Fill>& fills,
                                              std::int64_t terminal_mark) {
    PnlAttribution acct;
    acct.n = fills.size();
    acct.terminal_mark = terminal_mark;

    std::vector<OpenLot> lots;  // open FIFO lots, all share the position sign
    std::int64_t running_position = 0;

    for (const Fill& f : fills) {
        // (1) ORDER-INVARIANT cash/position identity and (2) ledger.
        acct.final_position += f.quantity;
        acct.accumulated_cash += -f.quantity * f.price;
        acct.total_fees += f.fee_paid;
        acct.total_rebates += f.rebate_recv;
        acct.total_carry += f.carry_charge;

        // (3) FIFO realized split: close oldest opposing lots first.
        std::int64_t remaining = f.quantity;
        while (remaining != 0 && !lots.empty() &&
               detail::isign(lots.front().quantity) == -detail::isign(remaining)) {
            OpenLot& front = lots.front();
            const std::int64_t lot_sign = detail::isign(front.quantity);
            const std::int64_t matched =
                std::min(detail::iabs(front.quantity), detail::iabs(remaining));
            // Closing a long lot realizes (price - cost); a short lot the inverse.
            acct.realized_pnl += lot_sign * (f.price - front.price) * matched;
            front.quantity -= lot_sign * matched;   // shrink lot toward zero
            remaining += lot_sign * matched;         // shrink remaining toward zero
            if (front.quantity == 0) {
                lots.erase(lots.begin());
            }
        }
        if (remaining != 0) {
            lots.push_back(OpenLot{remaining, f.price});
        }

        // (4) ORDER-SENSITIVE inventory-exposure path metric.
        running_position += f.quantity;
        const std::int64_t mag = detail::iabs(running_position);
        if (mag > acct.peak_abs_position) {
            acct.peak_abs_position = mag;
        }
    }

    // Mark-to-market identity and unrealized from the open-lot residue.
    acct.position_mark_value = acct.final_position * terminal_mark;
    acct.gross_mtm_pnl = acct.accumulated_cash + acct.position_mark_value;
    for (const OpenLot& lot : lots) {
        const std::int64_t lot_sign = detail::isign(lot.quantity);
        acct.unrealized_pnl +=
            lot_sign * (terminal_mark - lot.price) * detail::iabs(lot.quantity);
    }

    acct.open_lots = lots;
    acct.open_lot_count = lots.size();

    acct.net_fee_pnl = acct.total_rebates - acct.total_fees;
    acct.net_pnl = acct.gross_mtm_pnl - acct.total_fees + acct.total_rebates -
                   acct.total_carry;

    acct.replay_hash = pnl_attribution_hash(fills, acct);
    return acct;
}

// fills plus the terminal mark, and exposes ONLY the computation; it has no
// mutating member and no matching-engine coupling, making the offline /
// one-way-data-flow boundary explicit at the call site. The accounting numbers it
// returns can be read but never written back into matching.
class PnlAttributionReplay {
public:
    PnlAttributionReplay(const std::vector<Fill>& fills,
                         std::int64_t terminal_mark) noexcept
        : fills_(&fills), terminal_mark_(terminal_mark) {}

    PnlAttribution compute() const {
        return compute_pnl_attribution(*fills_, terminal_mark_);
    }
    ReplayHash hash() const { return compute().replay_hash; }
    std::size_t size() const noexcept { return fills_->size(); }
    bool empty() const noexcept { return fills_->empty(); }

private:
    const std::vector<Fill>* fills_;
    std::int64_t terminal_mark_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline accounting types do NOT,
// which is the type-level proof that a computed PnL value cannot be wired back
// into the kernel through these structs. A test supplies a POSITIVE CONTROL (a
// type that DOES expose such a member) to prove the trait actually discriminates
// and is not vacuously false.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// Static proof, evaluated at header-include time, that the offline types are pure
// data with no matching-engine coupling:
//   * trivially copyable + standard layout (for the fixed-size fill/lot types)
//     => no engine handle, pointer-back, or vtable reaching matching;
//   * no `apply_to_matching` feedback member => accounting numbers are one-way
//     OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<Fill>,
              "Fill must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<OpenLot>,
              "OpenLot must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<Fill>,
              "Fill must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<OpenLot>,
              "OpenLot must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<Fill>,
              "Fill must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<PnlAttribution>,
              "PnlAttribution must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<PnlAttributionReplay>,
              "PnlAttributionReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<PnlAttribution,
                          decltype(&PnlAttributionReplay::compute),
                          const PnlAttributionReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
