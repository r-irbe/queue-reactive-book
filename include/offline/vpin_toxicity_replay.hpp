#pragma once

// =============================================================================
// Offline VPIN / FLOW-TOXICITY BUCKET REPLAY fixture.
//
// by supplying the missing OFFLINE diagnostics-on/off falsifier the
// `needs-code-cross-validation` gate demands for the VPIN/flow-toxicity boundary.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. Every
//   quantity built here (a volume bucket, an order imbalance, a bucket toxicity,
//   a windowed VPIN) is a PURE OFFLINE artifact over an already-recorded,
//   replayable signed-trade stream. Data flows ONE WAY, OUT of replay and into
//   derived toxicity diagnostics; a VPIN value may NEVER be fed back as a
//   matching input, a hot-path branch, an order, a fill, a price, a queue
//   that one-way, kernel-free contract.
//
// WHAT IT PROVES (the load-bearing boundary the MOC defends):
//   VPIN (Volume-Synchronized Probability of Informed Trading) is built from
//   EQUAL-VOLUME buckets. The signed-trade stream is packed into consecutive
//   buckets each of exactly `bucket_size` volume (a trade that overflows the
//   current bucket is SPLIT across bucket boundaries). Per bucket the buy and
//   sell volume are accumulated and the order imbalance is
//       imbalance      = buy_vol - sell_vol                 (signed, integer)
//       abs_imbalance  = |buy_vol - sell_vol|               (0 .. bucket_size)
//       toxicity_scaled= abs_imbalance * SCALE / bucket_size (0 .. SCALE)
//   VPIN over the most recent `window` FULL buckets is the volume-weighted mean
//   absolute imbalance:
//       vpin_scaled    = (sum abs_imbalance) * SCALE / (window * bucket_size).
//   These are POST-TRADE analytics: a partial trailing bucket (< bucket_size) is
//   NOT counted, so every bucket is well-defined. The fixture demonstrates:
//     * deterministic, ORDER-SENSITIVE FNV-1a-64 replay hash over a fixed
//       synthetic signed-trade stream + bucket size -- reproducible and
//       tamper-evident;
//     * domain correctness of equal-volume bucketing (including a trade SPLIT
//       across a boundary), the signed imbalance, the bounded bucket toxicity,
//       and windowed VPIN on hand-built integer-tick streams;
//     * the toxicity boundary at TYPE level: a VPIN bucket / result is flagged
//       NOT a matching-kernel input, so it is excluded from any kernel-input set;
//       a positive-control "kernel gate" type IS flagged, proving the
//       discriminator is not vacuous;
//     * read-only / non-feedback: the VPIN computer consumes a CONST trade view
//       and cannot mutate engine/replay state, and no offline type exposes a
//       matching-feedback member (with a positive control proving the trait
//       discriminates).
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   evaluate_vpin() is a PURE function of the (immutable) signed-trade stream and
//   the bucket size: an identical ordered stream + size yields an identical
//   result and an identical hash; reordering the trades changes the buckets and
//   the hash (order-sensitive integrity). The computation takes its inputs by
//   const reference, returns a plain value, and exposes no API that could mutate
//   or be consumed by matching state.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;

// Fixed-point scale for the toxicity / VPIN ratios, so every quantity stays
// integer (no float in the replay path). A toxicity_scaled of SCALE == 1.0.
inline constexpr std::int64_t kVpinScale = 1000000;

// One replayable signed trade. Pure integer data: no pointer, no engine handle,
// no vtable. `seq` is the deterministic event ordinal; `ts_ns` is event-time in
// ns; `price` is the trade price in integer ticks; `volume` is a POSITIVE traded
// quantity; `side` is +1 for a buy-classified trade, -1 for a sell-classified
// trade (the bulk/signed classification is an input, computed upstream offline).
struct SignedTrade {
    std::uint64_t seq = 0;
    std::uint64_t ts_ns = 0;
    std::int64_t price = 0;    // trade price in ticks
    std::int64_t volume = 0;   // positive traded quantity
    std::int8_t side = 0;      // +1 buy-classified, -1 sell-classified
};

// One equal-volume VPIN bucket. Pure value data -- never an order, fill, price,
// queue priority, or matching input. Every full bucket carries exactly
// `bucket_size` total volume by construction.
struct VolumeBucket {
    std::size_t index = 0;            // bucket ordinal (0-based)
    std::int64_t buy_vol = 0;         // buy-classified volume in the bucket
    std::int64_t sell_vol = 0;        // sell-classified volume in the bucket
    std::int64_t total_vol = 0;       // buy_vol + sell_vol (== bucket_size)
    std::int64_t imbalance = 0;       // buy_vol - sell_vol (signed)
    std::int64_t abs_imbalance = 0;   // |buy_vol - sell_vol| (0 .. bucket_size)
    std::int64_t toxicity_scaled = 0; // abs_imbalance * SCALE / bucket_size

    // A VPIN bucket is a POST-TRADE diagnostic, NEVER hidden state that gates
    // FIFO matching. The flag is the type-level kernel-admissibility gate.
    static constexpr bool is_matching_kernel_input = false;
};

// Outcome of one replay: the resolved buckets, summary counts, and the
// order-sensitive replay hash. A trailing partial bucket (< bucket_size) is
// excluded, so `bucketed_volume == n_buckets * bucket_size`.
struct VpinResult {
    std::int64_t bucket_size = 0;
    std::size_t n_trades = 0;
    std::size_t n_buckets = 0;          // number of FULL buckets
    std::int64_t bucketed_volume = 0;   // n_buckets * bucket_size
    std::int64_t residual_volume = 0;   // volume left in the open partial bucket
    std::int64_t sum_abs_imbalance = 0; // sum over full buckets
    std::int64_t vpin_scaled = 0;       // over ALL full buckets, 0 .. SCALE
    std::vector<VolumeBucket> buckets;
    ReplayHash replay_hash = 0;

    // The aggregate result is likewise a post-trade diagnostic, never a kernel
    // input.
    static constexpr bool is_matching_kernel_input = false;
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

// |x| without <cstdlib>, branch-explicit for clarity.
inline constexpr std::int64_t iabs(std::int64_t x) noexcept {
    return x < 0 ? -x : x;
}

}  // namespace detail

// Read-only CONST view of an externally owned signed-trade stream. It exposes
// ONLY const reads of the trades; it has no mutating member, so the VPIN
// computer that consumes it provably cannot alter the replay/engine state it
// reads from. This is the const-correctness half of the no-feedback proof.
class TradeStreamView {
public:
    explicit TradeStreamView(const std::vector<SignedTrade>& stream) noexcept
        : stream_(&stream) {}

    std::size_t size() const noexcept { return stream_->size(); }
    bool empty() const noexcept { return stream_->empty(); }
    const SignedTrade& at(std::size_t i) const noexcept {
        return (*stream_)[i];
    }

private:
    const std::vector<SignedTrade>* stream_;
};

// Finalize the derived fields of a full bucket from its accumulated buy/sell
// volume. PURE: integer arithmetic only, bounded toxicity in [0, SCALE].
inline VolumeBucket finalize_bucket(std::size_t index, std::int64_t buy_vol,
                                    std::int64_t sell_vol,
                                    std::int64_t bucket_size) noexcept {
    VolumeBucket b;
    b.index = index;
    b.buy_vol = buy_vol;
    b.sell_vol = sell_vol;
    b.total_vol = buy_vol + sell_vol;
    b.imbalance = buy_vol - sell_vol;
    b.abs_imbalance = detail::iabs(b.imbalance);
    b.toxicity_scaled =
        bucket_size > 0 ? b.abs_imbalance * kVpinScale / bucket_size : 0;
    return b;
}

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered trades, the
// bucket size, and the derived buckets. Pure, read-only: folds each trade's
// ORDINAL with its bytes so reordering the input changes the hash; folds the
// bucket size and each bucket; folds the counts so truncation/extension cannot
// collide with a prefix.
inline ReplayHash vpin_hash(const std::vector<SignedTrade>& trades,
                            std::int64_t bucket_size,
                            const VpinResult& res) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    h = detail::fnv1a_i64(h, bucket_size);

    std::uint64_t ordinal = 0;
    for (const SignedTrade& t : trades) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, t.seq);
        h = detail::fnv1a_u64(h, t.ts_ns);
        h = detail::fnv1a_i64(h, t.price);
        h = detail::fnv1a_i64(h, t.volume);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(t.side));
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the trade count

    std::uint64_t bucket_ordinal = 0;
    for (const VolumeBucket& b : res.buckets) {
        h = detail::fnv1a_u64(h, bucket_ordinal);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(b.index));
        h = detail::fnv1a_i64(h, b.buy_vol);
        h = detail::fnv1a_i64(h, b.sell_vol);
        h = detail::fnv1a_i64(h, b.imbalance);
        h = detail::fnv1a_i64(h, b.toxicity_scaled);
        ++bucket_ordinal;
    }
    h = detail::fnv1a_u64(h, bucket_ordinal);  // fold the bucket count

    h = detail::fnv1a_i64(h, res.sum_abs_imbalance);
    h = detail::fnv1a_i64(h, res.bucketed_volume);
    h = detail::fnv1a_i64(h, res.vpin_scaled);
    return h;
}

// Evaluate a replayable signed-trade stream into equal-volume VPIN buckets.
// PURE, READ-ONLY, OFFLINE: the trades are taken by const reference and never
// modified; the result is a plain value. Each trade's volume is packed into the
// current bucket; when the bucket reaches `bucket_size` it is closed and a new
// one begins, SPLITTING a trade that overflows the boundary. A trailing partial
// bucket (< bucket_size) is reported as `residual_volume` and excluded from the
// VPIN average, so every counted bucket holds exactly `bucket_size` volume.
inline VpinResult evaluate_vpin(const std::vector<SignedTrade>& trades,
                                std::int64_t bucket_size) {
    VpinResult res;
    res.bucket_size = bucket_size;
    res.n_trades = trades.size();
    if (bucket_size <= 0) {
        res.replay_hash = vpin_hash(trades, bucket_size, res);
        return res;
    }

    const TradeStreamView view(trades);  // const, read-only trade view
    std::int64_t cur_buy = 0;
    std::int64_t cur_sell = 0;
    std::int64_t cur_total = 0;
    std::size_t bucket_index = 0;

    for (std::size_t i = 0; i < view.size(); ++i) {
        const SignedTrade& t = view.at(i);
        std::int64_t remaining = t.volume > 0 ? t.volume : 0;
        while (remaining > 0) {
            const std::int64_t room = bucket_size - cur_total;
            const std::int64_t take = remaining < room ? remaining : room;
            if (t.side >= 0) {
                cur_buy += take;
            } else {
                cur_sell += take;
            }
            cur_total += take;
            remaining -= take;

            if (cur_total == bucket_size) {
                VolumeBucket b =
                    finalize_bucket(bucket_index, cur_buy, cur_sell, bucket_size);
                res.sum_abs_imbalance += b.abs_imbalance;
                res.buckets.push_back(b);
                ++bucket_index;
                cur_buy = 0;
                cur_sell = 0;
                cur_total = 0;
            }
        }
    }

    res.n_buckets = res.buckets.size();
    res.bucketed_volume =
        static_cast<std::int64_t>(res.n_buckets) * bucket_size;
    res.residual_volume = cur_total;  // open partial bucket, excluded
    if (res.bucketed_volume > 0) {
        res.vpin_scaled =
            res.sum_abs_imbalance * kVpinScale / res.bucketed_volume;
    }
    res.replay_hash = vpin_hash(trades, bucket_size, res);
    return res;
}

// Windowed VPIN over the most recent `window` FULL buckets, as a fixed-point
// fraction in [0, SCALE]. PURE, READ-ONLY: reads the resolved buckets through a
// const reference. A window larger than the bucket count falls back to all
// buckets. Returns 0 when there are no buckets.
inline std::int64_t vpin_scaled_over_window(const VpinResult& res,
                                            std::size_t window) noexcept {
    if (res.n_buckets == 0 || res.bucket_size <= 0) return 0;
    const std::size_t w = window < res.n_buckets ? window : res.n_buckets;
    if (w == 0) return 0;
    std::int64_t sum_abs = 0;
    for (std::size_t k = res.n_buckets - w; k < res.n_buckets; ++k) {
        sum_abs += res.buckets[k].abs_imbalance;
    }
    const std::int64_t denom = static_cast<std::int64_t>(w) * res.bucket_size;
    return denom > 0 ? sum_abs * kVpinScale / denom : 0;
}

// The boundary gate: a VPIN value is a post-trade toxicity DIAGNOSTIC, so using
// it as a matching-kernel input (to gate a fill, match, queue priority, or
// cancel) would breach the offline boundary. This returns true whenever a type
// is flagged inadmissible as a kernel input (always, for VPIN diagnostics), so a
// caller that tries to wire it into matching is flagged.
inline constexpr bool is_toxicity_diagnostic(const VolumeBucket&) noexcept {
    return !VolumeBucket::is_matching_kernel_input;
}
inline constexpr bool is_toxicity_diagnostic(const VpinResult&) noexcept {
    return !VpinResult::is_matching_kernel_input;
}

// Read-only convenience wrapper. Holds a const view of an externally owned trade
// stream and exposes ONLY the evaluation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The diagnostic it returns can be read but never
// written back into matching.
class VpinToxicityReplay {
public:
    explicit VpinToxicityReplay(const std::vector<SignedTrade>& trades) noexcept
        : trades_(&trades) {}

    VpinResult evaluate(std::int64_t bucket_size) const {
        return evaluate_vpin(*trades_, bucket_size);
    }
    ReplayHash hash(std::int64_t bucket_size) const {
        return evaluate_vpin(*trades_, bucket_size).replay_hash;
    }
    std::size_t size() const noexcept { return trades_->size(); }
    bool empty() const noexcept { return trades_->empty(); }

private:
    const std::vector<SignedTrade>* trades_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member (a
// method named `apply_to_matching`). The offline VPIN types do NOT, which is the
// type-level proof that a toxicity value cannot be wired back into the kernel
// through these structs. A test supplies a POSITIVE CONTROL (a type that DOES
// expose such a member) to prove the trait actually discriminates.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// --- Compile-time matching-kernel-admissibility discriminator -----------------
//
// Reads the `is_matching_kernel_input` flag a type publishes. A VPIN bucket /
// result publishes `false` (post-trade diagnostic), a positive-control kernel
// gate publishes `true`. `is_matching_kernel_input_v<T>` is true ONLY for types
// that claim to be matching inputs; it is used to EXCLUDE the VPIN diagnostics
// from any matching-input set at compile time.
template <typename T, typename = void>
struct is_matching_kernel_input : std::false_type {};

template <typename T>
struct is_matching_kernel_input<
    T, std::void_t<decltype(T::is_matching_kernel_input)>>
    : std::bool_constant<T::is_matching_kernel_input> {};

template <typename T>
inline constexpr bool is_matching_kernel_input_v =
    is_matching_kernel_input<T>::value;

// Static proof, evaluated at header-include time, of the toxicity-as-diagnostic
// and no-coupling boundary:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * the VPIN bucket / result is NOT a matching-kernel input, while a positive
//     control IS => VPIN is excluded from the kernel-input set;
//   * evaluate() is invocable on a CONST wrapper, and the VPIN computer consumes
//     a CONST trade view => the diagnostic is read-only.
static_assert(std::is_trivially_copyable_v<SignedTrade>,
              "SignedTrade must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<VolumeBucket>,
              "VolumeBucket must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<SignedTrade>,
              "SignedTrade must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<VolumeBucket>,
              "VolumeBucket must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<SignedTrade>,
              "SignedTrade must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<VolumeBucket>,
              "VolumeBucket must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<VpinResult>,
              "VpinResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<VpinToxicityReplay>,
              "VpinToxicityReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<TradeStreamView>,
              "TradeStreamView must not expose a matching-feedback member (one-way data flow)");
// The load-bearing boundary gate: the VPIN bucket / result are EXCLUDED from the
// matching-kernel-input set, proving (with a positive control in the test) the
// discriminator is not vacuously false.
static_assert(!is_matching_kernel_input_v<VolumeBucket>,
              "a VPIN bucket is a post-trade diagnostic, never a matching-kernel input");
static_assert(!is_matching_kernel_input_v<VpinResult>,
              "a VPIN result is a post-trade diagnostic, never a matching-kernel input");
static_assert(
    std::is_invocable_r_v<VpinResult, decltype(&VpinToxicityReplay::evaluate),
                          const VpinToxicityReplay&, std::int64_t>,
    "evaluate() must be a read-only (const) diagnostic");

}  // namespace offline
