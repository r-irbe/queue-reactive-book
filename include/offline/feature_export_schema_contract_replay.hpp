#pragma once

// =============================================================================
// Offline FEATURE-EXPORT SCHEMA/HORIZON-CONTRACT REPLAY fixture.
//
// by supplying the missing "OFFLINE schema-round-trip + observer-on/off
// replay-hash-equality falsifier fixture" that channel demands.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and event-driven*; every artifact built
//   here (a self-describing export header, an exported feature/label row, a
//   point-in-time availability flag, a serialized byte image) is a PURE OFFLINE
//   product over an already-recorded, replayable event stream. Data flows ONE
//   WAY, OUT of replay and into a downstream model/calibration/TCA reader; an
//   exported row may NEVER be fed back as a matching input, a hot-path branch,
//   an order, a fill, a price, or accounting truth. This header is the
//   deterministic proof of that one-way, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-feature-export-schema-contracts` carries the load-bearing rule: an
//   exported observability artifact is reusable EVIDENCE only when it is
//   SELF-DESCRIBING -- it declares schema version, timestamp basis, horizon,
//   source/provenance hash, and label availability / censoring -- AND the act
//   of observing or exporting must NOT change the replay (contract) hash. This
//   fixture demonstrates all four:
//     * SELF-DESCRIBING EXPORT -- export() emits a header carrying
//       the input stream, so the artifact identifies its own contract;
//     * SCHEMA ROUND-TRIP -- serialize()/parse() round-trips the export to a
//       byte image and back to an identical export + identical contract hash;
//       parsing under a MISMATCHED expected schema version is DETECTED (the
//       schema-drift falsifier);
//     * OBSERVER NON-PERTURBATION -- attaching a read-only observer
//       (observer.enabled) does NOT change the contract hash; a PERTURBING
//       observer (observer.perturbs, a positive control) that writes its running
//       state back into a row DOES change the hash -- proving observation must
//       be side-effect-free and that the equality check is non-vacuous;
//     * POINT-IN-TIME AVAILABILITY / CENSORING -- a forward label at horizon h
//       is `available` iff its future event t+h is present in the recorded
//       suffix; otherwise it is CENSORED (sentinel value, available=false). A
//       label value/availability flips from censored to available IFF a future
//       event at t+h is appended to the input -- the no-future-leakage proof
//       (a value changes only when the future it depends on actually exists).
//     * It emits a deterministic, ORDER-SENSITIVE FNV-1a-64 contract hash over
//       the schema, the provenance, and the exported rows, so the artifact is
//       reproducible and tamper-evident.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   export() is a PURE function of the (immutable) event stream, the schema, and
//   the observer config: an identical ordered stream + schema yields an identical
//   export and an identical contract hash; reordering events changes the rows and
//   the hash (order-sensitive integrity); a NON-perturbing observer (on or off)
//   yields the SAME contract hash; serialize()->parse() is the identity on the
//   export. The computation takes the events by const reference, returns a plain
//   value, and exposes no API that could mutate or be consumed by matching state.
//
// Export-contract background (kept self-contained; nothing imported from a
// rendered source): a feature/label tensor is reusable downstream only if a
// reader can reconstruct its MEANING without out-of-band knowledge -- hence the
// self-describing header. A label measured over a forward horizon h is only
// defined once t+h has occurred; reporting a value before then would be
// future-fabrication, so the point-in-time rule censors it. Observation that
// changes the artifact (an "observer effect") destroys replay-equivalence, so
// the contract hash must be invariant to read-only observation.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;

// Sentinel label for a point-in-time CENSORED row (its forward horizon t+h is
// not yet present in the recorded stream, so the label is undefined).
inline constexpr std::int64_t kCensoredLabel = 0;

// The SELF-DESCRIBING export header. Every exported artifact carries it so a
// downstream reader can reconstruct the contract without out-of-band knowledge.
// Pure integer data: no pointer, no engine handle, no vtable.
struct ExportSchema {
    std::uint32_t schema_version = 1;  // contract/format version
    std::uint32_t ts_basis = 0;        // timestamp-basis id (e.g. 0 = event-ns)
    std::int32_t horizon = 1;          // forward label horizon h (>=1)
    std::int32_t lookback = 1;         // trailing feature window length (>=1)
};

// One replayable event-time observation. Pure integer data.
struct FeatureEvent {
    std::uint64_t seq = 0;
    std::uint64_t ts_ns = 0;
    std::int64_t price = 0;  // ticks
};

// One EXPORTED feature/label row. Self-describing via the export header; never
// an order, fill, or price input.
struct FeatureRow {
    std::uint64_t event_seq = 0;    // seq of the decision event t
    std::uint64_t ts_ns = 0;        // event-time of t
    std::int64_t feature = 0;       // causal trailing change ending at t
    std::int64_t label = 0;         // forward markout, or kCensoredLabel if censored
    bool label_available = true;    // false => point-in-time CENSORED (t+h absent)
};

// Read-only observer config. `enabled` attaches a diagnostic that counts rows as
// they are emitted (a side statistic that must NOT enter the contract hash).
// `perturbs` is the POSITIVE CONTROL: a (wrong) observer that writes its running
// state back into the exported row, which MUST change the contract hash.
struct ObserverConfig {
    bool enabled = false;
    bool perturbs = false;
};

// Outcome of one export: the self-describing schema, the exported rows, the
// availability/censoring tally, the provenance hash of the input, the read-only
// observer's row count, and the order-sensitive contract hash.
struct FeatureExport {
    ExportSchema schema;
    std::size_t n_events = 0;
    std::size_t n_rows = 0;
    std::size_t n_available = 0;   // rows whose forward label is defined
    std::size_t n_censored = 0;    // rows whose forward horizon is not yet present
    std::vector<FeatureRow> rows;
    ReplayHash provenance_hash = 0;  // hash over the ordered input events
    ReplayHash contract_hash = 0;    // hash over schema + provenance + rows
    std::size_t observer_count = 0;  // read-only observer's row tally (NOT hashed)

    // The contract gate: an export is admissible only if its declared schema is
    // well-formed (positive horizon and lookback).
    bool is_admissible() const noexcept {
        return schema.horizon >= 1 && schema.lookback >= 1;
    }
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

}  // namespace detail

// Deterministic, order-sensitive FNV-1a-64 provenance hash over the ordered
// input events. Pure, read-only: folds each event's ORDINAL with its bytes so
// reordering the input changes the hash, and folds the count so truncation
// cannot collide with a prefix.
inline ReplayHash feature_provenance_hash(
    const std::vector<FeatureEvent>& events) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    std::uint64_t ordinal = 0;
    for (const FeatureEvent& e : events) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, e.seq);
        h = detail::fnv1a_u64(h, e.ts_ns);
        h = detail::fnv1a_i64(h, e.price);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the event count
    return h;
}

// Deterministic, order-sensitive FNV-1a-64 contract hash over the SELF-DESCRIBING
// schema, the provenance hash, and the exported rows (including the availability
// flag). The read-only observer count is deliberately NOT folded: a read-only
// observation must leave the contract hash invariant.
inline ReplayHash feature_contract_hash(const ExportSchema& schema,
                                        ReplayHash provenance,
                                        const std::vector<FeatureRow>& rows) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(schema.schema_version));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(schema.ts_basis));
    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(schema.horizon));
    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(schema.lookback));
    h = detail::fnv1a_u64(h, provenance);

    std::uint64_t row_ordinal = 0;
    for (const FeatureRow& r : rows) {
        h = detail::fnv1a_u64(h, row_ordinal);
        h = detail::fnv1a_u64(h, r.event_seq);
        h = detail::fnv1a_u64(h, r.ts_ns);
        h = detail::fnv1a_i64(h, r.feature);
        h = detail::fnv1a_i64(h, r.label);
        h = detail::fnv1a_byte(h, r.label_available ? 1u : 0u);
        ++row_ordinal;
    }
    h = detail::fnv1a_u64(h, row_ordinal);  // fold the row count
    return h;
}

// Export a replayable event stream into a SELF-DESCRIBING feature/label artifact
// under the point-in-time availability rule. PURE, READ-ONLY, OFFLINE: the events
// are taken by const reference and never modified; the result is a plain value.
// The amount of work is FIXED per event (a constant-size trailing window and a
// single forward lookup), so the export is deterministic and never iterates a
// data-dependent number of times. This cannot mutate, and is not consumed by,
// any matching state.
//
// For each decision event at index t with a full trailing feature window
// (t >= lookback), the row carries:
//   * feature -- the causal trailing change price[t] - price[t - lookback];
//   * label   -- the forward markout price[t + h] - price[t] IFF the future
//     event t+h is present in the recorded suffix; otherwise the row is CENSORED
//     (label = kCensoredLabel, label_available = false). Censoring is the
//     no-future-fabrication rule: an undefined forward label is never invented.
inline FeatureExport export_features(const std::vector<FeatureEvent>& events,
                                     const ExportSchema& schema,
                                     const ObserverConfig& observer = {}) {
    FeatureExport out;
    out.schema = schema;
    out.n_events = events.size();

    const int h = schema.horizon >= 1 ? schema.horizon : 1;
    const int lb = schema.lookback >= 1 ? schema.lookback : 1;
    const std::size_t n = events.size();

    std::uint64_t observed = 0;
    for (std::size_t t = 0; t < n; ++t) {
        if (t < static_cast<std::size_t>(lb)) continue;  // need trailing window

        FeatureRow row;
        row.event_seq = events[t].seq;
        row.ts_ns = events[t].ts_ns;
        row.feature = events[t].price - events[t - static_cast<std::size_t>(lb)].price;

        const std::size_t fwd = t + static_cast<std::size_t>(h);
        if (fwd < n) {
            row.label = events[fwd].price - events[t].price;
            row.label_available = true;
            ++out.n_available;
        } else {
            // Point-in-time CENSORING: the forward horizon is not yet in the
            // recorded stream, so the label is undefined -- never fabricated.
            row.label = kCensoredLabel;
            row.label_available = false;
            ++out.n_censored;
        }

        // POSITIVE CONTROL: a perturbing observer writes its running state back
        // into the exported row, which must (and does) change the contract hash.
        if (observer.enabled && observer.perturbs) {
            row.feature += static_cast<std::int64_t>(observed);
        }

        out.rows.push_back(row);
        ++observed;
    }

    // Read-only observation: count rows as a side statistic that is NOT folded
    // into the contract hash.
    out.observer_count = observer.enabled ? observed : 0;

    out.n_rows = out.rows.size();
    out.provenance_hash = feature_provenance_hash(events);
    out.contract_hash =
        feature_contract_hash(out.schema, out.provenance_hash, out.rows);
    return out;
}

// --- Schema round-trip (serialize / parse) ----------------------------------
//
// A compact, deterministic byte image of an export. The round-trip proves the
// schema CONTRACT preserves replay meaning: parse(serialize(x)) reproduces x and
// its contract hash exactly. Parsing under a mismatched expected schema version
// is detected (the schema-drift falsifier).

namespace detail {

inline void put_u64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
}

inline std::uint64_t get_u64(const std::vector<std::uint8_t>& buf,
                             std::size_t& pos) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(buf[pos + static_cast<std::size_t>(i)])
             << (8 * i);
    }
    pos += 8;
    return v;
}

}  // namespace detail

// Serialize an export to a deterministic little-endian byte image.
inline std::vector<std::uint8_t> serialize_export(const FeatureExport& ex) {
    std::vector<std::uint8_t> buf;
    detail::put_u64(buf, ex.schema.schema_version);
    detail::put_u64(buf, ex.schema.ts_basis);
    detail::put_u64(buf, static_cast<std::uint64_t>(
                             static_cast<std::int64_t>(ex.schema.horizon)));
    detail::put_u64(buf, static_cast<std::uint64_t>(
                             static_cast<std::int64_t>(ex.schema.lookback)));
    detail::put_u64(buf, ex.provenance_hash);
    detail::put_u64(buf, static_cast<std::uint64_t>(ex.rows.size()));
    for (const FeatureRow& r : ex.rows) {
        detail::put_u64(buf, r.event_seq);
        detail::put_u64(buf, r.ts_ns);
        detail::put_u64(buf, static_cast<std::uint64_t>(r.feature));
        detail::put_u64(buf, static_cast<std::uint64_t>(r.label));
        detail::put_u64(buf, r.label_available ? 1u : 0u);
    }
    detail::put_u64(buf, ex.contract_hash);
    return buf;
}

// Outcome of a parse: whether the bytes were well-formed, whether the declared
// schema version matched the reader's expectation, and the reconstructed export.
struct ParseResult {
    bool ok = false;             // bytes were structurally well-formed
    bool schema_matched = false; // declared schema_version == expected_version
    FeatureExport value;
};

// Parse a byte image back into an export and check it against an expected schema
// version (the schema-drift falsifier). The reconstructed export's contract hash
// is recomputed and compared to the stored one so a corrupted image is rejected.
inline ParseResult parse_export(const std::vector<std::uint8_t>& buf,
                                std::uint32_t expected_version) {
    ParseResult pr;
    // Minimum image is the fixed header (6 u64) + trailing contract hash (1 u64).
    if (buf.size() < 8 * 7) return pr;

    std::size_t pos = 0;
    FeatureExport ex;
    ex.schema.schema_version = static_cast<std::uint32_t>(detail::get_u64(buf, pos));
    ex.schema.ts_basis = static_cast<std::uint32_t>(detail::get_u64(buf, pos));
    ex.schema.horizon =
        static_cast<std::int32_t>(static_cast<std::int64_t>(detail::get_u64(buf, pos)));
    ex.schema.lookback =
        static_cast<std::int32_t>(static_cast<std::int64_t>(detail::get_u64(buf, pos)));
    ex.provenance_hash = detail::get_u64(buf, pos);
    const std::uint64_t n_rows = detail::get_u64(buf, pos);

    // Bounds check: header(6) + n_rows*5 + trailing hash(1) u64 words.
    const std::size_t need = 8 * (6 + n_rows * 5 + 1);
    if (buf.size() != need) return pr;

    ex.rows.reserve(static_cast<std::size_t>(n_rows));
    for (std::uint64_t i = 0; i < n_rows; ++i) {
        FeatureRow r;
        r.event_seq = detail::get_u64(buf, pos);
        r.ts_ns = detail::get_u64(buf, pos);
        r.feature = static_cast<std::int64_t>(detail::get_u64(buf, pos));
        r.label = static_cast<std::int64_t>(detail::get_u64(buf, pos));
        r.label_available = detail::get_u64(buf, pos) != 0;
        if (r.label_available)
            ++ex.n_available;
        else
            ++ex.n_censored;
        ex.rows.push_back(r);
    }
    const ReplayHash stored_contract = detail::get_u64(buf, pos);

    ex.n_rows = ex.rows.size();
    ex.contract_hash =
        feature_contract_hash(ex.schema, ex.provenance_hash, ex.rows);
    // Reject a tampered image whose recomputed hash disagrees with the stored one.
    if (ex.contract_hash != stored_contract) return pr;

    pr.ok = true;
    pr.schema_matched = (ex.schema.schema_version == expected_version);
    pr.value = ex;
    return pr;
}

// Read-only convenience wrapper. Holds a const view of an externally owned event
// stream and exposes ONLY export; it has no mutating member and no matching-
// engine coupling, making the offline / one-way-data-flow boundary explicit at
// the call site.
class FeatureExportSchemaContractReplay {
public:
    explicit FeatureExportSchemaContractReplay(
        const std::vector<FeatureEvent>& events) noexcept
        : events_(&events) {}

    FeatureExport export_to(const ExportSchema& schema,
                            const ObserverConfig& observer = {}) const {
        return export_features(*events_, schema, observer);
    }
    ReplayHash contract_hash(const ExportSchema& schema) const {
        return export_features(*events_, schema, {}).contract_hash;
    }
    std::size_t size() const noexcept { return events_->size(); }
    bool empty() const noexcept { return events_->empty(); }

private:
    const std::vector<FeatureEvent>* events_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline export types do NOT, which
// is the type-level proof that an exported row cannot be wired back into the
// kernel through these structs. A test supplies a POSITIVE CONTROL (a type that
// DOES expose such a member) to prove the trait actually discriminates.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// Static proof, evaluated at header-include time, that the offline types are
// pure data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the artifact is one-way OUT;
//   * export() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<ExportSchema>,
              "ExportSchema must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<FeatureEvent>,
              "FeatureEvent must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<FeatureRow>,
              "FeatureRow must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ObserverConfig>,
              "ObserverConfig must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<FeatureEvent>,
              "FeatureEvent must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<FeatureRow>,
              "FeatureRow must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<FeatureEvent>,
              "FeatureEvent must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FeatureRow>,
              "FeatureRow must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FeatureExport>,
              "FeatureExport must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FeatureExportSchemaContractReplay>,
              "FeatureExportSchemaContractReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<FeatureExport,
                          decltype(&FeatureExportSchemaContractReplay::export_to),
                          const FeatureExportSchemaContractReplay&,
                          const ExportSchema&, const ObserverConfig&>,
    "export_to() must be a read-only (const) diagnostic");

}  // namespace offline
