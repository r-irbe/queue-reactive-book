#pragma once

#include "event_trace.hpp"   // EventSeq (kernel-free: pulls only types.hpp)
#include "export_schema.hpp" // ExportSchemaMeta / TimestampBasis
#include "types.hpp"         // TimeNs

#include <cstdint>

namespace sim {

// --- Feature/label JOIN MANIFEST (Tier-4 proposal CML-004, offline, cold path) -
//
// PROBLEM: features expose `seq`/`ts` (feature_export.hpp) and labels expose
// fill/eval `seq`/`ts` (label_export.hpp), but NO schema-level artifact states
// the join causality `feature_seq + embargo < label_start_seq <= label_eval_seq`,
// the embargo gap, or the train/test split boundary. A leaky join (a label whose
// information window opens at or before the feature observation, plus embargo)
// can therefore silently contaminate a training set.
//
// This is a header-only, cold-path VALIDATOR. It never feeds the frozen matching
// kernel; it only inspects already-exported feature/label row coordinates and
// classifies whether the join is causally usable for training. It does NOT
// impute, reorder, or mutate any row, and it does NOT change any existing column
// array. Determinism: a pure function of the integer/seq inputs.
//
// Half-open embargo semantics: the label's information window must START strictly
// after `feature_seq + embargo_events` (and `feature_ts + embargo_ns`). A label
// that starts AT the embargo boundary is rejected as leaky (the boundary event
// itself is inside the embargo).

enum class JoinValidity : std::uint8_t {
    Usable = 0,         // causally clean: safe to join feature->label for training
    LeakySeq,           // label_start_seq <= feature_seq + embargo_events
    LeakyTime,          // label_start_ts  <= feature_ts  + embargo_ns
    NonCausalEval,      // label_eval_seq < label_start_seq (eval precedes start)
    SplitMismatch,      // feature_split_id != label_split_id (cross-fold leak)
    DiagnosticOnly,     // explicitly flagged diagnostic; never a training label
};

inline bool join_is_usable(JoinValidity v) noexcept {
    return v == JoinValidity::Usable;
}

// One feature->label join coordinate row. All fields are copied from already
// exported rows; nothing here originates new market state.
struct JoinManifestRow {
    EventSeq feature_seq = 0;
    TimeNs   feature_ts = 0;
    EventSeq label_start_seq = 0;   // first event whose info the label may use
    TimeNs   label_start_ts = 0;
    EventSeq label_eval_seq = 0;    // event at which the label is realized
    TimeNs   label_eval_ts = 0;
    std::uint64_t feature_split_id = 0;  // train/test/fold id of the feature row
    std::uint64_t label_split_id = 0;    // train/test/fold id of the label row
    bool diagnostic_only = false;        // row is diagnostic, never a train label
};

// Embargo gap applied AFTER the feature observation before any label information
// window may open. 0 disables the corresponding (event or time) embargo axis.
struct JoinPolicy {
    EventSeq embargo_events = 0;
    TimeNs   embargo_ns = 0;
    bool     require_same_split = true;  // a usable training join stays in one fold
};

// Classify a single join row under a policy. Pure, deterministic, no allocation.
inline JoinValidity validate_join(const JoinManifestRow& r,
                                  const JoinPolicy& p) noexcept {
    if (r.diagnostic_only) {
        return JoinValidity::DiagnosticOnly;
    }
    if (r.label_eval_seq < r.label_start_seq) {
        return JoinValidity::NonCausalEval;
    }
    // Half-open embargo: label info must START strictly past the embargo boundary.
    if (r.label_start_seq <= r.feature_seq + p.embargo_events) {
        return JoinValidity::LeakySeq;
    }
    if (p.embargo_ns > 0 || r.label_start_ts <= r.feature_ts) {
        if (r.label_start_ts <= r.feature_ts + p.embargo_ns) {
            return JoinValidity::LeakyTime;
        }
    }
    if (p.require_same_split && r.feature_split_id != r.label_split_id) {
        return JoinValidity::SplitMismatch;
    }
    return JoinValidity::Usable;
}

// Aggregate count of join outcomes across a manifest (read-only diagnostics).
struct JoinManifestSummary {
    std::uint64_t total = 0;
    std::uint64_t usable = 0;
    std::uint64_t leaky_seq = 0;
    std::uint64_t leaky_time = 0;
    std::uint64_t non_causal = 0;
    std::uint64_t split_mismatch = 0;
    std::uint64_t diagnostic_only = 0;
};

inline void join_summary_add(JoinManifestSummary& s, JoinValidity v) noexcept {
    ++s.total;
    switch (v) {
        case JoinValidity::Usable:          ++s.usable; break;
        case JoinValidity::LeakySeq:        ++s.leaky_seq; break;
        case JoinValidity::LeakyTime:       ++s.leaky_time; break;
        case JoinValidity::NonCausalEval:   ++s.non_causal; break;
        case JoinValidity::SplitMismatch:   ++s.split_mismatch; break;
        case JoinValidity::DiagnosticOnly:  ++s.diagnostic_only; break;
    }
}

// Schema-metadata descriptor for an exported join manifest, so a consumer can
// interpret the seq/time coordinates without out-of-band knowledge.
inline ExportSchemaMeta join_manifest_schema_meta() noexcept {
    ExportSchemaMeta m{};
    m.schema_name = "join_manifest";
    m.timestamp_basis = TimestampBasis::EventTime;
    m.quantity_unit = QuantityUnit::Unspecified;
    m.price_unit = QuantityUnit::Unspecified;
    m.label_availability = LabelAvailability::Available;
    return m;
}

}  // namespace sim
