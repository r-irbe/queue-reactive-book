#pragma once

// Deterministic feature-export frame contract and synchronous reference sink.
// Each `ExportFrame` binds book observables at `seq` to the cumulative
// `replay_hash` after that event, enforcing strictly increasing sequence order.

#include "book_observables.hpp"
#include "event_trace.hpp"
#include "export_schema.hpp"   // ExportSchemaMeta (opt-in join-manifest hook)
#include "join_manifest.hpp"   // JoinValidity / JoinPolicy (opt-in, cold path)

#include <cstdint>
#include <vector>

namespace sim {

// One exported feature frame. `replay_hash` is the prefix hash through `seq`.
struct ExportFrame {
    EventSeq seq = 0;
    std::uint64_t replay_hash = 0;
    BookObservables snapshot{};
};

// Abstract downstream sink. A production async sink would implement this over a
// queue; the contract it must honour is defined by `SyncFeatureExportSink`.
class FeatureExportSink {
public:
    virtual ~FeatureExportSink() = default;
    virtual void on_frame(const ExportFrame& frame) = 0;
};

// Synchronous reference sink / test oracle. Records frames and enforces the
// prefix-hash determinism contract. It is intentionally allocation-simple and
// single-threaded; it is NOT a production exporter.
class SyncFeatureExportSink final : public FeatureExportSink {
public:
    void on_frame(const ExportFrame& frame) override {
        // Reject non-monotone seq instead of silently accepting reordered or
        // duplicated frames; the violation is observable via seq_violations().
        if (have_last_ && frame.seq <= last_seq_) {
            ++seq_violations_;
            return;
        }
        have_last_ = true;
        last_seq_ = frame.seq;
        last_hash_ = frame.replay_hash;
        frames_.push_back(frame);
    }

    // The final accepted frame's hash must equal the replay's final hash.
    bool verify_final_parity(std::uint64_t expected_final_hash) const noexcept {
        return have_last_ && last_hash_ == expected_final_hash;
    }

    bool seq_monotone() const noexcept { return seq_violations_ == 0; }
    std::size_t seq_violations() const noexcept { return seq_violations_; }
    std::uint64_t last_hash() const noexcept { return last_hash_; }
    std::size_t accepted_frames() const noexcept { return frames_.size(); }
    const std::vector<ExportFrame>& frames() const noexcept { return frames_; }

private:
    std::vector<ExportFrame> frames_{};
    EventSeq last_seq_ = 0;
    std::uint64_t last_hash_ = 0;
    std::size_t seq_violations_ = 0;
    bool have_last_ = false;
};

// Gates exported feature frames against a `JoinPolicy` causality and embargo
// check before forwarding usable rows to a `FeatureExportSink`.
class JoinManifestExportHook {
public:
    JoinManifestExportHook() noexcept = default;
    explicit JoinManifestExportHook(JoinPolicy policy) noexcept : policy_(policy) {}

    // Classify one feature->label join row, tally it into the running summary,
    // and return the verdict. No allocation, no mutation of any exported row.
    JoinValidity validate(const JoinManifestRow& row) noexcept {
        const JoinValidity v = validate_join(row, policy_);
        join_summary_add(summary_, v);
        return v;
    }

    // Convenience predicate: true iff the row is causally clean (Usable).
    bool admit(const JoinManifestRow& row) noexcept {
        return join_is_usable(validate(row));
    }

    // Gated emit: forward `frame` to `sink` ONLY if the row's join is causally
    // usable; otherwise record the leak and DROP the labelled frame (the kernel
    // and the feature columns are never touched). Returns the verdict so the
    // caller can react. Routing frames through this method is entirely opt-in;
    // sinks fed directly via `on_frame` are unaffected.
    JoinValidity emit_if_usable(FeatureExportSink& sink,
                                const ExportFrame& frame,
                                const JoinManifestRow& row) {
        const JoinValidity v = validate(row);
        if (join_is_usable(v)) {
            sink.on_frame(frame);
        }
        return v;
    }

    // Aggregate verdict counts observed so far (read-only diagnostics).
    const JoinManifestSummary& summary() const noexcept { return summary_; }
    const JoinPolicy& policy() const noexcept { return policy_; }

    // True iff any non-usable, non-diagnostic verdict (a leak) has been seen.
    bool any_leak() const noexcept {
        return summary_.leaky_seq != 0 || summary_.leaky_time != 0 ||
               summary_.non_causal != 0 || summary_.split_mismatch != 0;
    }

    // Schema-meta describing the emitted join manifest, so a downstream consumer
    // can interpret the exported seq/time coordinates without out-of-band
    // knowledge. Identical to the standalone `join_manifest_schema_meta()`.
    ExportSchemaMeta schema_meta() const noexcept {
        return join_manifest_schema_meta();
    }

private:
    JoinPolicy policy_{};
    JoinManifestSummary summary_{};
};

} // namespace sim
