// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "replay_diagnostics.hpp"

#include <cstdint>

namespace sim {

// Verifiable replay progress snapshot binding `replay_hash`, event/trade counts,
// last causal (seq, ts) position, and ordering-violation counters.

struct ReplayCheckpoint {
    std::uint64_t input_events = 0;
    std::uint64_t accepted_events = 0;
    std::uint64_t rejected_events = 0;
    std::uint64_t trades = 0;
    Qty filled_qty = 0;
    std::uint64_t replay_hash = 0;
    // Last causal position observed, so a resumed run can confirm it continues
    // from exactly where the checkpoint ended.
    bool has_last_seq = false;
    EventSeq last_seq = 0;
    bool has_last_ts = false;
    TimeNs last_ts = 0;
    // Causal-ordering health carried forward so a checkpoint cannot launder away
    // violations seen before it was taken.
    std::uint64_t seq_order_violations = 0;
    std::uint64_t time_order_violations = 0;
};

inline ReplayCheckpoint make_replay_checkpoint(const ReplayDiagnostics& d) noexcept {
    return ReplayCheckpoint{.input_events = d.input_events,
                            .accepted_events = d.accepted_events,
                            .rejected_events = d.rejected_events,
                            .trades = d.trades,
                            .filled_qty = d.filled_qty,
                            .replay_hash = d.replay_hash,
                            .has_last_seq = d.has_last_seq,
                            .last_seq = d.last_seq,
                            .has_last_ts = d.has_last_ts,
                            .last_ts = d.last_ts,
                            .seq_order_violations = d.seq_order_violations,
                            .time_order_violations = d.time_order_violations};
}

struct CheckpointVerification {
    bool matches = true;
    bool hash_mismatch = false;      // the content digest differs -> divergence
    bool progress_mismatch = false;  // event/trade/fill counts differ
    bool position_mismatch = false;  // last seq/ts differ
};

// Verify an `actual` checkpoint against an `expected` one. The hash is the
// decisive content-divergence signal; the count/position checks localize WHERE
// a divergence first shows up. A hash match with mismatched counts is itself a
// red flag (a checkpoint that under-reports progress), so it is reported, not
// hidden.
inline CheckpointVerification verify_checkpoint(const ReplayCheckpoint& expected,
                                                const ReplayCheckpoint& actual) noexcept {
    CheckpointVerification out{};
    out.hash_mismatch = expected.replay_hash != actual.replay_hash;
    out.progress_mismatch = expected.input_events != actual.input_events ||
                            expected.accepted_events != actual.accepted_events ||
                            expected.rejected_events != actual.rejected_events ||
                            expected.trades != actual.trades ||
                            expected.filled_qty != actual.filled_qty;
    out.position_mismatch = expected.has_last_seq != actual.has_last_seq ||
                            expected.last_seq != actual.last_seq ||
                            expected.has_last_ts != actual.has_last_ts ||
                            expected.last_ts != actual.last_ts;
    out.matches =
        !out.hash_mismatch && !out.progress_mismatch && !out.position_mismatch;
    return out;
}

} // namespace sim
