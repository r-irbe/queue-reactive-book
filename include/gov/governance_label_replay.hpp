#pragma once

#include <cstdint>
#include <vector>

// --- OFFLINE governance-label replay + accounting hash (cold path) -----------
//
// PROBLEM: the MOC's load-bearing boundary claim is that agent / governance /
// social-choice artifacts (HITL approvals, risk gates, authorization,
// safety-review, audit attestations) are CONTROL-PLANE / OPERATOR-WORKFLOW
// background, never matching/fill/fee/accounting truth-state. That disposition
// previously rested only on review-type hooks. This header is the deterministic
// fixture the boundary needs: a governance-decision/label event stream replayed
// deterministically into a stable, order-sensitive accounting hash.
//
// Invariants this provides:
//   * INVARIANCE  — replaying an identical label sequence yields an IDENTICAL
//                   accounting hash (deterministic, pure function of inputs).
//   * TAMPER-EVIDENCE — reordering the stream or mutating any field of any
//                   label changes the hash (order-sensitive integrity check).
//   * READ-ONLY / OFFLINE — replay never constructs, mutates, or references any
//                   matching-kernel state. This header includes NOTHING from
//                   `sim/` and knows nothing about orders, fills, fees, the book,
//                   or `FlatLadderBook`. It only folds already-recorded
//                   governance labels into a hash. It cannot alter
//                   matching/fills/fees/inventory.
//
// Determinism: a pure function of fixed-width integer/enum inputs. No floating
// point, no allocation beyond the caller-owned input, no hash-table iteration
// order, no platform-dependent `std::hash`. The accounting hash is a streaming
// FNV-1a-64 over a canonical little-endian byte encoding of (ordinal, fields).

namespace gov {

// Kinds of governance / control-plane labels. These annotate declared scenarios
// and offline review; none of them is exchange/accounting semantics.
enum class GovLabelKind : std::uint8_t {
    HitlApproval = 1,      // human-in-the-loop approval label
    RiskGate = 2,          // offline risk-gate decision
    Authorization = 3,     // permission / authorization label
    SafetyReview = 4,      // safety-review attestation
    AuditAttestation = 5,  // audit attestation / sign-off
};

// Governance decision recorded on a label. Not a fill, not an order.
enum class GovDecision : std::uint8_t {
    Pending = 0,
    Approve = 1,
    Reject = 2,
    Escalate = 3,
};

// One governance-label event. All fields are integer/enum control-plane
// metadata; nothing here originates market state.
struct GovernanceLabel {
    std::uint64_t seq = 0;         // monotonic governance-event sequence id
    GovLabelKind kind = GovLabelKind::HitlApproval;
    GovDecision decision = GovDecision::Pending;
    std::uint64_t subject_id = 0;  // artifact / order-intent the label governs
    std::uint64_t actor_id = 0;    // human / operator / agent emitting the label
    std::uint64_t ts_ns = 0;       // governance-event time (event-time, ns)
};

using AccountingHash = std::uint64_t;

namespace detail {

inline constexpr AccountingHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr AccountingHash kFnvPrime = 1099511628211ULL;

inline constexpr AccountingHash fnv1a_byte(AccountingHash h, std::uint8_t b) noexcept {
    h ^= static_cast<AccountingHash>(b);
    h *= kFnvPrime;
    return h;
}

// Fold a fixed-width integer in canonical little-endian order so the hash does
// not depend on host endianness.
inline constexpr AccountingHash fnv1a_u64(AccountingHash h, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = fnv1a_byte(h, static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
    return h;
}

}  // namespace detail

// Replay a governance-label stream into a deterministic, order-sensitive
// accounting hash. Pure, read-only, offline: it copies nothing, mutates
// nothing, and touches no matching-kernel state.
//
// Order sensitivity comes from folding each label's ORDINAL position together
// with its field bytes, so reordering two distinct labels changes the hash.
inline AccountingHash replay_accounting_hash(const std::vector<GovernanceLabel>& stream) noexcept {
    AccountingHash h = detail::kFnvOffsetBasis;
    std::uint64_t ordinal = 0;
    for (const GovernanceLabel& label : stream) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, label.seq);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(label.kind));
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(label.decision));
        h = detail::fnv1a_u64(h, label.subject_id);
        h = detail::fnv1a_u64(h, label.actor_id);
        h = detail::fnv1a_u64(h, label.ts_ns);
        ++ordinal;
    }
    // Fold the count so a truncated/extended stream cannot collide with a
    // prefix that happened to hash identically.
    h = detail::fnv1a_u64(h, ordinal);
    return h;
}

// Read-only convenience wrapper. Holds a const view of an externally owned
// stream and exposes only the replay; it has no mutating member and no engine
// coupling, making the offline / non-engine boundary explicit at the call site.
class GovernanceAccountingReplay {
public:
    explicit GovernanceAccountingReplay(const std::vector<GovernanceLabel>& stream) noexcept
        : stream_(&stream) {}

    AccountingHash replay() const noexcept { return replay_accounting_hash(*stream_); }

    std::size_t size() const noexcept { return stream_->size(); }

private:
    const std::vector<GovernanceLabel>* stream_;
};

}  // namespace gov
