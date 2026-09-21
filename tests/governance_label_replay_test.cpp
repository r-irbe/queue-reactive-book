#include <catch2/catch_test_macros.hpp>

#include "gov/governance_label_replay.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

//
// Asserts the three properties the MOC's boundary claim needs:
//   (a) replay-hash INVARIANCE for identical governance-label sequences;
//   (b) the hash CHANGES under reorder / tamper (tamper-evidence);
//   (c) replay performs NO engine/matching mutation (read-only, offline).

namespace {

std::vector<gov::GovernanceLabel> sample_stream() {
    using gov::GovDecision;
    using gov::GovLabelKind;
    return {
        {1, GovLabelKind::HitlApproval, GovDecision::Approve, 1001, 7, 1000},
        {2, GovLabelKind::RiskGate, GovDecision::Escalate, 1001, 9, 2500},
        {3, GovLabelKind::Authorization, GovDecision::Approve, 1002, 7, 4000},
        {4, GovLabelKind::SafetyReview, GovDecision::Reject, 1002, 11, 5200},
        {5, GovLabelKind::AuditAttestation, GovDecision::Approve, 1003, 7, 6100},
    };
}

}  // namespace

TEST_CASE("governance replay-hash is invariant for identical sequences") {
    const std::vector<gov::GovernanceLabel> a = sample_stream();
    const std::vector<gov::GovernanceLabel> b = sample_stream();

    const gov::AccountingHash h1 = gov::replay_accounting_hash(a);
    const gov::AccountingHash h2 = gov::replay_accounting_hash(b);

    // Identical sequences -> identical accounting hash, and replaying the same
    // sequence again is bit-stable.
    REQUIRE(h1 == h2);
    REQUIRE(gov::replay_accounting_hash(a) == h1);

    // The read-only wrapper agrees with the free function.
    const gov::GovernanceAccountingReplay replay(a);
    REQUIRE(replay.replay() == h1);
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<gov::GovernanceLabel> empty;
    REQUIRE(gov::replay_accounting_hash(empty) == gov::replay_accounting_hash(empty));
    REQUIRE(gov::replay_accounting_hash(empty) != h1);
}

TEST_CASE("governance replay-hash changes under reorder or tamper") {
    const std::vector<gov::GovernanceLabel> base = sample_stream();
    const gov::AccountingHash h_base = gov::replay_accounting_hash(base);

    SECTION("reordering two distinct labels changes the hash") {
        std::vector<gov::GovernanceLabel> reordered = base;
        std::swap(reordered[1], reordered[3]);
        REQUIRE(gov::replay_accounting_hash(reordered) != h_base);
    }

    SECTION("mutating the decision field changes the hash") {
        std::vector<gov::GovernanceLabel> tampered = base;
        tampered[2].decision = gov::GovDecision::Reject;  // was Approve
        REQUIRE(gov::replay_accounting_hash(tampered) != h_base);
    }

    SECTION("mutating the kind field changes the hash") {
        std::vector<gov::GovernanceLabel> tampered = base;
        tampered[0].kind = gov::GovLabelKind::AuditAttestation;  // was HitlApproval
        REQUIRE(gov::replay_accounting_hash(tampered) != h_base);
    }

    SECTION("mutating any integer field changes the hash") {
        std::vector<gov::GovernanceLabel> seq_t = base;
        seq_t[1].seq += 1;
        REQUIRE(gov::replay_accounting_hash(seq_t) != h_base);

        std::vector<gov::GovernanceLabel> subj_t = base;
        subj_t[3].subject_id += 1;
        REQUIRE(gov::replay_accounting_hash(subj_t) != h_base);

        std::vector<gov::GovernanceLabel> actor_t = base;
        actor_t[4].actor_id += 1;
        REQUIRE(gov::replay_accounting_hash(actor_t) != h_base);

        std::vector<gov::GovernanceLabel> ts_t = base;
        ts_t[0].ts_ns += 1;
        REQUIRE(gov::replay_accounting_hash(ts_t) != h_base);
    }

    SECTION("truncation / extension changes the hash") {
        std::vector<gov::GovernanceLabel> truncated = base;
        truncated.pop_back();
        REQUIRE(gov::replay_accounting_hash(truncated) != h_base);

        std::vector<gov::GovernanceLabel> extended = base;
        extended.push_back(base.front());
        REQUIRE(gov::replay_accounting_hash(extended) != h_base);
    }
}

TEST_CASE("governance replay is read-only and engine-free") {
    std::vector<gov::GovernanceLabel> stream = sample_stream();
    const std::vector<gov::GovernanceLabel> before = stream;

    const gov::AccountingHash h = gov::replay_accounting_hash(stream);
    (void)h;

    // (c) NO mutation of the input stream: replay is offline/read-only and
    // cannot alter matching/fills/fees because it never touches engine state.
    REQUIRE(stream.size() == before.size());
    for (std::size_t i = 0; i < stream.size(); ++i) {
        REQUIRE(stream[i].seq == before[i].seq);
        REQUIRE(stream[i].kind == before[i].kind);
        REQUIRE(stream[i].decision == before[i].decision);
        REQUIRE(stream[i].subject_id == before[i].subject_id);
        REQUIRE(stream[i].actor_id == before[i].actor_id);
        REQUIRE(stream[i].ts_ns == before[i].ts_ns);
    }

    // The replay accepts only a const stream and returns a value: it has no
    // mutating engine API surface (compile-time evidence of the offline boundary).
    static_assert(std::is_same_v<decltype(gov::replay_accounting_hash(stream)), gov::AccountingHash>,
                  "replay must return a plain accounting hash, not engine state");
    static_assert(noexcept(gov::replay_accounting_hash(stream)),
                  "offline replay must not throw / mutate engine state");
}
