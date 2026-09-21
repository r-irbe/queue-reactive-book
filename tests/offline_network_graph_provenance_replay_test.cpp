#include <catch2/catch_test_macros.hpp>

#include "offline/network_graph_provenance_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic worked, provenance-gated network-graph diagnostic fixture
// ("No worked, provenance-gated offline diagnostic example (schema +
//  diagnostics-on/off replay-hash fixture) exists yet").
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, the four properties the
// MOC's load-bearing off-kernel boundary needs:
//   (a) REPLAY-HASH DETERMINISM + ORDER-SENSITIVITY -- same ordered event
//       stream -> identical graph state + FNV hash; reordering the SAME multiset
//       of observations -> a different hash even though the set-defined graph
//       state is invariant.
//   (b) PROJECTION / WEIGHT / THRESHOLD CORRECTNESS on a small known bipartite
//       example with hand-computed expected edges, shared-neighbour weights, and
//       cutoff (w_o) behaviour.
//   (c) PROVENANCE COMPLETENESS -- every sub-threshold dropped edge and every
//       excluded node (with Isolated vs SubThresholdOnly reason) is recorded.
//   (d) DIAGNOSTICS-ON / DIAGNOSTICS-OFF EQUIVALENCE of the underlying graph
//       state (identical replay hash + edges + nodes), proven read-only /
//       non-feedback via static_assert + a POSITIVE control discriminator.
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/network_graph_provenance_replay.hpp` (std-includes
// only, zero `sim::` symbols).
// =============================================================================

namespace {

using offline::CoOccurrenceEvent;
using offline::DroppedEdge;
using offline::ExcludedNode;
using offline::ExclusionReason;
using offline::NetworkGraphProvenanceReplay;
using offline::NetworkProjection;
using offline::NodeId;
using offline::ProjectedEdge;

CoOccurrenceEvent obs(std::uint64_t seq, NodeId a, NodeId b) {
    CoOccurrenceEvent e;
    e.seq = seq;
    e.a_id = a;
    e.b_id = b;
    return e;
}

// A small known bipartite example: projected-kind (A) accounts {1,2,3,4,5} that
// co-occur with opposite-kind (B) instruments {10..14}.
//
//   neighbours(1) = {10,11}
//   neighbours(2) = {10,11,12}
//   neighbours(3) = {12,13}
//   neighbours(4) = {13}
//   neighbours(5) = {14}        (shared with nobody -> isolated)
//
// Hand-computed shared-neighbour edge weights (count of common instruments):
//   (1,2) = |{10,11}|        = 2
//   (2,3) = |{12}|           = 1
//   (3,4) = |{13}|           = 1
//   all other A-pairs        = 0  (no link)
//
//   At cutoff w_o = 1 : edges {(1,2):2,(2,3):1,(3,4):1}; included {1,2,3,4};
//                       node 5 excluded (Isolated); no dropped edges.
//   At cutoff w_o = 2 : edges {(1,2):2}; included {1,2};
//                       dropped {(2,3):1,(3,4):1}; excluded
//                       {3:SubThresholdOnly, 4:SubThresholdOnly, 5:Isolated}.
std::vector<CoOccurrenceEvent> sample_stream() {
    return {
        obs(1, 1, 10), obs(2, 1, 11),
        obs(3, 2, 10), obs(4, 2, 11), obs(5, 2, 12),
        obs(6, 3, 12), obs(7, 3, 13),
        obs(8, 4, 13),
        obs(9, 5, 14),
    };
}

const ProjectedEdge* find_edge(const NetworkProjection& p, NodeId lo, NodeId hi) {
    for (const ProjectedEdge& e : p.edges)
        if (e.lo == lo && e.hi == hi) return &e;
    return nullptr;
}

const ExcludedNode* find_excluded(const NetworkProjection& p, NodeId id) {
    for (const ExcludedNode& n : p.provenance.excluded_nodes)
        if (n.id == id) return &n;
    return nullptr;
}

bool has_dropped(const NetworkProjection& p, NodeId lo, NodeId hi, std::int64_t w) {
    for (const DroppedEdge& d : p.provenance.dropped_edges)
        if (d.lo == lo && d.hi == hi && d.weight == w) return true;
    return false;
}

bool has_node(const NetworkProjection& p, NodeId id) {
    for (NodeId n : p.included_nodes)
        if (n == id) return true;
    return false;
}

}  // namespace

TEST_CASE("network-graph replay-hash is deterministic for identical streams") {
    const std::vector<CoOccurrenceEvent> a = sample_stream();
    const std::vector<CoOccurrenceEvent> b = sample_stream();

    const NetworkProjection pa = offline::project_network(a, 1, true);
    const NetworkProjection pb = offline::project_network(b, 1, true);

    // Identical ordered streams -> identical graph state and identical hash, and
    // recomputing is bit-stable.
    REQUIRE(pa.replay_hash == pb.replay_hash);
    REQUIRE(offline::project_network(a, 1, true).replay_hash == pa.replay_hash);
    REQUIRE(offline::network_graph_hash(a, pa) == pa.replay_hash);

    // The read-only wrapper agrees with the free function.
    const NetworkGraphProvenanceReplay replay(a);
    REQUIRE(replay.hash(1) == pa.replay_hash);
    REQUIRE(replay.project(1).edges.size() == pa.edges.size());
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<CoOccurrenceEvent> empty;
    REQUIRE(offline::project_network(empty, 1, true).replay_hash ==
            offline::project_network(empty, 1, true).replay_hash);
    REQUIRE(offline::project_network(empty, 1, true).replay_hash != pa.replay_hash);
}

TEST_CASE("network-graph replay-hash is order-sensitive but graph state is not") {
    const std::vector<CoOccurrenceEvent> base = sample_stream();
    const NetworkProjection p_base = offline::project_network(base, 1, true);

    // Reorder the SAME multiset of observations: the set-defined projection is
    // invariant, yet the ordinal-folded replay hash changes (order-sensitive
    // integrity -- the falsifier for a silently reorder-insensitive replay).
    std::vector<CoOccurrenceEvent> reordered = base;
    std::swap(reordered[0], reordered.back());
    const NetworkProjection p_reordered = offline::project_network(reordered, 1, true);

    REQUIRE(p_reordered.replay_hash != p_base.replay_hash);

    // Graph state (edges + weights + included nodes) is identical under reorder.
    REQUIRE(p_reordered.edges.size() == p_base.edges.size());
    for (std::size_t i = 0; i < p_base.edges.size(); ++i) {
        REQUIRE(p_reordered.edges[i].lo == p_base.edges[i].lo);
        REQUIRE(p_reordered.edges[i].hi == p_base.edges[i].hi);
        REQUIRE(p_reordered.edges[i].weight == p_base.edges[i].weight);
    }
    REQUIRE(p_reordered.included_nodes == p_base.included_nodes);

    // Tampering with an observation (changing a shared neighbour) changes both.
    std::vector<CoOccurrenceEvent> tampered = base;
    tampered[2].b_id = 99;  // account 2 no longer shares instrument 10 with acct 1
    REQUIRE(offline::project_network(tampered, 1, true).replay_hash != p_base.replay_hash);
}

TEST_CASE("network-graph projection/weight/threshold correctness on known example") {
    const std::vector<CoOccurrenceEvent> stream = sample_stream();

    SECTION("cutoff w_o = 1: all shared-neighbour links survive") {
        const NetworkProjection p = offline::project_network(stream, 1, true);

        REQUIRE(p.weight_threshold == 1);
        REQUIRE(p.edges.size() == 3);

        const ProjectedEdge* e12 = find_edge(p, 1, 2);
        const ProjectedEdge* e23 = find_edge(p, 2, 3);
        const ProjectedEdge* e34 = find_edge(p, 3, 4);
        REQUIRE(e12 != nullptr);
        REQUIRE(e23 != nullptr);
        REQUIRE(e34 != nullptr);
        REQUIRE(e12->weight == 2);  // shared {10,11}
        REQUIRE(e23->weight == 1);  // shared {12}
        REQUIRE(e34->weight == 1);  // shared {13}

        // No spurious link between non-co-occurring accounts.
        REQUIRE(find_edge(p, 1, 3) == nullptr);
        REQUIRE(find_edge(p, 1, 4) == nullptr);

        // Included: {1,2,3,4}; node 5 shares with nobody -> isolated, excluded.
        REQUIRE(p.included_nodes == std::vector<NodeId>{1, 2, 3, 4});
        REQUIRE_FALSE(has_node(p, 5));
    }

    SECTION("cutoff w_o = 2: sub-threshold edges and their nodes drop out") {
        const NetworkProjection p = offline::project_network(stream, 2, true);

        REQUIRE(p.weight_threshold == 2);
        REQUIRE(p.edges.size() == 1);
        const ProjectedEdge* e12 = find_edge(p, 1, 2);
        REQUIRE(e12 != nullptr);
        REQUIRE(e12->weight == 2);

        // Sub-threshold (1,2)-aside edges are gone from the graph state.
        REQUIRE(find_edge(p, 2, 3) == nullptr);
        REQUIRE(find_edge(p, 3, 4) == nullptr);

        // Only the >= w_o edge's endpoints remain.
        REQUIRE(p.included_nodes == std::vector<NodeId>{1, 2});
    }

    SECTION("a threshold below 1 is clamped to 1 (a link needs a shared neighbour)") {
        const NetworkProjection p = offline::project_network(stream, 0, true);
        REQUIRE(p.weight_threshold == 1);
        REQUIRE(p.edges.size() == 3);
    }
}

TEST_CASE("network-graph provenance ledger is complete (explicit missingness)") {
    const std::vector<CoOccurrenceEvent> stream = sample_stream();
    const NetworkProjection p = offline::project_network(stream, 2, true);

    // Dropped edges: exactly the two sub-threshold (weight==1) links, recorded
    // with their original weight -- never silently discarded.
    REQUIRE(p.provenance.dropped_edges.size() == 2);
    REQUIRE(has_dropped(p, 2, 3, 1));
    REQUIRE(has_dropped(p, 3, 4, 1));

    // Excluded nodes: node 5 is Isolated (never shared a neighbour); nodes 3 and
    // 4 are SubThresholdOnly (their only edges were dropped by the cutoff).
    REQUIRE(p.provenance.excluded_nodes.size() == 3);

    const ExcludedNode* n5 = find_excluded(p, 5);
    const ExcludedNode* n3 = find_excluded(p, 3);
    const ExcludedNode* n4 = find_excluded(p, 4);
    REQUIRE(n5 != nullptr);
    REQUIRE(n3 != nullptr);
    REQUIRE(n4 != nullptr);
    REQUIRE(n5->reason == ExclusionReason::Isolated);
    REQUIRE(n3->reason == ExclusionReason::SubThresholdOnly);
    REQUIRE(n4->reason == ExclusionReason::SubThresholdOnly);

    // Conservation: every A-node that appears is either included or excluded,
    // with no double-counting -- the ledger accounts for all 5 accounts.
    REQUIRE(p.included_nodes.size() + p.provenance.excluded_nodes.size() == 5);

    // At w_o = 1, only the lone isolated node is missing; no edge is dropped.
    const NetworkProjection p1 = offline::project_network(stream, 1, true);
    REQUIRE(p1.provenance.dropped_edges.empty());
    REQUIRE(p1.provenance.excluded_nodes.size() == 1);
    REQUIRE(find_excluded(p1, 5) != nullptr);
    REQUIRE(find_excluded(p1, 5)->reason == ExclusionReason::Isolated);
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed a graph metric back into the kernel
};

}  // namespace

TEST_CASE("network-graph diagnostics on/off leave graph state and hash unchanged") {
    const std::vector<CoOccurrenceEvent> stream = sample_stream();

    for (std::int64_t w_o : {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}}) {
        const NetworkProjection on = offline::project_network(stream, w_o, true);
        const NetworkProjection off = offline::project_network(stream, w_o, false);

        // (d) Turning the diagnostic OFF must NOT change the provenance/graph
        // state: the diagnostics-on/off replay hash is EQUAL.
        REQUIRE(on.replay_hash == off.replay_hash);

        // Graph state (threshold, edges, included nodes) is byte-for-byte equal.
        REQUIRE(on.weight_threshold == off.weight_threshold);
        REQUIRE(on.edges.size() == off.edges.size());
        for (std::size_t i = 0; i < on.edges.size(); ++i) {
            REQUIRE(on.edges[i].lo == off.edges[i].lo);
            REQUIRE(on.edges[i].hi == off.edges[i].hi);
            REQUIRE(on.edges[i].weight == off.edges[i].weight);
        }
        REQUIRE(on.included_nodes == off.included_nodes);

        // The ONLY difference is the read-only provenance ledger: present with
        // diagnostics on, empty with diagnostics off.
        REQUIRE(off.provenance.empty());
        REQUIRE(off.diagnostics_enabled == false);
        REQUIRE(on.diagnostics_enabled == true);
    }

    // At w_o = 2 the diagnostic genuinely has something to report (so the on/off
    // equality above is not a vacuous "both empty" case).
    const NetworkProjection on2 = offline::project_network(stream, 2, true);
    REQUIRE_FALSE(on2.provenance.empty());
}

TEST_CASE("network-graph projection is read-only and has no matching coupling") {
    std::vector<CoOccurrenceEvent> stream = sample_stream();
    const std::vector<CoOccurrenceEvent> before = stream;

    const NetworkProjection p = offline::project_network(stream, 2, true);
    (void)p;

    // NO mutation of the input stream: the projection is offline/read-only and
    // cannot alter matching state because it never touches engine state.
    REQUIRE(stream.size() == before.size());
    for (std::size_t i = 0; i < stream.size(); ++i) {
        REQUIRE(stream[i].seq == before[i].seq);
        REQUIRE(stream[i].a_id == before[i].a_id);
        REQUIRE(stream[i].b_id == before[i].b_id);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::CoOccurrenceEvent>,
                  "CoOccurrenceEvent must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::ProjectedEdge>,
                  "ProjectedEdge must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::NetworkProjection>,
                  "NetworkProjection must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::ProvenanceRecord>,
                  "ProvenanceRecord must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::NetworkGraphProvenanceReplay>,
                  "NetworkGraphProvenanceReplay must not be a matching input");

    // The projection accepts only a const stream and returns a plain value: it
    // has no mutating engine API surface (compile-time evidence of the boundary).
    static_assert(
        std::is_same_v<decltype(offline::project_network(stream, 1, true)),
                       offline::NetworkProjection>,
        "project_network must return a derived projection, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::ProjectedEdge>,
                  "ProjectedEdge must be pure data");
}
