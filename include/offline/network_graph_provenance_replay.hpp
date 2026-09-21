#pragma once

// =============================================================================
// Offline provenance-gated network-graph diagnostic REPLAY fixture.
//
// ("No worked, provenance-gated offline diagnostic example (schema +
//  diagnostics-on/off replay-hash fixture) exists yet -- the validation hooks
//  are named but not implemented", Wave-K re-council).
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. (The
//   FlatLadderBook / sim:: names appear in THIS FILE only inside comments, to
//   state the non-import boundary; the grep audit expects no live reference.)
//   A graph/network is an INTERPRETATION of underlying data, never matching
//   data: the one-mode projection, shared-neighbour edge weights, the weight
//   cutoff, and the missingness/provenance record computed here are PURE OFFLINE
//   artifacts over an already-recorded, replayable co-occurrence event stream.
//   Data flows ONE WAY, OUT of replay and into derived diagnostics; no topology,
//   centrality, weight, or provenance value may ever enter matching, fills,
//   accounting, inventory, fees, replay hashes, or a hot-path branch. This
//   header is the deterministic proof of that one-way, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   The MOC's projection-provenance route (source-confirmed 2026-06-20 from
//   De Masi & Gallegati and Chmiel et al., in *Econophysics of Markets and
//   Business Networks*, 2007) fixes the construction semantics this fixture
//   implements verbatim:
//     * a bipartite (e.g. account<->instrument / maker<->venue) co-occurrence
//       graph is reduced to a ONE-MODE projection: two same-kind (A) nodes are
//       linked iff they share >= 1 common opposite-kind (B) neighbour;
//     * multiplicity is recovered as an EDGE WEIGHT = count of shared
//       neighbours (De Masi-Gallegati "weight = number of common firms";
//       Chmiel et al. w_AB = |Z(A) intersect Z(B)|, w_ij = |B(i) intersect B(j)|);
//     * a weight CUTOFF `w_o` drops sub-threshold edges AND the nodes left
//       isolated by that drop ("nodes are connected only when their edge weight
//       is no less than an assumed cutoff parameter w_o");
//     * MISSINGNESS is EXPLICIT: every dropped edge and every excluded node is
//       recorded in a provenance ledger, never silently discarded.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   project_network() is a PURE function of the (immutable) ordered event
//   stream: an identical ordered stream yields an identical graph state and an
//   identical replay hash; reordering the SAME multiset of observations changes
//   the replay hash (order-sensitive integrity) even though the set-defined
//   graph state is invariant. The computation takes the events by const
//   reference, returns a plain value, and exposes no API that could mutate or be
//   consumed by matching state.
//
// DIAGNOSTICS-ON / DIAGNOSTICS-OFF EQUIVALENCE (the read-only / non-feedback
// proof the MOC requires via diagnostics-on/off replay-hash equality):
//   The GRAPH STATE (threshold, surviving weighted edges, included nodes) is
//   computed identically whether diagnostics are ON or OFF. The diagnostic --
//   the missingness/provenance ledger of dropped edges and excluded nodes -- is
//   populated ONLY when diagnostics are ON, is strictly READ-ONLY observer
//   state, and is DELIBERATELY EXCLUDED from `replay_hash`. Hence turning the
//   diagnostic off cannot change the provenance/graph state hash: a diagnostic
//   that changed the replay hash would be feedback, not observation.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using NodeId = std::uint64_t;

// Kind of node in the bipartite source graph. The projection reduces onto the
// A-kind; B-kind nodes are the shared "opposite-kind neighbours". Pure label.
enum class NodeKind : std::uint8_t { A = 0, B = 1 };

// One replayable bipartite co-occurrence observation: projected-kind node
// `a_id` co-occurs with opposite-kind node `b_id` (e.g. account a_id traded
// instrument b_id, or maker a_id quoted on venue b_id). Pure integer data: no
// pointer, no engine handle, no vtable. `seq` is the deterministic event
// ordinal/time.
struct CoOccurrenceEvent {
    std::uint64_t seq = 0;
    NodeId a_id = 0;  // projected-kind (A) node
    NodeId b_id = 0;  // opposite-kind (B) shared neighbour
};

// A surviving weighted edge of the one-mode projection between two same-kind (A)
// nodes. `lo < hi` canonically; `weight` = count of shared opposite-kind
// neighbours (>= the cutoff w_o). Pure value type.
struct ProjectedEdge {
    NodeId lo = 0;
    NodeId hi = 0;
    std::int64_t weight = 0;  // shared-neighbour count, >= weight_threshold
};

// PROVENANCE: an edge that existed in the unthresholded co-occurrence projection
// (weight >= 1) but was dropped because its weight fell below the cutoff w_o.
// Recorded so missingness is explicit, never silent.
struct DroppedEdge {
    NodeId lo = 0;
    NodeId hi = 0;
    std::int64_t weight = 0;  // 1 <= weight < weight_threshold
};

// Why a node is absent from the projected graph. Pure label.
enum class ExclusionReason : std::uint8_t {
    None = 0,
    Isolated = 1,          // shared no opposite-kind neighbour with any node
    SubThresholdOnly = 2,  // had only sub-threshold edges, all dropped by w_o
};

// PROVENANCE: a node excluded from the projected graph, with the reason. A node
// is excluded iff it has no surviving edge after the cutoff.
struct ExcludedNode {
    NodeId id = 0;
    ExclusionReason reason = ExclusionReason::None;
};

// Explicit missingness ledger. Populated ONLY when diagnostics are ON; it is
// observer-only state that never feeds back and is excluded from `replay_hash`.
struct ProvenanceRecord {
    std::vector<DroppedEdge> dropped_edges;    // sub-threshold edges, sorted
    std::vector<ExcludedNode> excluded_nodes;  // isolated / sub-threshold-only

    bool empty() const noexcept {
        return dropped_edges.empty() && excluded_nodes.empty();
    }
};

// The one-mode projection result: graph state plus the (optional) provenance
// ledger and the diagnostics-on/off-stable replay hash. Pure value type:
// copyable, holds no reference to any external (matching) state.
struct NetworkProjection {
    std::int64_t weight_threshold = 1;        // cutoff w_o (>= 1)
    std::vector<ProjectedEdge> edges;          // surviving edges, sorted (lo,hi)
    std::vector<NodeId> included_nodes;        // nodes with >= 1 surviving edge
    ProvenanceRecord provenance;               // missingness ledger (ON only)
    bool diagnostics_enabled = false;

    // Order-sensitive hash over the ORDERED events + GRAPH STATE only (threshold,
    // edges, included nodes). Deliberately EXCLUDES `provenance`, so it is equal
    // whether diagnostics are ON or OFF -- the diagnostics-on/off replay hash.
    ReplayHash replay_hash = 0;

    std::size_t edge_count() const noexcept { return edges.size(); }
    std::size_t node_count() const noexcept { return included_nodes.size(); }
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

// Count shared elements of two sorted, unique-valued neighbour vectors (the
// shared-neighbour edge weight). Linear merge; deterministic.
inline std::int64_t shared_neighbour_count(const std::vector<NodeId>& a,
                                           const std::vector<NodeId>& b) noexcept {
    std::int64_t count = 0;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            ++i;
        } else if (b[j] < a[i]) {
            ++j;
        } else {
            ++count;
            ++i;
            ++j;
        }
    }
    return count;
}

}  // namespace detail

// Deterministic, order-sensitive FNV-1a-64 hash over the ORDERED event stream
// and the resulting GRAPH STATE. Pure, read-only: folds each event's ORDINAL
// with its bytes so reordering the input changes the hash; folds the cutoff, the
// surviving weighted edges, and the included nodes; folds each collection's
// count so truncation/extension cannot collide with a prefix. It does NOT fold
// the provenance ledger -- that omission is what makes the hash equal under
// diagnostics-on and diagnostics-off.
inline ReplayHash network_graph_hash(const std::vector<CoOccurrenceEvent>& events,
                                     const NetworkProjection& p) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const CoOccurrenceEvent& e : events) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, e.seq);
        h = detail::fnv1a_u64(h, e.a_id);
        h = detail::fnv1a_u64(h, e.b_id);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the event count

    h = detail::fnv1a_i64(h, p.weight_threshold);

    ordinal = 0;
    for (const ProjectedEdge& edge : p.edges) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, edge.lo);
        h = detail::fnv1a_u64(h, edge.hi);
        h = detail::fnv1a_i64(h, edge.weight);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the edge count

    ordinal = 0;
    for (NodeId id : p.included_nodes) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, id);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the node count
    return h;
}

// Build the one-mode co-occurrence projection from a replayable event stream.
// PURE, READ-ONLY, OFFLINE: the events are taken by const reference and never
// modified; the result is a plain value. Construction semantics mirror the
// MOC's source-confirmed projection-provenance route:
//   (1) per projected-kind (A) node, collect the SET of opposite-kind (B)
//       neighbours (duplicates folded: a shared neighbour counts once);
//   (2) for each unordered pair of A-nodes, the candidate edge weight is the
//       count of shared B-neighbours (link iff weight >= 1);
//   (3) the cutoff `weight_threshold` (w_o) keeps edges with weight >= w_o and
//       drops sub-threshold edges plus any node left with no surviving edge;
//   (4) when `diagnostics_enabled`, every dropped edge and excluded node is
//       recorded in the provenance ledger (explicit missingness).
// `diagnostics_enabled`; only the provenance ledger differs. Deterministic
// ordering throughout (std::map / std::set keyed by NodeId); no hash-map
// iteration order leaks into the result. This cannot mutate, and is not consumed
// by, any matching state.
inline NetworkProjection project_network(const std::vector<CoOccurrenceEvent>& events,
                                         std::int64_t weight_threshold,
                                         bool diagnostics_enabled) {
    NetworkProjection result;
    if (weight_threshold < 1) weight_threshold = 1;  // a link needs >= 1 neighbour
    result.weight_threshold = weight_threshold;
    result.diagnostics_enabled = diagnostics_enabled;

    // (1) A-node -> sorted, unique B-neighbour set. std::map keeps A-node order
    // deterministic; sorted-insert keeps each neighbour set unique and ordered.
    std::map<NodeId, std::vector<NodeId>> neighbours;
    for (const CoOccurrenceEvent& e : events) {
        std::vector<NodeId>& nbrs = neighbours[e.a_id];
        auto pos = std::lower_bound(nbrs.begin(), nbrs.end(), e.b_id);
        if (pos == nbrs.end() || *pos != e.b_id) nbrs.insert(pos, e.b_id);
    }

    // (2)+(3) Pairwise shared-neighbour weights over sorted A-node keys, then the
    // cutoff. Track which nodes had any candidate edge and any surviving edge so
    // the provenance reason is exact.
    std::set<NodeId> had_candidate_edge;  // weight >= 1 with some node
    std::set<NodeId> had_surviving_edge;  // weight >= w_o with some node
    for (auto it_i = neighbours.begin(); it_i != neighbours.end(); ++it_i) {
        for (auto it_j = std::next(it_i); it_j != neighbours.end(); ++it_j) {
            const std::int64_t weight =
                detail::shared_neighbour_count(it_i->second, it_j->second);
            if (weight < 1) continue;  // no shared neighbour -> no link
            had_candidate_edge.insert(it_i->first);
            had_candidate_edge.insert(it_j->first);
            if (weight >= weight_threshold) {
                result.edges.push_back(ProjectedEdge{it_i->first, it_j->first, weight});
                had_surviving_edge.insert(it_i->first);
                had_surviving_edge.insert(it_j->first);
            } else if (diagnostics_enabled) {
                result.provenance.dropped_edges.push_back(
                    DroppedEdge{it_i->first, it_j->first, weight});
            }
        }
    }

    // (3)+(4) Included nodes are graph state (always); excluded nodes are
    // provenance (diagnostics ON only). A node with no surviving edge is
    // excluded -- Isolated if it never shared a neighbour, SubThresholdOnly if
    // every edge it had was dropped by the cutoff.
    for (const auto& kv : neighbours) {
        const NodeId id = kv.first;
        if (had_surviving_edge.count(id) != 0) {
            result.included_nodes.push_back(id);
        } else if (diagnostics_enabled) {
            const ExclusionReason reason = had_candidate_edge.count(id) != 0
                                               ? ExclusionReason::SubThresholdOnly
                                               : ExclusionReason::Isolated;
            result.provenance.excluded_nodes.push_back(ExcludedNode{id, reason});
        }
    }

    result.replay_hash = network_graph_hash(events, result);
    return result;
}

// Read-only convenience wrapper. Holds a const view of an externally owned event
// stream and exposes ONLY the projection; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The projection it returns can be read but never
// written back into matching.
class NetworkGraphProvenanceReplay {
public:
    explicit NetworkGraphProvenanceReplay(
        const std::vector<CoOccurrenceEvent>& events) noexcept
        : events_(&events) {}

    NetworkProjection project(std::int64_t weight_threshold,
                              bool diagnostics_enabled = true) const {
        return project_network(*events_, weight_threshold, diagnostics_enabled);
    }

    // The diagnostics-on/off-stable graph-state replay hash at this cutoff.
    ReplayHash hash(std::int64_t weight_threshold) const {
        return project_network(*events_, weight_threshold, true).replay_hash;
    }

    std::size_t size() const noexcept { return events_->size(); }
    bool empty() const noexcept { return events_->empty(); }

private:
    const std::vector<CoOccurrenceEvent>* events_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline projection / provenance
// types do NOT, which is the type-level proof that a graph metric or provenance
// value cannot be wired back into the kernel through these structs. A test
// supplies a POSITIVE CONTROL (a type that DOES expose such a member) to prove
// the trait actually discriminates and is not vacuously false.
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
//   * no `apply_to_matching` feedback member => the projection is one-way OUT;
//   * project() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<CoOccurrenceEvent>,
              "CoOccurrenceEvent must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ProjectedEdge>,
              "ProjectedEdge must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<DroppedEdge>,
              "DroppedEdge must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ExcludedNode>,
              "ExcludedNode must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<CoOccurrenceEvent>,
              "CoOccurrenceEvent must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<ProjectedEdge>,
              "ProjectedEdge must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<CoOccurrenceEvent>,
              "CoOccurrenceEvent must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ProjectedEdge>,
              "ProjectedEdge must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<NetworkProjection>,
              "NetworkProjection must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ProvenanceRecord>,
              "ProvenanceRecord must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<NetworkGraphProvenanceReplay>,
              "NetworkGraphProvenanceReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<NetworkProjection,
                          decltype(&NetworkGraphProvenanceReplay::project),
                          const NetworkGraphProvenanceReplay&, std::int64_t, bool>,
    "project() must be a read-only (const) diagnostic");

}  // namespace offline
