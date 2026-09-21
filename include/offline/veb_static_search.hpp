#pragma once

// =============================================================================
// Offline cache-oblivious van-Emde-Boas static search layout.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It has NO dependency on FlatLadderBook, the
//   matching kernel, or any `sim::` symbol. It is a self-contained, header-only
//   static (immutable, build-once) ordered index over sorted integer keys,
//   intended for offline replay-trace / analytics index experiments and the
//   benchmark/scan-parity harnesses that the MOC
//   `moc-offline-cache-oblivious-data-structures` gates on. It must NEVER be
//   linked into the matching hot path without measured evidence + a human gate.
//
// WHAT IT IS:
//   A perfect binary search tree over the sorted key array, stored in
//   van-Emde-Boas (cache-oblivious) layout. Search proceeds purely inside the
//   vEB array using precomputed child links, so the memory-access pattern is
//   recursively block-local independent of the (unknown) cache-line / page
//   size -- the defining property of a cache-oblivious search structure
//   (Frigo-Leiserson-Prokop-Ramachandran, FOCS 1999;
//    Prokop's static cache-oblivious search tree).
//
// SEMANTIC CONTRACT (the replay-equivalence / scan-parity guarantee):
//   For any query q,
//     lower_bound_rank(q) == (std::lower_bound(sorted.begin(), sorted.end(), q)
//                             - sorted.begin())
//     contains(q)         == std::binary_search(sorted.begin(), sorted.end(), q)
//   i.e. it returns byte-identical results to a flat sorted-array baseline.
//   Determinism: results depend only on the (sorted) key set and the query, not
//   on layout, padding, or platform.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace offline {

// Static cache-oblivious (van-Emde-Boas layout) ordered search index over a set
// of sorted keys of an integral type `Key`. Build-once, immutable, query-only.
template <typename Key>
class VebStaticSearch {
public:
    using key_type = Key;
    using size_type = std::size_t;

    VebStaticSearch() = default;

    // Build from an already-sorted (ascending) key vector. Duplicates are
    // permitted and preserve std::lower_bound semantics. O(n) extra space.
    static VebStaticSearch from_sorted(std::vector<Key> sorted_keys) {
        VebStaticSearch idx;
        idx.build(std::move(sorted_keys));
        return idx;
    }

    // Build from an arbitrary (unsorted) key vector; sorts internally.
    static VebStaticSearch from_unsorted(std::vector<Key> keys) {
        std::sort(keys.begin(), keys.end());
        return from_sorted(std::move(keys));
    }

    size_type size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }

    // Number of stored keys strictly less than `q`. Equivalent to
    // std::lower_bound position. Result is in [0, size()].
    size_type lower_bound_rank(Key q) const noexcept {
        // Default: every key is < q (q falls past the end).
        size_type candidate = n_;
        if (n_ == 0) return candidate;
        std::int64_t pos = 0; // start at vEB root
        while (pos >= 0) {
            const Node& node = nodes_[static_cast<size_type>(pos)];
            if (!(node.key < q)) {
                // node.key >= q : node is a lower-bound candidate; go left for a
                // possibly-earlier candidate.
                if (node.inorder < candidate) candidate = node.inorder;
                pos = node.left;
            } else {
                // node.key < q : the lower bound is to the right.
                pos = node.right;
            }
        }
        return candidate;
    }

    // True iff `q` is present. Equivalent to std::binary_search.
    bool contains(Key q) const noexcept {
        const size_type r = lower_bound_rank(q);
        return r < n_ && keys_[r] == q;
    }

    // Direct access to the underlying sorted keys (offline introspection).
    const std::vector<Key>& sorted_keys() const noexcept { return keys_; }

private:
    struct Node {
        Key key;             // node key (sentinel = max for padding slots)
        std::int64_t left;   // vEB array index of left child, or -1
        std::int64_t right;  // vEB array index of right child, or -1
        size_type inorder;   // in-order rank of this node among all slots
    };

    void build(std::vector<Key> sorted_keys) {
        keys_ = std::move(sorted_keys);
        n_ = keys_.size();
        nodes_.clear();
        if (n_ == 0) return;

        // 1) Perfect binary tree with M = 2^H - 1 >= n_ heap slots (1-based).
        size_type height = 1;
        size_type m = 1; // 2^height - 1
        while (m < n_) {
            ++height;
            m = (m << 1) + 1;
        }
        const size_type slots = m;

        // 2) In-order rank of every heap node (1-based heap index 1..slots).
        //    inorder_of_heap[h] in [0, slots).
        std::vector<size_type> inorder_of_heap(slots + 1, 0);
        assign_inorder(1, slots, inorder_of_heap);

        // 3) Keys by in-order rank: first n_ ranks hold the sorted keys, the
        //    remaining padding slots hold a +inf sentinel.
        const Key sentinel = std::numeric_limits<Key>::max();
        std::vector<Key> key_by_heap(slots + 1, sentinel);
        for (size_type h = 1; h <= slots; ++h) {
            const size_type rank = inorder_of_heap[h];
            key_by_heap[h] = (rank < n_) ? keys_[rank] : sentinel;
        }

        // 4) van-Emde-Boas layout order of the heap indices.
        std::vector<size_type> veb_order;
        veb_order.reserve(slots);
        layout(1, height, slots, veb_order);

        // 5) heap index -> vEB array position.
        std::vector<std::int64_t> pos_of_heap(slots + 1, -1);
        for (size_type p = 0; p < veb_order.size(); ++p) {
            pos_of_heap[veb_order[p]] = static_cast<std::int64_t>(p);
        }

        // 6) Materialize nodes in vEB order with child links resolved to vEB
        //    positions, so search never leaves the vEB array.
        nodes_.resize(veb_order.size());
        for (size_type p = 0; p < veb_order.size(); ++p) {
            const size_type h = veb_order[p];
            Node& node = nodes_[p];
            node.key = key_by_heap[h];
            node.inorder = inorder_of_heap[h];
            const size_type lh = h << 1;        // left child heap index
            const size_type rh = lh + 1;        // right child heap index
            node.left = (lh <= slots) ? pos_of_heap[lh] : -1;
            node.right = (rh <= slots) ? pos_of_heap[rh] : -1;
        }
    }

    // Assign in-order ranks (0-based) to a complete binary tree rooted at heap
    // index `root` with `slots` total nodes. Iterative to avoid deep recursion.
    static void assign_inorder(size_type root, size_type slots,
                               std::vector<size_type>& out) {
        size_type counter = 0;
        std::vector<size_type> stack;
        size_type cur = root;
        while (!stack.empty() || cur <= slots) {
            while (cur <= slots) {
                stack.push_back(cur);
                cur = cur << 1; // go left
            }
            cur = stack.back();
            stack.pop_back();
            out[cur] = counter++;
            cur = (cur << 1) + 1; // go right
        }
    }

    // Recursive van-Emde-Boas layout of a perfect subtree of `tree_height`
    // levels rooted at heap index `root`. Appends heap indices to `order` in
    // vEB order. `slots` bounds valid heap indices.
    static void layout(size_type root, size_type tree_height, size_type slots,
                       std::vector<size_type>& order) {
        if (tree_height == 0 || root > slots) return;
        if (tree_height == 1) {
            order.push_back(root);
            return;
        }
        const size_type top_h = tree_height / 2;
        const size_type bottom_h = tree_height - top_h;
        // Lay out the top subtree (levels [0, top_h)).
        layout(root, top_h, slots, order);
        // Then each of the 2^top_h bottom subtrees, left to right. Their roots
        // are the nodes at relative depth top_h: root*2^top_h .. +2^top_h-1.
        const size_type num_bottom = static_cast<size_type>(1) << top_h;
        const size_type first = root << top_h;
        for (size_type j = 0; j < num_bottom; ++j) {
            layout(first + j, bottom_h, slots, order);
        }
    }

    std::vector<Key> keys_;   // sorted keys (offline introspection + contains)
    std::vector<Node> nodes_; // vEB-laid-out search nodes
    size_type n_ = 0;
};

} // namespace offline
