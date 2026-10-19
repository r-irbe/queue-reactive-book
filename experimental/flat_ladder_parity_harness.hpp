// Copyright (c) 2026 Open-WebUI & Project Q Systems Engineering
// SPDX-License-Identifier: Apache-2.0
//
// Experimental byte-frozen flat_ladder_book parity harness and cache-oblivious
// vEB static search verifier absorbed from /home/radu/code/q (commit f190d0a4).
// Isolated from the -fno-exceptions matching core behind QRLOB_BUILD_EXPERIMENTAL.

#pragma once

#include "sim/event_trace.hpp"
#include "sim/flat_ladder_book.hpp"
#include "sim/lob.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace sim {
namespace experimental {

// Static cache-oblivious (van-Emde-Boas layout) ordered search index absorbed
// from Project Q (offline/veb_static_search.hpp).
template <typename Key>
class VebStaticSearch {
public:
    using key_type = Key;
    using size_type = std::size_t;

    VebStaticSearch() = default;

    static VebStaticSearch from_sorted(std::vector<Key> sorted_keys) {
        VebStaticSearch idx;
        idx.build(std::move(sorted_keys));
        return idx;
    }

    static VebStaticSearch from_unsorted(std::vector<Key> keys) {
        std::sort(keys.begin(), keys.end());
        return from_sorted(std::move(keys));
    }

    size_type size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }

    size_type lower_bound_rank(Key q) const noexcept {
        size_type candidate = n_;
        if (n_ == 0) return candidate;
        std::int64_t pos = 0;
        while (pos >= 0) {
            const Node& node = nodes_[static_cast<size_type>(pos)];
            if (!(node.key < q)) {
                if (node.inorder < candidate) candidate = node.inorder;
                pos = node.left;
            } else {
                pos = node.right;
            }
        }
        return candidate;
    }

    bool contains(Key q) const noexcept {
        const size_type r = lower_bound_rank(q);
        return r < n_ && keys_[r] == q;
    }

    const std::vector<Key>& sorted_keys() const noexcept { return keys_; }

private:
    struct Node {
        Key key;
        std::int64_t left;
        std::int64_t right;
        size_type inorder;
    };

    void build(std::vector<Key> sorted_keys) {
        keys_ = std::move(sorted_keys);
        n_ = keys_.size();
        nodes_.clear();
        if (n_ == 0) return;

        size_type height = 1;
        size_type m = 1;
        while (m < n_) {
            ++height;
            m = (m << 1) + 1;
        }
        const size_type slots = m;

        std::vector<size_type> inorder_of_heap(slots + 1, 0);
        assign_inorder(1, slots, inorder_of_heap);

        const Key sentinel = std::numeric_limits<Key>::max();
        std::vector<Key> key_by_heap(slots + 1, sentinel);
        for (size_type h = 1; h <= slots; ++h) {
            const size_type rank = inorder_of_heap[h];
            key_by_heap[h] = (rank < n_) ? keys_[rank] : sentinel;
        }

        std::vector<size_type> veb_order;
        veb_order.reserve(slots);
        layout(1, height, slots, veb_order);

        std::vector<std::int64_t> pos_of_heap(slots + 1, -1);
        for (size_type p = 0; p < veb_order.size(); ++p) {
            pos_of_heap[veb_order[p]] = static_cast<std::int64_t>(p);
        }

        nodes_.resize(veb_order.size());
        for (size_type p = 0; p < veb_order.size(); ++p) {
            const size_type h = veb_order[p];
            Node& node = nodes_[p];
            node.key = key_by_heap[h];
            node.inorder = inorder_of_heap[h];
            const size_type lh = h << 1;
            const size_type rh = lh + 1;
            node.left = (lh <= slots) ? pos_of_heap[lh] : -1;
            node.right = (rh <= slots) ? pos_of_heap[rh] : -1;
        }
    }

    static void assign_inorder(size_type root, size_type slots,
                               std::vector<size_type>& out) {
        size_type counter = 0;
        std::vector<size_type> stack;
        size_type cur = root;
        while (!stack.empty() || cur <= slots) {
            while (cur <= slots) {
                stack.push_back(cur);
                cur = cur << 1;
            }
            cur = stack.back();
            stack.pop_back();
            out[cur] = counter++;
            cur = (cur << 1) + 1;
        }
    }

    static void layout(size_type root, size_type tree_height, size_type slots,
                       std::vector<size_type>& order) {
        if (tree_height == 0 || root > slots) return;
        if (tree_height == 1) {
            order.push_back(root);
            return;
        }
        const size_type top_h = tree_height / 2;
        const size_type bottom_h = tree_height - top_h;
        layout(root, top_h, slots, order);
        const size_type num_bottom = static_cast<size_type>(1) << top_h;
        const size_type first = root << top_h;
        for (size_type j = 0; j < num_bottom; ++j) {
            layout(first + j, bottom_h, slots, order);
        }
    }

    std::vector<Key> keys_;
    std::vector<Node> nodes_;
    size_type n_ = 0;
};

// Replay parity summary comparing sim::FlatLadderBook against sim::LimitOrderBook
// across a deterministic trace of SubmitLimit / SubmitMarket / Cancel events.
struct FlatLadderParityReport {
    std::size_t events_replayed = 0;
    std::size_t trades_matched = 0;
    std::size_t mismatches = 0;
    std::uint64_t flat_digest = 0;
    std::uint64_t ref_digest = 0;
    bool parity_ok = false;
};

class FlatLadderParityHarness {
public:
    static constexpr std::uint64_t kFnvBasis = 14695981039346656037ULL;
    static constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

    static std::uint64_t fnv_mix(std::uint64_t h, std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) {
            h ^= (v & 0xFFULL);
            h *= kFnvPrime;
            v >>= 8;
        }
        return h;
    }

    static std::uint64_t hash_exec_and_bbo(std::uint64_t h,
                                           const ExecutionResult& res,
                                           bool has_bb, Tick bb_p, Qty bb_v,
                                           bool has_ba, Tick ba_p, Qty ba_v,
                                           std::size_t order_count) noexcept {
        h = fnv_mix(h, static_cast<std::uint64_t>(res.filled));
        h = fnv_mix(h, static_cast<std::uint64_t>(res.remaining));
        h = fnv_mix(h, static_cast<std::uint64_t>(res.trades.size()));
        for (const Trade& tr : res.trades) {
            h = fnv_mix(h, static_cast<std::uint64_t>(tr.maker_id));
            h = fnv_mix(h, static_cast<std::uint64_t>(tr.taker_id));
            h = fnv_mix(h, static_cast<std::uint64_t>(tr.price));
            h = fnv_mix(h, static_cast<std::uint64_t>(tr.qty));
            h = fnv_mix(h, static_cast<std::uint64_t>(tr.ts));
        }
        h = fnv_mix(h, has_bb ? 1ULL : 0ULL);
        if (has_bb) {
            h = fnv_mix(h, static_cast<std::uint64_t>(bb_p));
            h = fnv_mix(h, static_cast<std::uint64_t>(bb_v));
        }
        h = fnv_mix(h, has_ba ? 1ULL : 0ULL);
        if (has_ba) {
            h = fnv_mix(h, static_cast<std::uint64_t>(ba_p));
            h = fnv_mix(h, static_cast<std::uint64_t>(ba_v));
        }
        h = fnv_mix(h, static_cast<std::uint64_t>(order_count));
        return h;
    }

    static FlatLadderParityReport verify_trace(Tick min_tick,
                                               Tick max_tick,
                                               const std::vector<TraceEvent>& trace) {
        FlatLadderParityReport rep{};
        FlatLadderBook flat(min_tick, max_tick, trace.size() + 64);
        LimitOrderBook ref_lob(trace.size() + 64);

        rep.flat_digest = kFnvBasis;
        rep.ref_digest = kFnvBasis;

        for (const TraceEvent& ev : trace) {
            ++rep.events_replayed;
            ExecutionResult flat_res{};
            ExecutionResult ref_res{};
            bool flat_cancel = false;
            bool ref_cancel = false;

            if (ev.kind == TraceEventKind::SubmitLimit ||
                ev.kind == TraceEventKind::SubmitMarket) {
                Order o{};
                o.agent = ev.agent;
                o.side = ev.side;
                o.price = ev.price;
                o.qty = ev.qty;
                o.orig_qty = ev.qty;
                o.type = (ev.kind == TraceEventKind::SubmitMarket)
                             ? OrderType::Market
                             : OrderType::Limit;
                ref_lob.set_time(ev.ts);
                flat_res = flat.add(o, ev.ts);
                ref_res = ref_lob.add(o);
                rep.trades_matched += flat_res.trades.size();
            } else if (ev.kind == TraceEventKind::Cancel) {
                ref_lob.set_time(ev.ts);
                flat_cancel = flat.cancel(ev.order_id, ev.ts);
                ref_cancel = ref_lob.cancel(ev.order_id);
            }

            Tick flat_bb_p = 0, flat_ba_p = 0;
            Qty flat_bb_v = 0, flat_ba_v = 0;
            const bool flat_has_bb = flat.best_bid(flat_bb_p, flat_bb_v);
            const bool flat_has_ba = flat.best_ask(flat_ba_p, flat_ba_v);

            const auto ref_bb = ref_lob.best_bid();
            const auto ref_ba = ref_lob.best_ask();
            const bool ref_has_bb = ref_bb.has_value();
            const bool ref_has_ba = ref_ba.has_value();
            const Tick ref_bb_p = ref_has_bb ? *ref_bb : 0;
            const Qty ref_bb_v = ref_has_bb ? ref_lob.volume_at(Side::Buy, *ref_bb) : 0;
            const Tick ref_ba_p = ref_has_ba ? *ref_ba : 0;
            const Qty ref_ba_v = ref_has_ba ? ref_lob.volume_at(Side::Sell, *ref_ba) : 0;

            rep.flat_digest = hash_exec_and_bbo(
                fnv_mix(rep.flat_digest, flat_cancel ? 1ULL : 0ULL),
                flat_res, flat_has_bb, flat_bb_p, flat_bb_v,
                flat_has_ba, flat_ba_p, flat_ba_v, flat.order_count());
            rep.ref_digest = hash_exec_and_bbo(
                fnv_mix(rep.ref_digest, ref_cancel ? 1ULL : 0ULL),
                ref_res, ref_has_bb, ref_bb_p, ref_bb_v,
                ref_has_ba, ref_ba_p, ref_ba_v, ref_lob.order_count());

            if (flat_cancel != ref_cancel ||
                flat_res.filled != ref_res.filled ||
                flat_res.remaining != ref_res.remaining ||
                flat_res.trades.size() != ref_res.trades.size() ||
                flat_has_bb != ref_has_bb ||
                (flat_has_bb && (flat_bb_p != ref_bb_p || flat_bb_v != ref_bb_v)) ||
                flat_has_ba != ref_has_ba ||
                (flat_has_ba && (flat_ba_p != ref_ba_p || flat_ba_v != ref_ba_v)) ||
                flat.order_count() != ref_lob.order_count()) {
                ++rep.mismatches;
            }
        }

        if (!flat.check_invariants_report().ok) {
            ++rep.mismatches;
        }
        rep.parity_ok = (rep.mismatches == 0 && rep.flat_digest == rep.ref_digest);
        return rep;
    }
};

} // namespace experimental
} // namespace sim
