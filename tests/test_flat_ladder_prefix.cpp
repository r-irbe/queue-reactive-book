// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// Differential verification for the O(1) arrival-cumulus queue_ahead
// watermark and O(log k) per-level Fenwick interior-cancel tree:
//   - randomized add / cancel / cross fuzz against a shadow L3 model,
//   - interior-cancel Fenwick tree activation and maintenance,
//   - rollback via ReversibleSubgameScope with post-rollback rebuild plus
//     post-rollback add continuity.

#include "catch2/catch_test_macros.hpp"
#define QRLOB_NO_GLOBAL_NEW_OVERRIDE 1
#include "qr/alloc_guard.hpp"

#include "sim/flat_ladder_book.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace {

struct ShadowOrder {
    sim::Side side;
    sim::Tick price;
    sim::Qty qty;
};

// Shadow L3: per price level, FIFO list of (id, qty) mirroring book semantics.
struct ShadowBook {
    std::map<sim::Tick, std::vector<std::pair<sim::OrderId, sim::Qty>>> bids;
    std::map<sim::Tick, std::vector<std::pair<sim::OrderId, sim::Qty>>> asks;
    std::map<sim::OrderId, ShadowOrder> live;

    static auto& level_for(std::map<sim::Tick, std::vector<std::pair<sim::OrderId, sim::Qty>>>& m,
                           sim::Tick p) {
        return m[p];
    }

    sim::Qty queue_ahead(sim::OrderId id) const {
        const auto it = live.find(id);
        if (it == live.end()) return 0;
        const ShadowOrder& o = it->second;
        const auto& m = (o.side == sim::Side::Buy) ? bids : asks;
        const auto lit = m.find(o.price);
        if (lit == m.end()) return 0;
        sim::Qty ahead = 0;
        for (const auto& [oid, oq] : lit->second) {
            if (oid == id) break;
            ahead += oq;
        }
        return ahead;
    }

    void add(sim::OrderId id, sim::Side s, sim::Tick p, sim::Qty q) {
        live[id] = ShadowOrder{s, p, q};
        auto& lvl = level_for((s == sim::Side::Buy) ? bids : asks, p);
        lvl.emplace_back(id, q);
    }

    // Reduces qty (partial fill) or removes (full fill/cancel). Returns filled.
    sim::Qty consume_front(sim::Side maker_side, sim::Qty take) {
        auto& m = (maker_side == sim::Side::Buy) ? bids : asks;
        sim::Qty filled = 0;
        // Best maker price first: asks ascending (buy taker lifts asks),
        // bids DESCENDING (sell taker hits bids). Key on which book we walk,
        // not on the maker side variable.
        const bool is_ask_book = (maker_side == sim::Side::Sell);
        while (take > 0 && !m.empty()) {
            auto lit = is_ask_book ? m.begin() : std::prev(m.end());
            auto& vec = lit->second;
            while (take > 0 && !vec.empty()) {
                auto& [oid, oq] = vec.front();
                const sim::Qty dec = std::min(take, oq);
                oq -= dec;
                take -= dec;
                filled += dec;
                live[oid].qty -= dec;
                if (oq == 0) {
                    live.erase(oid);
                    vec.erase(vec.begin());
                }
            }
            if (vec.empty()) m.erase(lit);
        }
        return filled;
    }

    bool cancel(sim::OrderId id) {
        const auto it = live.find(id);
        if (it == live.end()) return false;
        auto& m = (it->second.side == sim::Side::Buy) ? bids : asks;
        auto lit = m.find(it->second.price);
        if (lit == m.end()) return false;
        auto& vec = lit->second;
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (vec[i].first == id) {
                vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        if (vec.empty()) m.erase(lit);
        live.erase(it);
        return true;
    }
};

sim::Order make_passive(sim::AgentId agent, sim::OrderId id, sim::Side s, sim::Tick p, sim::Qty q) {
    sim::Order o{};
    o.id = id;
    o.agent = agent;
    o.qty = q;
    o.orig_qty = q;
    o.price = p;
    o.side = s;
    o.type = sim::OrderType::Limit;
    return o;
}

}  // namespace

TEST_CASE("queue_ahead watermark matches shadow L3 under randomized add/cancel/cross",
          "[flat-ladder][queue-ahead][fuzz]") {
    sim::FlatLadderBook book(90, 110, 1024);
    ShadowBook shadow;
    std::mt19937_64 rng(0xC0FFEE123456789ull);
    sim::OrderId next_id = 1;
    sim::TimeNs ts = 1;

    for (int step = 0; step < 4000; ++step) {
        const int action = static_cast<int>(rng() % 100);
        if (action < 55) {
            // Passive add at a random level near the mid. The book assigns ids.
            const sim::Side s = (rng() & 1) ? sim::Side::Buy : sim::Side::Sell;
            const sim::Tick p = (s == sim::Side::Buy)
                                    ? static_cast<sim::Tick>(100 - static_cast<sim::Tick>(rng() % 5))
                                    : static_cast<sim::Tick>(100 + static_cast<sim::Tick>(rng() % 5));
            const sim::Qty q = static_cast<sim::Qty>(1 + rng() % 20);
            sim::Order o = make_passive(777, 0, s, p, q);
            const auto res = book.add(o, ts++);
            // Mirror any taker fills the crossing limit caused, then rest the
            // remainder (the book assigns ids to every accepted order).
            if (res.filled > 0) {
                shadow.consume_front(sim::opposite(s), res.filled);
            }
            if (res.remaining > 0 && res.id != 0) {
                shadow.add(res.id, s, p, res.remaining);
            }
        } else if (action < 80) {
            // Cancel a random live order (interior cancels included).
            if (!shadow.live.empty()) {
                auto it = shadow.live.begin();
                std::advance(it, static_cast<std::ptrdiff_t>(rng() % shadow.live.size()));
                const sim::OrderId victim = it->first;
                const bool ok = book.cancel(victim, ts++);
                if (ok) {
                    shadow.cancel(victim);
                }
            }
        } else {
            // Cross: aggressive marketable order against the opposite best.
            const sim::Side taker = (rng() & 1) ? sim::Side::Buy : sim::Side::Sell;
            const sim::Qty take = static_cast<sim::Qty>(1 + rng() % 30);
            sim::Order mkt{};
            mkt.id = next_id++;
            mkt.agent = 888;
            mkt.qty = take;
            mkt.orig_qty = take;
            mkt.side = taker;
            mkt.type = sim::OrderType::Market;
            const auto cres = book.add(mkt, ts++);
            shadow.consume_front(taker == sim::Side::Buy ? sim::Side::Sell : sim::Side::Buy, take);
        }

        // Differential check on a sample of live orders every step.
        for (int k = 0; k < 8 && !shadow.live.empty(); ++k) {
            auto it = shadow.live.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(rng() % shadow.live.size()));
            const auto got = book.queue_ahead(it->first);
            const sim::Qty want = shadow.queue_ahead(it->first);
            if (!got.has_value() || *got != want) {
                std::printf("MISMATCH step=%d id=%llu live_in_shadow=1 book_has_ref=%d\n",
                            step, (unsigned long long)it->first,
                            book.queue_ahead(it->first).has_value() ? 1 : 0);
                std::printf("  want=%d got=%s ts=%llu\n", (int)want,
                            got.has_value() ? std::to_string(*got).c_str() : "nullopt",
                            (unsigned long long)ts);
                std::fflush(stdout);
                REQUIRE(false);
            }
            REQUIRE(got.has_value());
            REQUIRE(*got == want);
        }
    }
    REQUIRE(book.check_invariants_report().ok);
}

TEST_CASE("queue_ahead survives rollback and post-rollback adds", "[flat-ladder][queue-ahead][rollback]") {
    sim::FlatLadderBook book(90, 110, 256);
    sim::TimeNs ts = 1;

    // Pre-scope state: 3 x 10 at 100 (buy), ids 1..3.
    for (sim::OrderId id = 1; id <= 3; ++id) {
        REQUIRE(book.add(make_passive(1, id, sim::Side::Buy, 100, 10), ts++).remaining == 10);
    }
    REQUIRE(book.queue_ahead(2) == sim::Qty{10});
    REQUIRE(book.queue_ahead(3) == sim::Qty{20});

    sim::Qty ahead_after_scope = 0;
    sim::Qty ahead_post_rollback_add = 0;
    {
        sim::ReversibleSubgameScope scope(book);
        // Partial fill of the head (10 -> 4) and an interior cancel of id 2.
        REQUIRE(scope.execute(sim::Side::Sell, 6, ts++).filled == 6);
        REQUIRE(scope.cancel(2, ts++));
        // Scope-internal queue_ahead reflects the partial head.
        auto a3 = book.queue_ahead(3);
        REQUIRE(a3.has_value());
        REQUIRE(*a3 == sim::Qty{4});
        ahead_after_scope = *a3;
        // Rollback on scope destruction (no commit).
    }

    // Post-rollback: full restore (head qty back to 10, id 2 alive).
    REQUIRE(book.queue_ahead(2) == sim::Qty{10});
    REQUIRE(book.queue_ahead(3) == sim::Qty{20});
    (void)ahead_after_scope;

    // Continuity: adds after a rollback must extend the rebuilt cumulus.
    REQUIRE(book.add(make_passive(1, 4, sim::Side::Buy, 100, 5), ts++).remaining == 5);
    auto a4 = book.queue_ahead(4);
    REQUIRE(a4.has_value());
    ahead_post_rollback_add = *a4;
    REQUIRE(ahead_post_rollback_add == sim::Qty{30});

    // Interior cancel permanently demotes this level to the exact walk; the
    // walk must agree with the head-cumulus values on fresh heads.
    REQUIRE(book.cancel(1, ts++));
    REQUIRE(book.queue_ahead(2) == sim::Qty{0});
    REQUIRE(book.queue_ahead(3) == sim::Qty{10});
    REQUIRE(book.queue_ahead(4) == sim::Qty{20});
    REQUIRE(book.check_invariants_report().ok);
}

TEST_CASE("queue_ahead Fenwick tree handles deep 1024-order level with interior cancels and partial fills",
          "[flat-ladder][queue-ahead][fenwick]") {
    sim::FlatLadderBook book(90, 110, 4096);
    ShadowBook shadow;

    constexpr int kDeepOrders = 1024;
    for (int i = 0; i < kDeepOrders; ++i) {
        const sim::Qty q = static_cast<sim::Qty>(10 + (i % 19));
        sim::Order o = make_passive(static_cast<sim::AgentId>((i % 7) + 1), 0, sim::Side::Buy, 100, q);
        const auto m = book.add(o, static_cast<sim::TimeNs>(i + 1));
        REQUIRE(m.id == static_cast<sim::OrderId>(i + 1));
        shadow.add(m.id, o.side, o.price, m.remaining);
    }

    // Cancel 150 interior orders across the 1024-order level
    for (int step = 0; step < 150; ++step) {
        const sim::OrderId cid = static_cast<sim::OrderId>(2 + ((step * 6 + 3) % (kDeepOrders - 2)));
        if (book.cancel(cid, static_cast<sim::TimeNs>(2000 + step))) {
            shadow.cancel(cid);
        }
    }

    // Partially consume from the head via market sell orders
    for (int mstep = 0; mstep < 12; ++mstep) {
        sim::Order mkt{};
        mkt.side = sim::Side::Sell;
        mkt.type = sim::OrderType::Market;
        mkt.qty = 45;
        mkt.orig_qty = 45;
        mkt.agent = 99;
        book.add(mkt, static_cast<sim::TimeNs>(3000 + mstep));
        shadow.consume_front(sim::Side::Buy, 45);
    }

    // Add 64 more orders at the tail while Fenwick tree is active
    for (int i = 0; i < 64; ++i) {
        const sim::Qty q = static_cast<sim::Qty>(15 + (i % 11));
        sim::Order o = make_passive(5, 0, sim::Side::Buy, 100, q);
        const auto m = book.add(o, static_cast<sim::TimeNs>(4000 + i));
        shadow.add(m.id, o.side, o.price, m.remaining);
    }

    // Cancel a few more interior orders among both old and newly added orders
    for (sim::OrderId cid : {sim::OrderId{500}, sim::OrderId{900}, sim::OrderId{1050}}) {
        if (book.cancel(cid, 5000)) {
            shadow.cancel(cid);
        }
    }

    // Verify every single live order matches the shadow book queue_ahead
    for (const auto& kv : shadow.live) {
        const auto got = book.queue_ahead(kv.first);
        REQUIRE(got.has_value());
        REQUIRE(*got == shadow.queue_ahead(kv.first));
    }
}

TEST_CASE("queue_ahead Fenwick tree across 3-level nested ReversibleSubgameScope rollbacks and partial commits",
          "[flat-ladder][queue-ahead][fenwick][rollback]") {
    sim::FlatLadderBook book(90, 110, 512);
    ShadowBook shadow;
    sim::TimeNs ts = 1;

    // Seed 32 resting buy orders on price level 100
    for (int i = 0; i < 32; ++i) {
        const sim::Qty q = static_cast<sim::Qty>(10 + (i % 7));
        sim::Order o = make_passive(1, 0, sim::Side::Buy, 100, q);
        const auto m = book.add(o, ts++);
        shadow.add(m.id, o.side, o.price, m.remaining);
    }

    // Perform an interior cancel so the level activates its Fenwick tree before scopes
    REQUIRE(book.cancel(5, ts++));
    shadow.cancel(5);

    for (const auto& kv : shadow.live) {
        REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
    }

    ShadowBook snap_outer = shadow;
    {
        sim::ReversibleSubgameScope scope1(book);
        // Partial fill at head + interior cancel inside scope1
        REQUIRE(scope1.execute(sim::Side::Sell, 6, ts++).filled == 6);
        shadow.consume_front(sim::Side::Buy, 6);
        REQUIRE(scope1.cancel(10, ts++));
        shadow.cancel(10);

        for (const auto& kv : shadow.live) {
            REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
        }

        ShadowBook snap_mid = shadow;
        {
            sim::ReversibleSubgameScope scope2(book);
            // Add new order at tail + cancel interior order 15 inside scope2
            const auto m_new = book.add(make_passive(2, 0, sim::Side::Buy, 100, 25), ts++);
            shadow.add(m_new.id, sim::Side::Buy, 100, m_new.remaining);
            REQUIRE(scope2.cancel(15, ts++));
            shadow.cancel(15);

            for (const auto& kv : shadow.live) {
                REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
            }

            {
                sim::ReversibleSubgameScope scope3(book);
                // Consume across head + cancel interior order 20 inside scope3
                REQUIRE(scope3.execute(sim::Side::Sell, 18, ts++).filled == 18);
                shadow.consume_front(sim::Side::Buy, 18);
                REQUIRE(scope3.cancel(20, ts++));
                shadow.cancel(20);
                // Commit scope3 into scope2!
                scope3.commit();
            }

            // Verify scope3 committed state inside scope2
            for (const auto& kv : shadow.live) {
                REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
            }
            // Let scope2 roll back (reverting both scope2 and committed scope3!)
        }
        shadow = snap_mid;

        // Verify state inside scope1 after scope2 rollback
        for (const auto& kv : shadow.live) {
            REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
        }

        // Perform one more interior cancel and tail add inside scope1 after scope2 rollback
        REQUIRE(scope1.cancel(12, ts++));
        shadow.cancel(12);
        const auto m_post = book.add(make_passive(3, 0, sim::Side::Buy, 100, 19), ts++);
        shadow.add(m_post.id, sim::Side::Buy, 100, m_post.remaining);

        for (const auto& kv : shadow.live) {
            REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
        }
        // Let scope1 roll back to snap_outer!
    }
    shadow = snap_outer;

    // Verify full restoration to outer pre-scope state
    for (const auto& kv : shadow.live) {
        REQUIRE(book.queue_ahead(kv.first) == shadow.queue_ahead(kv.first));
    }
    REQUIRE(book.check_invariants_report().ok);
}
