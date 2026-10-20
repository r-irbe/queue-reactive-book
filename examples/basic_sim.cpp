#include "qr/alloc_guard.hpp"
#include "sim/sim.hpp"

#include <iostream>

int main() {
    sim::FlatLadderBook book(90, 110, 512);

    sim::Order bid{};
    bid.agent = 1;
    bid.side = sim::Side::Buy;
    bid.price = 99;
    bid.qty = 50;
    bid.orig_qty = 50;
    bid.type = sim::OrderType::Limit;
    book.add(bid, 100);

    sim::Order ask{};
    ask.agent = 2;
    ask.side = sim::Side::Sell;
    ask.price = 101;
    ask.qty = 50;
    ask.orig_qty = 50;
    ask.type = sim::OrderType::Limit;
    book.add(ask, 200);

    sim::QueueReactiveProcess qrp({}, sim::HawkesMatrix::default_microstructure(), 42ULL);
    sim::Trade trades[16]{};
    std::uint64_t total_trades = 0;

    sim::diag::AllocStats allocs{};
    {
        qr::alloc_guard::ScopedAllocCounter guard;
        for (int i = 0; i < 1000; ++i) {
            sim::Tick bb = 99, ba = 101;
            sim::Qty bv = 50, av = 50;
            book.best_bid(bb, bv);
            book.best_ask(ba, av);

            const auto ev = qrp.next_event(bb, bv, ba, av);
            if (ev.order_type == sim::OrderType::Limit || ev.order_type == sim::OrderType::Market) {
                sim::Order o{};
                o.agent = (ev.order_type == sim::OrderType::Limit) ? 10 : 20;
                o.side = ev.side;
                o.price = ev.price;
                o.qty = ev.qty;
                o.orig_qty = ev.qty;
                o.type = ev.order_type;
                const auto res = book.add_into(o, qrp.current_time(), trades, 16);
                total_trades += res.trade_count;
            }
        }
        allocs = guard.delta();
    }

    std::cout << "Basic Queue-Reactive Simulation complete: 1000 events, "
              << total_trades << " trades, "
              << allocs.allocations << " steady-state heap allocations.\n";
    return 0;
}
