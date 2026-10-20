#include "qr/alloc_guard.hpp"
#include "sim/sim.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS] [STEPS]\n"
              << "\n"
              << "Options:\n"
              << "  -h, --help        Show this help message and exit\n"
              << "  --steps <N>       Number of simulation events to run (default: 10000, min: 100)\n"
              << "  --seed <S>        PRNG seed for QueueReactiveProcess (default: 0x9E3779B97F4A7C15)\n";
}

} // namespace

int main(int argc, char* argv[]) {
    int sim_events = 10000;
    std::uint64_t seed = sim::rng_constants::kDefaultSeed;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--steps") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --steps requires a numeric argument\n";
                print_usage(argv[0]);
                return 1;
            }
            sim_events = std::max(100, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--steps=", 8) == 0) {
            sim_events = std::max(100, std::atoi(argv[i] + 8));
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --seed requires a numeric argument\n";
                print_usage(argv[0]);
                return 1;
            }
            seed = std::strtoull(argv[++i], nullptr, 0);
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            seed = std::strtoull(argv[i] + 7, nullptr, 0);
        } else if (argv[i][0] != '-') {
            sim_events = std::max(100, std::atoi(argv[i]));
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Queue-Reactive Limit Order Book Simulator (C++20)\n";
    std::cout << "--------------------------------------------------------------------\n\n";

    const sim::Tick min_tick = 90;
    const sim::Tick max_tick = 110;
    sim::FlatLadderBook ladder(min_tick, max_tick, 8192);

    std::cout << "[1] Self-Trade Prevention (CancelPassive):\n";
    {
        sim::Order maker{};
        maker.agent = 10;
        maker.side = sim::Side::Sell;
        maker.price = 100;
        maker.qty = 50;
        maker.orig_qty = 50;
        maker.type = sim::OrderType::Limit;
        ladder.add(maker, 1000);

        std::cout << "  Resting ask:  50 lots @ 100 (agent 10)\n";

        sim::Order taker{};
        taker.agent = 10;
        taker.side = sim::Side::Buy;
        taker.price = 100;
        taker.qty = 30;
        taker.orig_qty = 30;
        taker.type = sim::OrderType::Limit;

        auto res = ladder.add(taker, 2000, sim::StpMode::CancelPassive);
        std::cout << "  Incoming buy: 30 lots @ 100 (agent 10, CancelPassive)\n";
        std::cout << "  Fills:        " << res.trades.size() << " (resting self-order cancelled)\n";
        std::cout << "  Ask vol @100: " << ladder.volume_at(sim::Side::Sell, 100) << "\n";
        std::cout << "  Bid vol @100: " << ladder.volume_at(sim::Side::Buy, 100) << "\n\n";
    }

    std::cout << "[2] L3 FIFO Queue Priority (queue_ahead):\n";
    {
        sim::FlatLadderBook book(90, 110);
        for (int i = 1; i <= 4; ++i) {
            sim::Order o{};
            o.agent = static_cast<sim::AgentId>(i);
            o.side = sim::Side::Buy;
            o.price = 100;
            o.qty = 20;
            o.orig_qty = 20;
            o.type = sim::OrderType::Limit;
            book.add(o, 1000 * i);
        }

        std::cout << "  4 buy orders of 20 lots queued at tick 100:\n";
        for (sim::OrderId id = 1; id <= 4; ++id) {
            auto ahead = book.queue_ahead(id);
            std::cout << "    order " << id << ": "
                      << (ahead ? std::to_string(*ahead) + " lots ahead" : "not live") << "\n";
        }

        book.cancel(2, 5000);
        std::cout << "  After cancelling order 2:\n";
        for (sim::OrderId id = 1; id <= 4; ++id) {
            auto ahead = book.queue_ahead(id);
            std::cout << "    order " << id << ": "
                      << (ahead ? std::to_string(*ahead) + " lots ahead" : "cancelled") << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "[3] Queue-Reactive Simulation (" << sim_events << " events via add_into, seed=" << seed << "):\n";
    {
        sim::FlatLadderBook sim_book(80, 120, static_cast<std::size_t>(sim_events * 2 + 64));
        sim_book.add_into(
            sim::Order{.qty = 100, .orig_qty = 100, .price = 99, .side = sim::Side::Buy, .type = sim::OrderType::Limit},
            100, nullptr, 0);
        sim_book.add_into(
            sim::Order{.qty = 100, .orig_qty = 100, .price = 101, .side = sim::Side::Sell, .type = sim::OrderType::Limit},
            100, nullptr, 0);

        sim::QueueReactiveProcess qrp({}, sim::HawkesMatrix::default_microstructure(), seed);
        sim::Trade trade_buf[32]{};
        std::uint64_t total_trades = 0;
        std::uint64_t total_volume = 0;

        const auto t0 = std::chrono::high_resolution_clock::now();
        sim::diag::AllocStats alloc_delta{};
        {
            sim::diag::ScopedAllocCounter alloc_counter;
            for (int i = 0; i < sim_events; ++i) {
                sim::Tick bb = 99, ba = 101;
                sim::Qty bv = 100, av = 100;
                sim_book.best_bid(bb, bv);
                sim_book.best_ask(ba, av);

                auto ev = qrp.next_event(bb, bv, ba, av);

                if (ev.order_type == sim::OrderType::Limit) {
                    sim::Order o{};
                    o.agent = 100;
                    o.side = ev.side;
                    o.price = ev.price;
                    o.qty = ev.qty;
                    o.orig_qty = ev.qty;
                    o.type = sim::OrderType::Limit;
                    const auto summary = sim_book.add_into(o, qrp.current_time(), trade_buf, 32);
                    total_trades += summary.trade_count;
                    total_volume += static_cast<std::uint64_t>(summary.filled);
                } else if (ev.order_type == sim::OrderType::Market) {
                    sim::Order o{};
                    o.agent = 200;
                    o.side = ev.side;
                    o.qty = ev.qty;
                    o.orig_qty = ev.qty;
                    o.type = sim::OrderType::Market;
                    const auto summary = sim_book.add_into(o, qrp.current_time(), trade_buf, 32);
                    total_trades += summary.trade_count;
                    total_volume += static_cast<std::uint64_t>(summary.filled);
                }
            }
            alloc_delta = alloc_counter.delta();
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto elapsed_us = std::max<std::int64_t>(
            1, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        const double throughput = (static_cast<double>(sim_events) / static_cast<double>(elapsed_us)) * 1e6;

        std::cout << "  Events processed:   " << sim_events << "\n";
        std::cout << "  Trades executed:    " << total_trades << "\n";
        std::cout << "  Volume matched:     " << total_volume << " lots\n";
        std::cout << "  Heap allocations:   " << alloc_delta.allocations
                  << " (" << alloc_delta.bytes << " bytes)\n";
        std::cout << "  Wall-clock elapsed: " << elapsed_us << " us ("
                  << std::fixed << std::setprecision(2) << (static_cast<double>(elapsed_us) / 1000.0) << " ms)\n";
        std::cout << "  Throughput:         " << std::fixed << std::setprecision(2)
                  << (throughput / 1e6) << " M events/s ("
                  << std::fixed << std::setprecision(1)
                  << (static_cast<double>(elapsed_us * 1000) / sim_events) << " ns/event)\n";
    }

    return 0;
}
