// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#include "qr/alloc_guard.hpp"
#include "sim/sim.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS] [STEPS]\n"
              << "\n"
              << "Options:\n"
              << "  -h, --help                Show this help message and exit\n"
              << "  --mode <MODE>             Execution mode:\n"
              << "                              demo          (default interactive walkthrough)\n"
              << "                              qr-stylized   (5-level HLR IIb/III + Hawkes + Ogata trajectory)\n"
              << "                              latency-sweep (3-stage latency adverse selection sweep)\n"
              << "                              mm-tournament (AS vs GLFT vs CJP vs Passive vs Sniper)\n"
              << "                              exec-impact   (TWAP vs VWAP vs POV execution & sqrt impact)\n"
              << "                              parity        (FlatLadderBook vs LimitOrderBook FNV-1a parity)\n"
              << "                              calibrate     (Offline Hawkes & VPIN calibration on LOBSTER CSV)\n"
              << "  --steps <N>, --events <N> Number of simulation events to run (default: 10000, min: 100)\n"
              << "  --seed <S>                PRNG seed for QueueReactiveProcess (default: 0x9E3779B97F4A7C15)\n"
              << "  --dataset <PATH>          Input CSV dataset path for calibrate/qr-stylized modes\n"
              << "  --export-dir <DIR>        Optional directory to export CSV/JSON experiment artifacts\n";
}

const char* agent_kind_name(sim::StrategicAgentKind k) noexcept {
    switch (k) {
        case sim::StrategicAgentKind::AvellanedaStoikov: return "AvellanedaStoikov";
        case sim::StrategicAgentKind::GlftHeuristic:     return "GLFT_2013";
        case sim::StrategicAgentKind::CjpHeuristic:      return "CJP_2015_OFI";
        case sim::StrategicAgentKind::PassiveMaker:      return "PassiveMaker";
        case sim::StrategicAgentKind::LatencyTaker:      return "LatencyTaker";
        case sim::StrategicAgentKind::TwapExecution:     return "TWAP";
        case sim::StrategicAgentKind::VwapExecution:     return "VWAP";
        case sim::StrategicAgentKind::PovExecution:      return "POV";
    }
    return "Unknown";
}

int run_demo_mode(int sim_events, std::uint64_t seed) {
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

    std::cout << "[2] L3 FIFO Queue Priority (queue_ahead & peek_head/tail):\n";
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

        std::cout << "  4 buy orders of 20 lots queued at tick 100 (head="
                  << book.peek_head_order_id(sim::Side::Buy, 100)
                  << ", tail=" << book.peek_tail_order_id(sim::Side::Buy, 100) << "):\n";
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
        std::uint64_t total_cancels = 0;

        const auto t0 = std::chrono::high_resolution_clock::now();
        sim::diag::AllocStats alloc_delta{};
        {
            sim::diag::ScopedAllocCounter alloc_counter;
            for (int i = 0; i < sim_events; ++i) {
                sim::Tick bb = 99, ba = 101;
                sim::Qty bv = 0, av = 0;
                if (!sim_book.best_bid(bb, bv)) {
                    bb = 99;
                    bv = 0;
                }
                if (!sim_book.best_ask(ba, av)) {
                    ba = 101;
                    av = 0;
                }

                auto ev = qrp.next_event(bb, bv, ba, av);

                if (ev.order_type == sim::OrderType::Limit) {
                    sim::Order o{};
                    o.agent = 100;
                    o.side = ev.side;
                    o.price = std::clamp<sim::Tick>(ev.price, 80, 120);
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
                } else if (ev.order_type == sim::OrderType::Cancel) {
                    const sim::Tick px = std::clamp<sim::Tick>(ev.price, 80, 120);
                    const sim::OrderId target_id = (i & 1)
                        ? sim_book.peek_tail_order_id(ev.side, px)
                        : sim_book.peek_head_order_id(ev.side, px);
                    if (target_id != 0 && sim_book.cancel(target_id, qrp.current_time())) {
                        ++total_cancels;
                    }
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
        std::cout << "  Cancels executed:   " << total_cancels << "\n";
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

int run_qr_stylized_mode(int sim_events, std::uint64_t seed,
                         const std::string& dataset_path,
                         const std::string& export_dir) {
    sim::QRParams qrp_cfg{};
    qrp_cfg.num_levels = 5;
    qrp_cfg.theta_ergodic = 0.35;
    qrp_cfg.theta_reprice = 0.65;
    qrp_cfg.size_model = sim::OrderSizeModel::LogBinnedEmpirical;

    sim::QueueReactiveProcess qrp(qrp_cfg, sim::HawkesMatrix::default_microstructure(), seed);
    sim::Empirical3DIntensityTable table = sim::Empirical3DIntensityTable::default_calibrated();
    if (!dataset_path.empty()) {
        table.load_csv(dataset_path);
    }
    qrp.set_empirical_3d_table(table);

    sim::FlatLadderBook book(50, 150, static_cast<std::size_t>(sim_events * 2 + 512));
    for (int lvl = 0; lvl < 5; ++lvl) {
        book.add(sim::Order{.qty = 100, .orig_qty = 100, .price = 99 - lvl,
                            .side = sim::Side::Buy, .type = sim::OrderType::Limit}, 10 + lvl);
        book.add(sim::Order{.qty = 100, .orig_qty = 100, .price = 101 + lvl,
                            .side = sim::Side::Sell, .type = sim::OrderType::Limit}, 20 + lvl);
    }

    std::ofstream csv;
    if (!export_dir.empty()) {
        csv.open(export_dir + "/qr_stylized_trajectory.csv");
        if (csv.is_open()) {
            csv << "step,time_us,best_bid,best_ask,bid_vol_l1,ask_vol_l1,kind,level,side,qty,price_shift,intensity,ogata_residual\n";
        }
    }

    sim::Trade tbuf[32]{};
    std::uint64_t trades = 0, cancels = 0, shifts = 0;
    for (int i = 0; i < sim_events; ++i) {
        sim::Tick bb = 99, ba = 101;
        sim::Qty bv = 0, av = 0;
        if (!book.best_bid(bb, bv) || bv == 0) {
            bb = 99;
            const sim::Qty q_init = qrp.sample_stationary_queue_depth(0);
            book.add_into(sim::Order{.qty = q_init, .orig_qty = q_init, .price = bb,
                                     .side = sim::Side::Buy, .type = sim::OrderType::Limit},
                          qrp.current_time(), tbuf, 32);
            book.best_bid(bb, bv);
        }
        if (!book.best_ask(ba, av) || av == 0) {
            ba = std::max<sim::Tick>(bb + 2, 101);
            const sim::Qty q_init = qrp.sample_stationary_queue_depth(0);
            book.add_into(sim::Order{.qty = q_init, .orig_qty = q_init, .price = ba,
                                     .side = sim::Side::Sell, .type = sim::OrderType::Limit},
                          qrp.current_time(), tbuf, 32);
            book.best_ask(ba, av);
        }

        std::array<sim::Qty, sim::kMaxQRLevels> bid_q{};
        std::array<sim::Qty, sim::kMaxQRLevels> ask_q{};
        for (std::size_t l = 0; l < sim::kMaxQRLevels; ++l) {
            const sim::Tick bp = bb - static_cast<sim::Tick>(l);
            const sim::Tick ap = ba + static_cast<sim::Tick>(l);
            bid_q[l] = (bp >= 50) ? book.volume_at(sim::Side::Buy, bp) : 0;
            ask_q[l] = (ap <= 150) ? book.volume_at(sim::Side::Sell, ap) : 0;
        }

        const auto ev = qrp.next_event_ogata_thinning(bb, bid_q, ba, ask_q);
        if (ev.reprice_ticks != 0) ++shifts;
        if (ev.order_type == sim::OrderType::Limit) {
            const sim::Tick px = std::clamp<sim::Tick>(ev.price, 50, 150);
            const auto res = book.add_into(
                sim::Order{.qty = ev.qty, .orig_qty = ev.qty, .price = px,
                           .side = ev.side, .type = sim::OrderType::Limit},
                qrp.current_time(), tbuf, 32);
            trades += res.trade_count;
        } else if (ev.order_type == sim::OrderType::Market) {
            const auto res = book.add_into(
                sim::Order{.qty = ev.qty, .orig_qty = ev.qty, .price = 0,
                           .side = ev.side, .type = sim::OrderType::Market},
                qrp.current_time(), tbuf, 32);
            trades += res.trade_count;
        } else if (ev.order_type == sim::OrderType::Cancel) {
            const sim::Tick px = std::clamp<sim::Tick>(ev.price, 50, 150);
            const sim::OrderId tid = (i & 1)
                ? book.peek_tail_order_id(ev.side, px)
                : book.peek_head_order_id(ev.side, px);
            if (tid != 0 && book.cancel(tid, qrp.current_time())) {
                ++cancels;
            }
        }

        if (csv.is_open()) {
            const double og_res = qrp.ogata_residuals().empty() ? 0.0 : qrp.ogata_residuals().back();
            csv << i << "," << qrp.current_time() << "," << bb << "," << ba << ","
                << bv << "," << av << "," << static_cast<int>(ev.type) << ","
                << static_cast<int>(ev.level) << ","
                << (ev.side == sim::Side::Buy ? 1 : -1) << "," << ev.qty << ","
                << static_cast<int>(ev.reprice_ticks) << "," << qrp.last_total_intensity() << ","
                << og_res << "\n";
        }
    }

    double mean_res = 0.0;
    for (double r : qrp.ogata_residuals()) mean_res += r;
    if (!qrp.ogata_residuals().empty()) {
        mean_res /= static_cast<double>(qrp.ogata_residuals().size());
    }

    std::cout << "[qr-stylized] events=" << sim_events
              << " trades=" << trades
              << " cancels=" << cancels
              << " reprice_shifts=" << shifts
              << " ogata_mean_residual=" << std::fixed << std::setprecision(4) << mean_res << "\n";
    return 0;
}

int run_latency_sweep_mode(int sim_events, std::uint64_t seed, const std::string& export_dir) {
    const std::int64_t latencies_ns[] = {0, 500, 2000, 10000, 50000, 250000, 1000000};
    std::ofstream csv;
    if (!export_dir.empty()) {
        csv.open(export_dir + "/latency_sweep.csv");
        if (csv.is_open()) {
            csv << "latency_ns,quotes,passive_fills,final_inv,mtm_pnl,sharpe_proxy,maker_markout,maker_adv_x2,queue_risk_x2\n";
        }
    }
    std::cout << "[latency-sweep] Avellaneda-Stoikov across 7 latency tiers (events=" << sim_events << "):\n";
    for (std::int64_t lat : latencies_ns) {
        sim::SimulationSessionConfig scfg{};
        scfg.num_events = static_cast<std::size_t>(sim_events);
        scfg.seed = seed;

        sim::StrategicAgentConfig acfg{};
        acfg.agent_id = 10;
        acfg.kind = sim::StrategicAgentKind::AvellanedaStoikov;
        acfg.latency = sim::LatencyConfig{lat * 4 / 10, lat * 2 / 10, lat * 4 / 10};

        sim::SimulationSession session(scfg, {acfg});
        const auto res = session.run();
        const auto& m = res.agent_metrics.front();
        std::cout << "  latency=" << std::setw(8) << lat << " ns"
                  << " | fills=" << std::setw(4) << m.passive_fills
                  << " | PnL=" << std::setw(9) << std::fixed << std::setprecision(1) << m.mark_to_mid_pnl
                  << " | maker_adv_x2=" << std::setw(6) << m.risk_triplet.maker_adversity_x2
                  << " | queue_risk_x2=" << std::setw(6) << m.risk_triplet.queue_risk_x2 << "\n";
        if (csv.is_open()) {
            csv << lat << "," << m.quotes_emitted << "," << m.passive_fills << ","
                << m.final_inventory << "," << m.mark_to_mid_pnl << ","
                << m.sharpe_proxy << "," << m.mean_maker_markout_ticks << ","
                << m.risk_triplet.maker_adversity_x2 << ","
                << m.risk_triplet.queue_risk_x2 << "\n";
        }
    }
    return 0;
}

int run_mm_tournament_mode(int sim_events, std::uint64_t seed, const std::string& export_dir) {
    sim::SimulationSessionConfig scfg{};
    scfg.num_events = static_cast<std::size_t>(sim_events);
    scfg.seed = seed;

    std::vector<sim::StrategicAgentConfig> agents;
    const sim::StrategicAgentKind kinds[] = {
        sim::StrategicAgentKind::AvellanedaStoikov,
        sim::StrategicAgentKind::GlftHeuristic,
        sim::StrategicAgentKind::CjpHeuristic,
        sim::StrategicAgentKind::PassiveMaker,
        sim::StrategicAgentKind::LatencyTaker
    };
    for (std::size_t k = 0; k < 5; ++k) {
        sim::StrategicAgentConfig acfg{};
        acfg.agent_id = static_cast<sim::AgentId>(10 + k);
        acfg.kind = kinds[k];
        acfg.latency = (kinds[k] == sim::StrategicAgentKind::LatencyTaker)
            ? sim::LatencyConfig{200, 100, 200}
            : sim::LatencyConfig{1500, 500, 1500};
        agents.push_back(acfg);
    }

    sim::SimulationSession session(scfg, agents);
    const auto res = session.run();

    std::ofstream csv;
    if (!export_dir.empty()) {
        csv.open(export_dir + "/mm_tournament.csv");
        if (csv.is_open()) {
            csv << "agent_id,strategy,quotes,passive_fills,aggressive_fills,final_inv,mtm_pnl,sharpe,max_dd,maker_adv_x2,taker_adv_x2,queue_risk_x2\n";
        }
    }

    std::cout << "[mm-tournament] Multi-Agent Market-Making Tournament (events=" << sim_events << "):\n";
    for (const auto& m : res.agent_metrics) {
        std::cout << "  " << std::left << std::setw(18) << agent_kind_name(m.kind)
                  << " | passive=" << std::right << std::setw(4) << m.passive_fills
                  << " | aggr=" << std::setw(4) << m.aggressive_fills
                  << " | inv=" << std::setw(5) << m.final_inventory
                  << " | PnL=" << std::setw(9) << std::fixed << std::setprecision(1) << m.mark_to_mid_pnl
                  << " | Sharpe=" << std::setw(6) << std::setprecision(2) << m.sharpe_proxy << "\n";
        if (csv.is_open()) {
            csv << m.agent_id << "," << agent_kind_name(m.kind) << ","
                << m.quotes_emitted << "," << m.passive_fills << ","
                << m.aggressive_fills << "," << m.final_inventory << ","
                << m.mark_to_mid_pnl << "," << m.sharpe_proxy << ","
                << m.max_drawdown_ticks << ","
                << m.risk_triplet.maker_adversity_x2 << ","
                << m.risk_triplet.taker_adversity_x2 << ","
                << m.risk_triplet.queue_risk_x2 << "\n";
        }
    }
    std::cout << "  Exploitability gain_lower_bound=" << res.exploitability.gain_lower_bound
              << " (is_equilibrium_certificate=" << (res.exploitability.is_equilibrium_certificate ? "true" : "false")
              << ") | Impact RMSE=" << res.impact_score.rmse() << "\n";
    return 0;
}

int run_exec_impact_mode(int sim_events, std::uint64_t seed, const std::string& export_dir) {
    const sim::StrategicAgentKind schedules[] = {
        sim::StrategicAgentKind::TwapExecution,
        sim::StrategicAgentKind::VwapExecution,
        sim::StrategicAgentKind::PovExecution
    };
    std::ofstream csv;
    if (!export_dir.empty()) {
        csv.open(export_dir + "/exec_impact.csv");
        if (csv.is_open()) {
            csv << "schedule,slices_sent,filled_qty,cash_spent,taker_adv_x2,impact_rmse,impact_bias\n";
        }
    }
    std::cout << "[exec-impact] Execution Schedule Comparison (TWAP vs VWAP vs POV):\n";
    for (auto sched : schedules) {
        sim::SimulationSessionConfig scfg{};
        scfg.num_events = static_cast<std::size_t>(sim_events);
        scfg.seed = seed;

        sim::StrategicAgentConfig acfg{};
        acfg.agent_id = 50;
        acfg.kind = sched;
        acfg.execution_target_qty = 120;
        acfg.execution_slices = 12;
        acfg.pov_participation = 0.12;

        sim::SimulationSession session(scfg, {acfg});
        const auto res = session.run();
        const auto& m = res.agent_metrics.front();
        std::cout << "  " << std::left << std::setw(8) << agent_kind_name(sched)
                  << " | slices=" << std::right << std::setw(3) << m.exec_slices_sent
                  << " | filled=" << std::setw(5) << m.total_filled_qty
                  << " | taker_adv_x2=" << std::setw(6) << m.risk_triplet.taker_adversity_x2
                  << " | impact_rmse=" << std::fixed << std::setprecision(4) << res.impact_score.rmse() << "\n";
        if (csv.is_open()) {
            csv << agent_kind_name(sched) << "," << m.exec_slices_sent << ","
                << m.total_filled_qty << "," << (-m.final_cash) << ","
                << m.risk_triplet.taker_adversity_x2 << ","
                << res.impact_score.rmse() << "," << res.impact_score.bias() << "\n";
        }
    }
    return 0;
}

int run_parity_mode(int sim_events, std::uint64_t seed) {
    std::vector<sim::TraceEvent> trace;
    trace.reserve(static_cast<std::size_t>(sim_events));
    sim::Rng rng(seed);
    for (int i = 0; i < sim_events; ++i) {
        sim::TraceEvent ev{};
        ev.seq = static_cast<sim::EventSeq>(i + 1);
        ev.ts = static_cast<sim::TimeNs>((i + 1) * 100);
        ev.agent = static_cast<sim::AgentId>((i % 8) + 1);
        ev.side = (rng.next_u64() & 1ULL) ? sim::Side::Buy : sim::Side::Sell;
        ev.qty = static_cast<sim::Qty>(10 + (rng.next_u64() % 40));
        if ((rng.next_u64() % 100) < 70) {
            ev.kind = sim::TraceEventKind::SubmitLimit;
            ev.price = (ev.side == sim::Side::Buy)
                ? static_cast<sim::Tick>(95 + (rng.next_u64() % 6))
                : static_cast<sim::Tick>(100 + (rng.next_u64() % 6));
        } else {
            ev.kind = sim::TraceEventKind::SubmitMarket;
            ev.price = 0;
        }
        trace.push_back(ev);
    }
    const auto rep = sim::experimental::FlatLadderParityHarness::verify_trace(80, 120, trace);
    std::cout << "[parity] events=" << rep.events_replayed
              << " trades=" << rep.trades_matched
              << " mismatches=" << rep.mismatches
              << " flat_digest=0x" << std::hex << rep.flat_digest
              << " ref_digest=0x" << rep.ref_digest << std::dec
              << " parity_ok=" << (rep.parity_ok ? "true" : "false") << "\n";
    return rep.parity_ok ? 0 : 1;
}

struct TickerCalibrationEntry {
    std::string ticker;
    std::string message_path;
    std::string orderbook_path;
    std::size_t n_events = 0;
    std::size_t n_trades = 0;
    sim::experimental::CalibratedHawkesSummary moment{};
    sim::experimental::HawkesMleResult mle{};
    sim::experimental::VpinResult vpin{};
};

void write_calibration_json(const std::string& json_path,
                            const std::vector<TickerCalibrationEntry>& entries) {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) return;
    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"schema_version\": \"2.0\",\n";
    ofs << "  \"model\": \"Ozaki_Exact_O_N_Queue_Reactive_Hawkes_MLE\",\n";
    ofs << "  \"solver\": \"Hybrid_EM_Projected_Damped_Newton_Armijo\",\n";
    ofs << "  \"train_fraction\": 0.70,\n";
    ofs << "  \"tickers\": {\n";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        const double vpin_val = static_cast<double>(e.vpin.vpin_scaled) / 1e6;
        ofs << "    \"" << e.ticker << "\": {\n";
        ofs << "      \"message_file\": \"" << e.message_path << "\",\n";
        ofs << "      \"orderbook_file\": \"" << e.orderbook_path << "\",\n";
        ofs << "      \"n_events\": " << e.n_events << ",\n";
        ofs << "      \"n_train\": " << e.mle.n_train << ",\n";
        ofs << "      \"n_oos\": " << e.mle.n_oos << ",\n";
        ofs << "      \"n_trades\": " << e.n_trades << ",\n";
        ofs << "      \"mean_resting_l1_depth\": " << e.mle.mean_queue_depth << ",\n";
        ofs << "      \"mean_spread_ticks\": " << e.mle.mean_spread_ticks << ",\n";
        ofs << "      \"empirical_rate_hz\": " << (e.moment.empirical_rate * 1e6) << ",\n";
        ofs << "      \"moment_branching_ratio\": " << e.moment.branching_ratio << ",\n";
        ofs << "      \"mle_mu_hz\": " << e.mle.mu_hz << ",\n";
        ofs << "      \"mle_alpha_hz\": " << e.mle.alpha_hz << ",\n";
        ofs << "      \"mle_beta_hz\": " << e.mle.beta_hz << ",\n";
        ofs << "      \"mle_branching_ratio\": " << e.mle.branching_ratio << ",\n";
        ofs << "      \"mle_gamma_qr\": " << e.mle.gamma_qr << ",\n";
        ofs << "      \"se_mu\": " << e.mle.se_mu << ",\n";
        ofs << "      \"se_alpha\": " << e.mle.se_alpha << ",\n";
        ofs << "      \"se_beta\": " << e.mle.se_beta << ",\n";
        ofs << "      \"se_gamma\": " << e.mle.se_gamma << ",\n";
        ofs << "      \"log_likelihood_init\": " << e.mle.log_likelihood_init << ",\n";
        ofs << "      \"log_likelihood_train\": " << e.mle.log_likelihood_train << ",\n";
        ofs << "      \"log_likelihood_poisson_train\": " << e.mle.log_likelihood_poisson_train << ",\n";
        ofs << "      \"aic_hawkes\": " << e.mle.aic_hawkes << ",\n";
        ofs << "      \"bic_hawkes\": " << e.mle.bic_hawkes << ",\n";
        ofs << "      \"aic_poisson\": " << e.mle.aic_poisson << ",\n";
        ofs << "      \"log_likelihood_oos\": " << e.mle.log_likelihood_oos << ",\n";
        ofs << "      \"log_likelihood_poisson_oos\": " << e.mle.log_likelihood_poisson_oos << ",\n";
        ofs << "      \"oos_llr_per_event\": " << e.mle.oos_llr_per_event << ",\n";
        ofs << "      \"ogata_residual_mean\": " << e.mle.ogata_residual_mean << ",\n";
        ofs << "      \"ogata_residual_std\": " << e.mle.ogata_residual_std << ",\n";
        ofs << "      \"ks_statistic\": " << e.mle.ks_statistic << ",\n";
        ofs << "      \"ks_pvalue\": " << e.mle.ks_pvalue << ",\n";
        ofs << "      \"vpin\": " << vpin_val << ",\n";
        ofs << "      \"vpin_buckets\": " << e.vpin.n_buckets << "\n";
        ofs << "    }" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    ofs << "  }\n";
    ofs << "}\n";
}

int run_calibrate_mode(const std::string& dataset_path) {
    struct TickerSpec {
        const char* ticker;
        const char* msg_path;
        const char* ob_path;
    };
    const TickerSpec default_specs[] = {
        {"AAPL", "data/lobster_AAPL_2012-06-21_message_5.csv", "data/lobster_AAPL_2012-06-21_orderbook_5.csv"},
        {"AMZN", "data/lobster_AMZN_2012-06-21_message_5.csv", "data/lobster_AMZN_2012-06-21_orderbook_5.csv"},
        {"GOOG", "data/lobster_GOOG_2012-06-21_message_5.csv", "data/lobster_GOOG_2012-06-21_orderbook_5.csv"},
        {"INTC", "data/lobster_INTC_2012-06-21_message_5.csv", "data/lobster_INTC_2012-06-21_orderbook_5.csv"},
        {"MSFT", "data/lobster_MSFT_2012-06-21_message_5.csv", "data/lobster_MSFT_2012-06-21_orderbook_5.csv"},
    };

    std::vector<TickerCalibrationEntry> entries;
    if (!dataset_path.empty()) {
        std::vector<sim::experimental::ArrivalEvent> arrivals;
        std::vector<sim::experimental::SignedTrade> trades;
        if (!sim::experimental::load_lobster_paired_csv(dataset_path, arrivals, trades)) {
            std::cerr << "Error: cannot open dataset '" << dataset_path << "'\n";
            return 1;
        }
        TickerCalibrationEntry e{};
        e.ticker = "CUSTOM";
        e.message_path = dataset_path;
        e.orderbook_path = dataset_path;
        const std::size_t pos = e.orderbook_path.find("_message_5.csv");
        if (pos != std::string::npos) {
            e.orderbook_path.replace(pos, 14, "_orderbook_5.csv");
        }
        e.n_events = arrivals.size();
        e.n_trades = trades.size();
        e.moment = sim::experimental::StateDependentHawkesCalibrator::calibrate(arrivals, 1);
        e.mle = sim::experimental::StateDependentHawkesCalibrator::calibrate_mle(
            std::span<const sim::experimental::ArrivalEvent>(arrivals.data(), arrivals.size()), 0.70);
        e.vpin = sim::experimental::evaluate_vpin(trades, 1000);
        entries.push_back(e);
    } else {
        for (const auto& sp : default_specs) {
            std::vector<sim::experimental::ArrivalEvent> arrivals;
            std::vector<sim::experimental::SignedTrade> trades;
            if (!sim::experimental::load_lobster_paired_csv(sp.msg_path, arrivals, trades)) {
                std::cerr << "Error: cannot open dataset '" << sp.msg_path << "'\n";
                return 1;
            }
            TickerCalibrationEntry e{};
            e.ticker = sp.ticker;
            e.message_path = sp.msg_path;
            e.orderbook_path = sp.ob_path;
            e.n_events = arrivals.size();
            e.n_trades = trades.size();
            e.moment = sim::experimental::StateDependentHawkesCalibrator::calibrate(arrivals, 1);
            e.mle = sim::experimental::StateDependentHawkesCalibrator::calibrate_mle(
                std::span<const sim::experimental::ArrivalEvent>(arrivals.data(), arrivals.size()), 0.70);
            e.vpin = sim::experimental::evaluate_vpin(trades, 1000);
            entries.push_back(e);
        }
        write_calibration_json("data/lobster_calibration_summary.json", entries);
    }

    std::cout << "==========================================================================================================\n"
              << "  LOBSTER 5-Level Paired Co-Read & Exact O(N) Ozaki + Queue-Reactive Hawkes MLE Summary (70/30 OOS Split)\n"
              << "==========================================================================================================\n"
              << std::left << std::setw(8) << "Ticker"
              << std::right << std::setw(8) << "Events"
              << std::setw(10) << "MeanQ1"
              << std::setw(10) << "mu(Hz)"
              << std::setw(11) << "alpha(Hz)"
              << std::setw(11) << "beta(Hz)"
              << std::setw(9) << "n_hat"
              << std::setw(9) << "gamma_qr"
              << std::setw(11) << "OOS_LLR"
              << std::setw(9) << "KS_D"
              << std::setw(9) << "VPIN" << "\n"
              << "----------------------------------------------------------------------------------------------------------\n";

    for (const auto& e : entries) {
        const double vpin_val = static_cast<double>(e.vpin.vpin_scaled) / 1e6;
        std::cout << std::left << std::setw(8) << e.ticker
                  << std::right << std::setw(8) << e.n_events
                  << std::fixed << std::setprecision(1) << std::setw(10) << e.mle.mean_queue_depth
                  << std::setprecision(2) << std::setw(10) << e.mle.mu_hz
                  << std::setw(11) << e.mle.alpha_hz
                  << std::setw(11) << e.mle.beta_hz
                  << std::setprecision(4) << std::setw(9) << e.mle.branching_ratio
                  << std::setw(9) << e.mle.gamma_qr
                  << std::setw(11) << e.mle.oos_llr_per_event
                  << std::setw(9) << e.mle.ks_statistic
                  << std::setw(9) << vpin_val << "\n";
    }
    std::cout << "==========================================================================================================\n";

    const auto& first = entries.front();
    std::cout << "[calibrate] dataset=" << first.message_path
              << " events=" << first.n_events
              << " trades=" << first.n_trades
              << " empirical_rate_hz=" << std::fixed << std::setprecision(2) << (first.moment.empirical_rate * 1e6)
              << " empirical_rate_per_us=" << std::scientific << std::setprecision(4) << first.moment.empirical_rate
              << std::fixed << std::setprecision(4)
              << " branching_ratio=" << first.mle.branching_ratio
              << " baseline_mu_hz=" << std::setprecision(2) << first.mle.mu_hz
              << std::setprecision(4)
              << " oos_llr_per_event=" << first.mle.oos_llr_per_event
              << " vpin_buckets=" << first.vpin.n_buckets
              << " vpin=" << (static_cast<double>(first.vpin.vpin_scaled) / 1e6) << "\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    int sim_events = 10000;
    std::uint64_t seed = sim::rng_constants::kDefaultSeed;
    std::string mode = "demo";
    std::string dataset_path;
    std::string export_dir;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--steps") == 0 || std::strcmp(argv[i], "--events") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << argv[i] << " requires a numeric argument\n";
                print_usage(argv[0]);
                return 1;
            }
            sim_events = std::max(100, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--steps=", 8) == 0) {
            sim_events = std::max(100, std::atoi(argv[i] + 8));
        } else if (std::strncmp(argv[i], "--events=", 9) == 0) {
            sim_events = std::max(100, std::atoi(argv[i] + 9));
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --seed requires a numeric argument\n";
                print_usage(argv[0]);
                return 1;
            }
            seed = std::strtoull(argv[++i], nullptr, 0);
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            seed = std::strtoull(argv[i] + 7, nullptr, 0);
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            if (i + 1 >= argc) return 1;
            mode = argv[++i];
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else if (std::strcmp(argv[i], "--dataset") == 0) {
            if (i + 1 >= argc) return 1;
            dataset_path = argv[++i];
        } else if (std::strncmp(argv[i], "--dataset=", 10) == 0) {
            dataset_path = argv[i] + 10;
        } else if (std::strcmp(argv[i], "--export-dir") == 0) {
            if (i + 1 >= argc) return 1;
            export_dir = argv[++i];
        } else if (std::strncmp(argv[i], "--export-dir=", 13) == 0) {
            export_dir = argv[i] + 13;
        } else if (argv[i][0] != '-') {
            sim_events = std::max(100, std::atoi(argv[i]));
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (mode == "demo") {
        return run_demo_mode(sim_events, seed);
    } else if (mode == "qr-stylized") {
        return run_qr_stylized_mode(sim_events, seed, dataset_path, export_dir);
    } else if (mode == "latency-sweep") {
        return run_latency_sweep_mode(sim_events, seed, export_dir);
    } else if (mode == "mm-tournament") {
        return run_mm_tournament_mode(sim_events, seed, export_dir);
    } else if (mode == "exec-impact") {
        return run_exec_impact_mode(sim_events, seed, export_dir);
    } else if (mode == "parity") {
        return run_parity_mode(sim_events, seed);
    } else if (mode == "calibrate") {
        return run_calibrate_mode(dataset_path);
    } else {
        std::cerr << "Error: unknown mode '" << mode << "'\n";
        print_usage(argv[0]);
        return 1;
    }
}
