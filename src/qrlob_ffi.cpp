// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#include "sim/qrlob_ffi.h"
#include "sim/sim.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {

int qrlob_ffi_run_microbenchmarks(
    uint64_t num_ops,
    uint64_t seed,
    QrlobBenchSummary* out_summary) {
    if (!out_summary) return -1;
    std::memset(out_summary, 0, sizeof(QrlobBenchSummary));
    const std::size_t n = std::max<std::size_t>(1000, static_cast<std::size_t>(num_ops));

    // 1. BitwiseVeb4096 predecessor/successor
    {
        sim::BitwiseVeb4096 veb;
        sim::Rng rng(seed);
        for (std::size_t i = 0; i < 512; ++i) {
            veb.insert(static_cast<std::uint16_t>(rng.next_u64() & 4095ULL));
        }
        std::uint64_t sink = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const auto q = static_cast<std::uint16_t>((i * 73ULL + 19ULL) & 4095ULL);
            sink += static_cast<std::uint64_t>(veb.find_prev(q) + veb.find_next(q));
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (sink == 0xFFFFFFFFFFFFFFFFULL) return -2;
        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        out_summary->veb_pred_ns_per_op = ns / static_cast<double>(n);
    }

    // 2. EytzingerTickGrid vs std::lower_bound
    {
        std::vector<std::int64_t> ticks(1024);
        for (std::size_t i = 0; i < 1024; ++i) {
            ticks[i] = static_cast<std::int64_t>(10000 + i * 5);
        }
        sim::EytzingerTickGrid<std::int64_t> grid(ticks);

        std::uint64_t sink1 = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const auto q = static_cast<std::int64_t>(10000 + ((i * 37ULL) & 1023ULL) * 5);
            sink1 += grid.lower_bound_rank(q);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        std::uint64_t sink2 = 0;
        const auto t2 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const auto q = static_cast<std::int64_t>(10000 + ((i * 37ULL) & 1023ULL) * 5);
            auto it = std::lower_bound(ticks.begin(), ticks.end(), q);
            sink2 += static_cast<std::uint64_t>(it - ticks.begin());
        }
        const auto t3 = std::chrono::high_resolution_clock::now();
        if (sink1 != sink2) return -3;
        out_summary->eytzinger_lb_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(n);
        out_summary->std_lower_bound_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) / static_cast<double>(n);
    }

    // 3. OnlineHawkesQ16 step
    {
        sim::OnlineHawkesQ16 hawkes(65536, 16384, 4);
        std::uint64_t sink = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            sink += hawkes.record_event(static_cast<sim::TimeNs>(i * 100), static_cast<std::uint32_t>(i & 7ULL));
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (sink == 0) return -4;
        out_summary->hawkes_q16_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(n);
    }

    // 4. ShmSpscProducer / ShmSpscConsumer roundtrip
    {
        sim::ShmSpscEndpoint ep;
        ep.open("/tmp/qrlob_ffi_bench_shm", true);
        sim::ShmSpscProducer prod(ep.request_channel());
        sim::ShmSpscConsumer cons(ep.request_channel());
        sim::ShmIpcEvent out_ev{};
        std::uint64_t sink = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            prod.publish_add(i + 1, 100, 10, 0, i * 100);
            if (cons.try_dequeue(out_ev)) {
                sink += out_ev.order_id;
            }
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (sink == 0) return -5;
        out_summary->spsc_ring_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(n);
    }

    // 5. FlatLadderBook vs LimitOrderBook + Parity Harness
    {
        const std::size_t trace_len = std::min<std::size_t>(n, 25000);
        std::vector<sim::TraceEvent> trace;
        trace.reserve(trace_len);
        sim::Rng rng(seed ^ 0xA5A5A5A5ULL);
        for (std::size_t i = 0; i < trace_len; ++i) {
            sim::TraceEvent ev{};
            ev.seq = i + 1;
            ev.ts = static_cast<sim::TimeNs>((i + 1) * 100);
            const std::uint64_t r = rng.next_u64() % 100;
            ev.agent = static_cast<sim::AgentId>((i % 8) + 1);
            ev.side = (rng.next_u64() & 1ULL) ? sim::Side::Buy : sim::Side::Sell;
            ev.qty = static_cast<sim::Qty>(10 + (rng.next_u64() % 40));
            if (r < 65) {
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

        sim::FlatLadderBook flat(80, 120, trace_len + 64);
        sim::Trade tbuf[32]{};
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& ev : trace) {
            sim::Order o{};
            o.agent = ev.agent;
            o.side = ev.side;
            o.price = ev.price;
            o.qty = ev.qty;
            o.orig_qty = ev.qty;
            o.type = (ev.kind == sim::TraceEventKind::SubmitMarket)
                ? sim::OrderType::Market : sim::OrderType::Limit;
            flat.add_into(o, ev.ts, tbuf, 32);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        sim::LimitOrderBook ref_lob(trace_len + 64);
        const auto t2 = std::chrono::high_resolution_clock::now();
        for (const auto& ev : trace) {
            sim::Order o{};
            o.agent = ev.agent;
            o.side = ev.side;
            o.price = ev.price;
            o.qty = ev.qty;
            o.orig_qty = ev.qty;
            o.type = (ev.kind == sim::TraceEventKind::SubmitMarket)
                ? sim::OrderType::Market : sim::OrderType::Limit;
            ref_lob.set_time(ev.ts);
            ref_lob.add(o);
        }
        const auto t3 = std::chrono::high_resolution_clock::now();

        out_summary->flat_ladder_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(trace_len);
        out_summary->ref_lob_ns_per_op = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) / static_cast<double>(trace_len);

        const auto parity = sim::experimental::FlatLadderParityHarness::verify_trace(80, 120, trace);
        out_summary->parity_events_replayed = parity.events_replayed;
        out_summary->parity_mismatches = parity.mismatches;
        out_summary->flat_digest = parity.flat_digest;
        out_summary->ref_digest = parity.ref_digest;
    }

    return 0;
}

int qrlob_ffi_simulate_qr_trajectory(
    uint64_t num_events,
    uint64_t seed,
    int use_3d_table,
    const char* table_csv_path,
    int use_ogata_thinning,
    double theta_ergodic,
    double theta_reprice,
    QrlobStylizedStep* out_steps,
    size_t max_steps,
    size_t* out_num_steps) {
    if (!out_steps || !out_num_steps || max_steps == 0) return -1;

    sim::QRParams qrp_cfg{};
    qrp_cfg.num_levels = 5;
    qrp_cfg.theta_ergodic = theta_ergodic;
    qrp_cfg.theta_reprice = theta_reprice;
    qrp_cfg.size_model = sim::OrderSizeModel::LogBinnedEmpirical;

    sim::QueueReactiveProcess qrp(qrp_cfg, sim::HawkesMatrix::default_microstructure(), seed);
    if (use_3d_table) {
        sim::Empirical3DIntensityTable table = sim::Empirical3DIntensityTable::default_calibrated();
        if (table_csv_path && table_csv_path[0] != '\0') {
            table.load_csv(std::string(table_csv_path));
        }
        qrp.set_empirical_3d_table(table);
    }

    sim::FlatLadderBook book(50, 150, static_cast<std::size_t>(num_events * 2 + 512));
    for (int lvl = 0; lvl < 5; ++lvl) {
        book.add(sim::Order{.qty = 100, .orig_qty = 100, .price = 99 - lvl,
                            .side = sim::Side::Buy, .type = sim::OrderType::Limit}, 10 + lvl);
        book.add(sim::Order{.qty = 100, .orig_qty = 100, .price = 101 + lvl,
                            .side = sim::Side::Sell, .type = sim::OrderType::Limit}, 20 + lvl);
    }

    sim::Trade tbuf[32]{};
    const std::size_t steps = std::min<std::size_t>(static_cast<std::size_t>(num_events), max_steps);
    for (std::size_t i = 0; i < steps; ++i) {
        sim::Tick bb = 99, ba = 101;
        sim::Qty bv = 0, av = 0;
        if (!book.best_bid(bb, bv) || bv == 0) {
            bb = 99;
            book.add_into(sim::Order{.qty = 100, .orig_qty = 100, .price = bb,
                                     .side = sim::Side::Buy, .type = sim::OrderType::Limit},
                          qrp.current_time(), tbuf, 32);
            book.best_bid(bb, bv);
        }
        if (!book.best_ask(ba, av) || av == 0) {
            ba = std::max<sim::Tick>(bb + 2, 101);
            book.add_into(sim::Order{.qty = 100, .orig_qty = 100, .price = ba,
                                     .side = sim::Side::Sell, .type = sim::OrderType::Limit},
                          qrp.current_time(), tbuf, 32);
            book.best_ask(ba, av);
        }

        std::array<sim::Qty, sim::kMaxQRLevels> bid_q{};
        std::array<sim::Qty, sim::kMaxQRLevels> ask_q{};
        std::int64_t sum_b5 = 0, sum_a5 = 0;
        for (std::size_t l = 0; l < sim::kMaxQRLevels; ++l) {
            const sim::Tick bp = bb - static_cast<sim::Tick>(l);
            const sim::Tick ap = ba + static_cast<sim::Tick>(l);
            bid_q[l] = (bp >= 50) ? book.volume_at(sim::Side::Buy, bp) : 0;
            ask_q[l] = (ap <= 150) ? book.volume_at(sim::Side::Sell, ap) : 0;
            sum_b5 += bid_q[l];
            sum_a5 += ask_q[l];
        }

        const sim::QREvent ev = use_ogata_thinning
            ? qrp.next_event_ogata_thinning(bb, bid_q, ba, ask_q)
            : qrp.next_event_multilevel(bb, bid_q, ba, ask_q);

        if (ev.order_type == sim::OrderType::Limit) {
            const sim::Tick px = std::clamp<sim::Tick>(ev.price, 50, 150);
            book.add_into(sim::Order{.qty = ev.qty, .orig_qty = ev.qty, .price = px,
                                     .side = ev.side, .type = sim::OrderType::Limit},
                          qrp.current_time(), tbuf, 32);
        } else if (ev.order_type == sim::OrderType::Market) {
            book.add_into(sim::Order{.qty = ev.qty, .orig_qty = ev.qty, .price = 0,
                                     .side = ev.side, .type = sim::OrderType::Market},
                          qrp.current_time(), tbuf, 32);
        } else if (ev.order_type == sim::OrderType::Cancel) {
            const sim::Tick px = std::clamp<sim::Tick>(ev.price, 50, 150);
            const sim::OrderId tid = (i & 1ULL)
                ? book.peek_tail_order_id(ev.side, px)
                : book.peek_head_order_id(ev.side, px);
            if (tid != 0) {
                book.cancel(tid, qrp.current_time());
            }
        }

        QrlobStylizedStep st{};
        st.time_us = qrp.current_time();
        st.best_bid = bb;
        st.best_ask = ba;
        st.bid_vol_l1 = bv;
        st.ask_vol_l1 = av;
        st.total_bid_vol_5 = sum_b5;
        st.total_ask_vol_5 = sum_a5;
        st.event_kind = static_cast<int32_t>(ev.type);
        st.event_level = static_cast<int32_t>(ev.level);
        st.event_side = (ev.side == sim::Side::Buy) ? 1 : -1;
        st.event_qty = ev.qty;
        st.price_shift = ev.reprice_ticks;
        st.total_intensity = qrp.last_total_intensity();
        st.ogata_residual = (!qrp.ogata_residuals().empty())
            ? qrp.ogata_residuals().back()
            : 0.0;
        out_steps[i] = st;
    }

    *out_num_steps = steps;
    return 0;
}

int qrlob_ffi_stationary_queue_pmf(
    size_t max_q,
    double* out_pmf) {
    if (!out_pmf || max_q == 0) return -1;
    sim::QueueReactiveProcess qrp({}, sim::HawkesMatrix::default_microstructure(), 42ULL);
    const auto pmf = qrp.stationary_queue_pmf(0);
    for (std::size_t i = 0; i < max_q && i < pmf.size(); ++i) {
        out_pmf[i] = pmf[i];
    }
    return 0;
}

int qrlob_ffi_run_latency_sweep(
    uint32_t agent_kind,
    const int64_t* latency_ns_grid,
    size_t num_latencies,
    uint64_t events_per_run,
    uint64_t seed,
    QrlobAgentSummary* out_summaries) {
    if (!latency_ns_grid || !out_summaries || num_latencies == 0) return -1;

    for (std::size_t i = 0; i < num_latencies; ++i) {
        const std::int64_t total_lat = std::max<std::int64_t>(0, latency_ns_grid[i]);
        const std::int64_t md_lat = total_lat * 4 / 10;
        const std::int64_t dec_lat = total_lat * 2 / 10;
        const std::int64_t oe_lat = total_lat - md_lat - dec_lat;

        sim::SimulationSessionConfig scfg{};
        scfg.num_events = static_cast<std::size_t>(events_per_run);
        scfg.seed = seed;

        sim::StrategicAgentConfig acfg{};
        acfg.agent_id = 10;
        acfg.scenario_id = static_cast<sim::ScenarioId>(i + 1);
        acfg.kind = static_cast<sim::StrategicAgentKind>(agent_kind);
        acfg.latency = sim::LatencyConfig{md_lat, dec_lat, oe_lat};

        sim::SimulationSession session(scfg, {acfg});
        const auto res = session.run();
        if (!res.agent_metrics.empty()) {
            const auto& m = res.agent_metrics.front();
            QrlobAgentSummary s{};
            s.agent_id = m.agent_id;
            s.kind = static_cast<uint32_t>(m.kind);
            s.total_latency_ns = (m.latency.md_delay_ns + m.latency.decision_delay_ns + m.latency.oe_delay_ns);
            s.quotes_emitted = m.quotes_emitted;
            s.intents_submitted = m.intents_submitted;
            s.intents_rejected = m.intents_rejected;
            s.passive_fills = m.passive_fills;
            s.aggressive_fills = m.aggressive_fills;
            s.total_filled_qty = m.total_filled_qty;
            s.final_inventory = m.final_inventory;
            s.final_cash = m.final_cash;
            s.mark_to_mid_pnl = m.mark_to_mid_pnl;
            s.sharpe_proxy = m.sharpe_proxy;
            s.max_drawdown_ticks = m.max_drawdown_ticks;
            s.mean_maker_markout_ticks = m.mean_maker_markout_ticks;
            s.mean_taker_markout_ticks = m.mean_taker_markout_ticks;
            s.maker_adversity_x2 = m.risk_triplet.maker_adversity_x2;
            s.taker_adversity_x2 = m.risk_triplet.taker_adversity_x2;
            s.queue_risk_x2 = m.risk_triplet.queue_risk_x2;
            out_summaries[i] = s;
        }
    }
    return 0;
}

int qrlob_ffi_run_mm_tournament(
    uint64_t num_events,
    uint64_t seed,
    QrlobAgentSummary* out_summaries,
    size_t max_agents,
    size_t* out_num_agents,
    double* out_exploitability_gain_lb,
    double* out_impact_rmse) {
    if (!out_summaries || !out_num_agents || max_agents == 0) return -1;

    sim::SimulationSessionConfig scfg{};
    scfg.num_events = static_cast<std::size_t>(num_events);
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
        acfg.scenario_id = 1;
        acfg.kind = kinds[k];
        acfg.latency = (kinds[k] == sim::StrategicAgentKind::LatencyTaker)
            ? sim::LatencyConfig{200, 100, 200}
            : sim::LatencyConfig{1500, 500, 1500};
        agents.push_back(acfg);
    }

    sim::SimulationSession session(scfg, agents);
    const auto res = session.run();

    const std::size_t count = std::min<std::size_t>(res.agent_metrics.size(), max_agents);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& m = res.agent_metrics[i];
        QrlobAgentSummary s{};
        s.agent_id = m.agent_id;
        s.kind = static_cast<uint32_t>(m.kind);
        s.total_latency_ns = (m.latency.md_delay_ns + m.latency.decision_delay_ns + m.latency.oe_delay_ns);
        s.quotes_emitted = m.quotes_emitted;
        s.intents_submitted = m.intents_submitted;
        s.intents_rejected = m.intents_rejected;
        s.passive_fills = m.passive_fills;
        s.aggressive_fills = m.aggressive_fills;
        s.total_filled_qty = m.total_filled_qty;
        s.final_inventory = m.final_inventory;
        s.final_cash = m.final_cash;
        s.mark_to_mid_pnl = m.mark_to_mid_pnl;
        s.sharpe_proxy = m.sharpe_proxy;
        s.max_drawdown_ticks = m.max_drawdown_ticks;
        s.mean_maker_markout_ticks = m.mean_maker_markout_ticks;
        s.mean_taker_markout_ticks = m.mean_taker_markout_ticks;
        s.maker_adversity_x2 = m.risk_triplet.maker_adversity_x2;
        s.taker_adversity_x2 = m.risk_triplet.taker_adversity_x2;
        s.queue_risk_x2 = m.risk_triplet.queue_risk_x2;
        out_summaries[i] = s;
    }
    *out_num_agents = count;
    if (out_exploitability_gain_lb) {
        *out_exploitability_gain_lb = res.exploitability.gain_lower_bound;
    }
    if (out_impact_rmse) {
        *out_impact_rmse = res.impact_score.rmse();
    }
    return 0;
}

} // extern "C"
