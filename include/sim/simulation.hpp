#pragma once

// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// ============================================================================
// Discrete-Event Multi-Agent Microstructure Simulation Coordinator
//
// Integrates:
//   1. Three-Stage Latency Pipeline (md_delay_ns -> decision_delay_ns -> oe_delay_ns)
//      with delayed BBO/OFI observation and in-flight order action queues.
//   2. Background L3 Queue-Reactive + Hawkes Point-Process Order Flow
//      (Limit, Cancel via Head/Tail/Nth order IDs, Market, Inside-Spread Create,
//      HLR Model III reference-price shifts & stationary queue redraws).
//   3. Strategic Market-Making & Execution Agents (Avellaneda-Stoikov, GLFT,
//      Cartea-Jaimungal-Penalva, PassiveMaker, LatencyTaker, TWAP/VWAP/POV)
//      guarded by Pre-Trade RiskGuard.
//   4. End-to-End Analytics & Governance Wiring:
//      ReplayPipeline, QueueStateObserver, SurvivalTracker, TcaAccumulator,
//      ThreeRiskTriplet, RewardLedger, ExploitabilityProbe, ImpactScore,
//      SyncFeatureExportSink, and JoinManifestExportHook.
// ============================================================================

#include "analytics.hpp"
#include "features.hpp"
#include "flat_ladder_book.hpp"
#include "queue_reactive.hpp"
#include "quote_adapters.hpp"
#include "replay.hpp"
#include "risk_guard.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace sim {

enum class StrategicAgentKind : std::uint8_t {
    AvellanedaStoikov = 0,
    GlftHeuristic = 1,
    CjpHeuristic = 2,
    PassiveMaker = 3,
    LatencyTaker = 4,
    TwapExecution = 5,
    VwapExecution = 6,
    PovExecution = 7
};

struct StrategicAgentConfig {
    AgentId agent_id = 10;
    ScenarioId scenario_id = 1;
    StrategicAgentKind kind = StrategicAgentKind::AvellanedaStoikov;
    LatencyConfig latency{
        .md_delay_ns = 1500,
        .decision_delay_ns = 500,
        .oe_delay_ns = 1500
    };
    QuoteModelParams quote_params{
        .risk_aversion = 0.08,
        .sigma2_per_second = 1.2,
        .fill_k = 1.5,
        .quote_qty = 10,
        .arrival_a = 1.5,
        .glft_tau_eff = 2.0,
        .q_max = 80,
        .phi_inventory = 0.05,
        .ofi_alpha_weight = 0.60
    };
    RiskLimits risk_limits{
        .max_abs_inventory = 250,
        .max_order_qty = 500,
        .max_order_notional = 5000000,
        .min_price = 50,
        .max_price = 150,
        .max_messages = 100000,
        .message_window_ns = 1000000000LL,
        .max_loss = 0,
        .kill_switch = false
    };
    Qty execution_target_qty = 100;
    Side execution_side = Side::Buy;
    std::uint32_t execution_slices = 10;
    double pov_participation = 0.10;
};

struct DelayedMdObservation {
    TimeNs arrival_at_agent_ns = 0;
    EventSeq source_seq = 0;
    TimeNs exchange_ts_ns = 0;
    Tick best_bid = 99;
    Qty best_bid_qty = 100;
    Tick best_ask = 101;
    Qty best_ask_qty = 100;
    double ofi_signal = 0.0;

    bool operator>(const DelayedMdObservation& other) const noexcept {
        if (arrival_at_agent_ns != other.arrival_at_agent_ns) {
            return arrival_at_agent_ns > other.arrival_at_agent_ns;
        }
        return source_seq > other.source_seq;
    }
};

struct InFlightAgentAction {
    TimeNs arrival_at_exchange_ns = 0;
    std::size_t agent_idx = 0;
    StrategyIntent intent{};
    Tick observed_mid_x2_at_decision = 200;

    bool operator>(const InFlightAgentAction& other) const noexcept {
        if (arrival_at_exchange_ns != other.arrival_at_exchange_ns) {
            return arrival_at_exchange_ns > other.arrival_at_exchange_ns;
        }
        return intent.seq > other.intent.seq;
    }
};

struct AgentRuntimeMetrics {
    AgentId agent_id = 0;
    StrategicAgentKind kind = StrategicAgentKind::AvellanedaStoikov;
    LatencyConfig latency{};
    std::uint64_t quotes_emitted = 0;
    std::uint64_t intents_submitted = 0;
    std::uint64_t intents_rejected = 0;
    std::uint64_t passive_fills = 0;
    std::uint64_t aggressive_fills = 0;
    Qty total_filled_qty = 0;
    Qty final_inventory = 0;
    std::int64_t final_cash = 0;
    double mark_to_mid_pnl = 0.0;
    double sharpe_proxy = 0.0;
    double max_drawdown_ticks = 0.0;
    double mean_maker_markout_ticks = 0.0;
    double mean_taker_markout_ticks = 0.0;
    ThreeRiskTriplet risk_triplet{};
    OrderId resting_bid_order_id = 0;
    OrderId resting_ask_order_id = 0;
    Tick resting_bid_price = 0;
    Tick resting_ask_price = 0;
    DelayedMdObservation latest_known_md{};
    bool has_known_md = false;
    TimeNs last_decision_ns = -1000000000LL;
    std::uint32_t exec_slices_sent = 0;
    double pnl_sum = 0.0;
    double pnl_sq_sum = 0.0;
    double peak_pnl = 0.0;
    std::uint64_t pnl_samples = 0;
};

struct SimulationSessionConfig {
    Tick min_tick = 50;
    Tick max_tick = 150;
    Tick initial_bid = 99;
    Tick initial_ask = 101;
    Qty initial_depth_per_level = 100;
    std::size_t initial_levels = 5;
    std::size_t num_events = 5000;
    std::uint64_t seed = rng_constants::kDefaultSeed;
    TimeNs agent_decision_interval_ns = 25000; // 25 us periodic / reactive wake-up
    TimeNs markout_horizon_ns = 50000;         // 50 us forward markout
    EventSeq markout_horizon_events = 25;
    QRParams qr_params{};
    HawkesMatrix hawkes_params = HawkesMatrix::default_microstructure();
    FeeSchedule fees{
        .maker_fee_per_qty = -1, // 1 tick maker rebate
        .taker_fee_per_qty = 2   // 2 ticks taker fee
    };
    bool enable_multilevel_qr = true;
    bool enable_feature_export = true;
    std::size_t feature_export_stride = 10;
    double impact_law_coef = 0.85;
};

struct SimulationSessionResult {
    std::size_t events_processed = 0;
    std::size_t total_trades = 0;
    std::uint64_t total_volume = 0;
    std::uint64_t background_cancels = 0;
    std::uint64_t price_shifts = 0;
    std::uint64_t final_replay_hash = 0;
    bool invariants_ok = true;
    bool export_parity_ok = true;
    bool join_manifest_clean = true;
    std::vector<AgentRuntimeMetrics> agent_metrics{};
    ImpactScore impact_score{};
    ExploitabilityResult exploitability{};
    JoinManifestSummary join_summary{};
    std::size_t survival_tracked_orders = 0;
    std::size_t survival_filled_orders = 0;
    std::size_t survival_censored_cancel = 0;
    std::size_t survival_censored_horizon = 0;
    std::size_t survival_censored_eod = 0;
};

class SimulationSession {
public:
    explicit SimulationSession(
        SimulationSessionConfig cfg = {},
        std::vector<StrategicAgentConfig> agents = {})
        : cfg_(std::move(cfg)),
          agent_cfgs_(std::move(agents)),
          pipeline_(make_pipeline_cfg(cfg_)),
          qrp_(cfg_.qr_params, cfg_.hawkes_params, cfg_.seed),
          join_hook_(JoinPolicy{
              .embargo_events = 1,
              .embargo_ns = 100,
              .require_same_split = true}) {
        agent_metrics_.resize(agent_cfgs_.size());
        for (std::size_t i = 0; i < agent_cfgs_.size(); ++i) {
            agent_metrics_[i].agent_id = agent_cfgs_[i].agent_id;
            agent_metrics_[i].kind = agent_cfgs_[i].kind;
            agent_metrics_[i].latency = agent_cfgs_[i].latency;
        }
    }

    SimulationSessionResult run() {
        seed_initial_book();

        Tick prev_bb = cfg_.initial_bid;
        Qty prev_bv = cfg_.initial_depth_per_level;
        Tick prev_ba = cfg_.initial_ask;
        Qty prev_av = cfg_.initial_depth_per_level;
        double rolling_ofi = 0.0;

        for (std::size_t step = 0; step < cfg_.num_events; ++step) {
            const FlatLadderBook& book = pipeline_.replay().book();

            Tick bb = cfg_.initial_bid;
            Tick ba = cfg_.initial_ask;
            Qty bv = 0;
            Qty av = 0;
            const bool has_bb = book.best_bid(bb, bv);
            const bool has_ba = book.best_ask(ba, av);
            if (!has_bb) {
                bb = std::max<Tick>(cfg_.min_tick + 2, prev_bb);
                bv = 0;
            }
            if (!has_ba) {
                ba = std::min<Tick>(cfg_.max_tick - 2, std::max<Tick>(bb + 2, prev_ba));
                av = 0;
            }
            if (ba <= bb) {
                ba = std::min<Tick>(cfg_.max_tick - 1, bb + 1);
            }

            // Build 5-level queue depth arrays for multi-level HLR IIb/III generator
            std::array<Qty, kMaxQRLevels> bid_q{};
            std::array<Qty, kMaxQRLevels> ask_q{};
            for (std::size_t lvl = 0; lvl < kMaxQRLevels; ++lvl) {
                const Tick bp = bb - static_cast<Tick>(lvl);
                const Tick ap = ba + static_cast<Tick>(lvl);
                bid_q[lvl] = (bp >= cfg_.min_tick) ? book.volume_at(Side::Buy, bp) : 0;
                ask_q[lvl] = (ap <= cfg_.max_tick) ? book.volume_at(Side::Sell, ap) : 0;
            }

            const QREvent qrev = cfg_.enable_multilevel_qr
                ? qrp_.next_event_multilevel(bb, bid_q, ba, ask_q)
                : qrp_.next_event(bb, bv, ba, av);

            const TimeNs now_ns = qrp_.current_time() * 1000LL; // microseconds -> nanoseconds

            // Drain any in-flight agent actions that arrived at exchange before or at now_ns
            drain_in_flight_actions(now_ns);

            // Execute the background Queue-Reactive event on the exchange book
            execute_background_qr_event(qrev, now_ns, bb, ba);

            // Replenish book if either side completely depleted to preserve realistic two-sided BBO
            ensure_two_sided_liquidity(now_ns, bb, ba);

            // Compute updated BBO & OFI after background event
            Tick new_bb = bb, new_ba = ba;
            Qty new_bv = 0, new_av = 0;
            pipeline_.replay().book().best_bid(new_bb, new_bv);
            pipeline_.replay().book().best_ask(new_ba, new_av);

            const double delta_bid = (new_bb > prev_bb)
                ? static_cast<double>(new_bv)
                : ((new_bb < prev_bb) ? -static_cast<double>(prev_bv)
                                      : static_cast<double>(new_bv - prev_bv));
            const double delta_ask = (new_ba < prev_ba)
                ? static_cast<double>(new_av)
                : ((new_ba > prev_ba) ? -static_cast<double>(prev_av)
                                      : static_cast<double>(new_av - prev_av));
            const double step_ofi = (delta_bid - delta_ask) /
                std::max<double>(1.0, static_cast<double>(new_bv + new_av));
            rolling_ofi = 0.85 * rolling_ofi + 0.15 * step_ofi;

            prev_bb = new_bb;
            prev_bv = new_bv;
            prev_ba = new_ba;
            prev_av = new_av;

            // Enqueue market data observation for each strategic agent through md_delay_ns
            for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
                DelayedMdObservation obs{};
                obs.arrival_at_agent_ns = now_ns + std::max<TimeNs>(0, agent_cfgs_[idx].latency.md_delay_ns);
                obs.source_seq = seq_;
                obs.exchange_ts_ns = now_ns;
                obs.best_bid = new_bb;
                obs.best_bid_qty = new_bv;
                obs.best_ask = new_ba;
                obs.best_ask_qty = new_av;
                obs.ofi_signal = rolling_ofi;
                md_queues_[idx].push(obs);
            }

            // Deliver any market data observations that have reached agents by now_ns
            // and trigger agent decisions scheduled through decision_delay_ns + oe_delay_ns
            process_agent_decisions(now_ns, step);

            // Advance survival trackers
            for (auto& tracker : survival_trackers_) {
                tracker.advance(seq_, now_ns);
            }

            // Periodic feature export through JoinManifestExportHook -> SyncFeatureExportSink
            if (cfg_.enable_feature_export &&
                cfg_.feature_export_stride > 0 &&
                (step % cfg_.feature_export_stride == 0 || step + 1 == cfg_.num_events)) {
                ExportFrame frame{};
                frame.seq = seq_;
                frame.replay_hash = pipeline_.replay().book().state_digest();
                frame.snapshot = make_book_observables(pipeline_.replay().book(), seq_);

                JoinManifestRow jrow{};
                jrow.feature_seq = seq_;
                jrow.feature_ts = now_ns;
                jrow.label_start_seq = seq_ + 2;
                jrow.label_start_ts = now_ns + 200;
                jrow.label_eval_seq = seq_ + cfg_.markout_horizon_events;
                jrow.label_eval_ts = now_ns + cfg_.markout_horizon_ns;
                jrow.feature_split_id = 0;
                jrow.label_split_id = 0;
                jrow.diagnostic_only = false;
                join_hook_.emit_if_usable(export_sink_, frame, jrow);
            }
        }

        // Final drain of remaining in-flight actions
        const TimeNs final_ns = qrp_.current_time() * 1000LL + 10000000LL;
        drain_in_flight_actions(final_ns);

        // Finalize survival trackers and queue state observer
        for (auto& tracker : survival_trackers_) {
            tracker.finalize_end_of_data(seq_, final_ns);
        }
        queue_observer_.finalize_right_censored();
        reward_ledger_.compute_discounted_returns();

        return build_result();
    }

    const ReplayPipeline& pipeline() const noexcept { return pipeline_; }
    const QueueReactiveProcess& qrp() const noexcept { return qrp_; }
    const QueueStateObserver& queue_observer() const noexcept { return queue_observer_; }
    const RewardLedger& reward_ledger() const noexcept { return reward_ledger_; }
    const SyncFeatureExportSink& export_sink() const noexcept { return export_sink_; }
    const JoinManifestExportHook& join_hook() const noexcept { return join_hook_; }
    const std::vector<SurvivalTracker>& survival_trackers() const noexcept {
        return survival_trackers_;
    }

private:
    static ReplayPipelineConfig make_pipeline_cfg(const SimulationSessionConfig& cfg) {
        ReplayPipelineConfig pcfg{};
        pcfg.risk_limits = RiskLimits{
            .max_abs_inventory = 1000000,
            .max_order_qty = 10000,
            .max_order_notional = 1000000000LL,
            .min_price = cfg.min_tick,
            .max_price = cfg.max_tick,
            .max_messages = 10000000,
            .message_window_ns = 1000000000LL,
            .max_loss = 0,
            .kill_switch = false
        };
        pcfg.min_tick = cfg.min_tick;
        pcfg.max_tick = cfg.max_tick;
        pcfg.expected_orders = cfg.num_events * 3 + 512;
        pcfg.enable_compaction = true;
        pcfg.fees = cfg.fees;
        pcfg.markout_horizons = {
            MarkoutHorizon{.events = cfg.markout_horizon_events, .ns = 0},
            MarkoutHorizon{.events = 0, .ns = cfg.markout_horizon_ns}
        };
        return pcfg;
    }

    void seed_initial_book() {
        md_queues_.resize(agent_cfgs_.size());
        for (std::size_t lvl = 0; lvl < cfg_.initial_levels; ++lvl) {
            const Tick bp = cfg_.initial_bid - static_cast<Tick>(lvl);
            const Tick ap = cfg_.initial_ask + static_cast<Tick>(lvl);
            if (bp >= cfg_.min_tick) {
                submit_background_limit(Side::Buy, bp, cfg_.initial_depth_per_level,
                                        static_cast<TimeNs>(100 + lvl * 10));
            }
            if (ap <= cfg_.max_tick) {
                submit_background_limit(Side::Sell, ap, cfg_.initial_depth_per_level,
                                        static_cast<TimeNs>(105 + lvl * 10));
            }
        }
    }

    OrderId submit_background_limit(Side side, Tick price, Qty qty, TimeNs ts_ns) {
        if (price < cfg_.min_tick || price > cfg_.max_tick || qty <= 0) return 0;
        const OrderId ext_id = next_order_id_++;
        StrategyIntent intent{};
        intent.seq = ++seq_;
        intent.ts = ts_ns;
        intent.kind = StrategyIntentKind::SubmitLimit;
        intent.order_id = ext_id;
        intent.parent_id = 0;
        intent.agent = 1; // background liquidity provider
        intent.source = 0;
        intent.side = side;
        intent.price = price;
        intent.qty = qty;

        const Qty ahead = pipeline_.replay().book().volume_at(side, price);
        const auto step = pipeline_.apply(intent);
        if (step.replayed && step.replay.execution.remaining > 0) {
            queue_observer_.on_submit(
                step.replay.execution.id, side, price,
                step.replay.execution.remaining, seq_, ts_ns, ahead);
            if (survival_trackers_.size() < 512) {
                survival_trackers_.emplace_back(SurvivalConfig{
                    .start_seq = seq_,
                    .start_ts = ts_ns,
                    .original_qty = step.replay.execution.remaining,
                    .horizon_events = cfg_.markout_horizon_events * 2,
                    .horizon_ns = cfg_.markout_horizon_ns * 2
                });
                survival_order_ids_.push_back(step.replay.execution.id);
            }
        }
        record_step_fills_and_impact(step, side, qty, ts_ns);
        return ext_id;
    }

    void execute_background_qr_event(const QREvent& qrev, TimeNs now_ns, Tick bb, Tick ba) {
        ++events_processed_;
        if (qrev.reprice_ticks != 0) {
            ++price_shifts_;
        }

        if (qrev.order_type == OrderType::Limit) {
            const Tick clamped_px = std::clamp(qrev.price, cfg_.min_tick, cfg_.max_tick);
            submit_background_limit(qrev.side, clamped_px, std::max<Qty>(1, qrev.qty), now_ns);
        } else if (qrev.order_type == OrderType::Market) {
            const OrderId ext_id = next_order_id_++;
            StrategyIntent intent{};
            intent.seq = ++seq_;
            intent.ts = now_ns;
            intent.kind = StrategyIntentKind::SubmitMarket;
            intent.order_id = ext_id;
            intent.parent_id = 0;
            intent.agent = 2; // background market taker
            intent.source = 0;
            intent.side = qrev.side;
            intent.price = 0;
            intent.qty = std::max<Qty>(1, qrev.qty);
            const auto step = pipeline_.apply(intent);
            record_step_fills_and_impact(step, qrev.side, intent.qty, now_ns);
        } else if (qrev.order_type == OrderType::Cancel) {
            const FlatLadderBook& book = pipeline_.replay().book();
            const Tick target_px = std::clamp(qrev.price, cfg_.min_tick, cfg_.max_tick);
            // Alternate tail and head cancellation for realistic L3 queue depletion
            OrderId internal_target = (events_processed_ & 1ULL)
                ? book.peek_tail_order_id(qrev.side, target_px)
                : book.peek_head_order_id(qrev.side, target_px);
            if (internal_target == 0) {
                internal_target = book.peek_head_order_id(
                    qrev.side, qrev.side == Side::Buy ? bb : ba);
            }
            if (internal_target != 0) {
                // Cancel directly via ReplayDriver using internal order ID
                TraceEvent cev{};
                cev.seq = ++seq_;
                cev.ts = now_ns;
                cev.kind = TraceEventKind::Cancel;
                cev.order_id = internal_target;
                cev.agent = 1;
                cev.side = qrev.side;
                cev.price = target_px;
                const auto rstep = pipeline_.replay().apply(cev);
                if (rstep.cancel_accepted) {
                    ++background_cancels_;
                    queue_observer_.on_own_cancel(internal_target);
                    notify_survival_cancel(internal_target, seq_, now_ns);
                }
            }
        }
    }

    void ensure_two_sided_liquidity(TimeNs now_ns, Tick fallback_bb, Tick fallback_ba) {
        const FlatLadderBook& book = pipeline_.replay().book();
        Tick bb = 0, ba = 0;
        Qty bv = 0, av = 0;
        if (!book.best_bid(bb, bv) || bv == 0) {
            const Tick replenish_bb = std::clamp<Tick>(fallback_bb, cfg_.min_tick + 1, cfg_.max_tick - 2);
            submit_background_limit(Side::Buy, replenish_bb, cfg_.initial_depth_per_level, now_ns);
        }
        if (!book.best_ask(ba, av) || av == 0) {
            const Tick replenish_ba = std::clamp<Tick>(
                std::max<Tick>(fallback_ba, fallback_bb + 2), cfg_.min_tick + 2, cfg_.max_tick - 1);
            submit_background_limit(Side::Sell, replenish_ba, cfg_.initial_depth_per_level, now_ns);
        }
    }

    void process_agent_decisions(TimeNs now_ns, std::size_t step_idx) {
        for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
            auto& mq = md_queues_[idx];
            auto& m = agent_metrics_[idx];
            while (!mq.empty() && mq.top().arrival_at_agent_ns <= now_ns) {
                m.latest_known_md = mq.top();
                m.has_known_md = true;
                mq.pop();
            }
            if (!m.has_known_md) continue;

            const StrategicAgentConfig& acfg = agent_cfgs_[idx];
            if (now_ns - m.last_decision_ns < cfg_.agent_decision_interval_ns) {
                continue;
            }
            m.last_decision_ns = now_ns;

            const DelayedMdObservation& md = m.latest_known_md;
            const Tick mid_x2 = md.best_bid + md.best_ask;
            const TimeNs arrival_at_exchange =
                now_ns + std::max<TimeNs>(0, acfg.latency.decision_delay_ns) +
                std::max<TimeNs>(0, acfg.latency.oe_delay_ns);

            const InventoryState* inv = pipeline_.inventory().find(
                InventoryKey{.agent = acfg.agent_id, .source = acfg.scenario_id});
            const Qty cur_pos = inv ? inv->position : 0;

            if (acfg.kind == StrategicAgentKind::AvellanedaStoikov ||
                acfg.kind == StrategicAgentKind::GlftHeuristic ||
                acfg.kind == StrategicAgentKind::CjpHeuristic ||
                acfg.kind == StrategicAgentKind::PassiveMaker) {
                QuoteInputs qin{};
                qin.mid_x2 = mid_x2;
                qin.inventory = cur_pos;
                qin.now = now_ns;
                qin.horizon_ts = now_ns + 5000000000LL; // 5 s horizon
                qin.alpha_ofi = md.ofi_signal;

                GuardedQuote gq{};
                if (acfg.kind == StrategicAgentKind::PassiveMaker) {
                    gq.model_supported = true;
                    gq.has_bid = (md.best_ask >= md.best_bid);
                    gq.has_ask = (md.best_ask >= md.best_bid);
                    gq.bid_px = md.best_bid;
                    gq.ask_px = md.best_ask;
                    gq.bid_qty = acfg.quote_params.quote_qty;
                    gq.ask_qty = acfg.quote_params.quote_qty;
                } else {
                    const QuoteModelKind qmk =
                        (acfg.kind == StrategicAgentKind::GlftHeuristic)
                            ? QuoteModelKind::GlftHeuristic
                            : ((acfg.kind == StrategicAgentKind::CjpHeuristic)
                                   ? QuoteModelKind::CjpHeuristic
                                   : QuoteModelKind::AvellanedaStoikov);
                    gq = make_guarded_quote(qmk, acfg.quote_params, qin, acfg.risk_limits);
                }

                if (gq.model_supported && (gq.has_bid || gq.has_ask)) {
                    ++m.quotes_emitted;
                    const Tick bid_px = gq.has_bid
                        ? std::clamp(gq.bid_px, cfg_.min_tick, std::min<Tick>(cfg_.max_tick, md.best_ask - 1))
                        : md.best_bid;
                    const Tick ask_px = gq.has_ask
                        ? std::clamp(gq.ask_px, std::max<Tick>(cfg_.min_tick, bid_px + 1), cfg_.max_tick)
                        : md.best_ask;

                    if (m.resting_bid_order_id != 0 && (!gq.has_bid || m.resting_bid_price != bid_px)) {
                        StrategyIntent cancel_bid{};
                        cancel_bid.seq = ++seq_;
                        cancel_bid.ts = arrival_at_exchange;
                        cancel_bid.kind = StrategyIntentKind::Cancel;
                        cancel_bid.order_id = m.resting_bid_order_id;
                        cancel_bid.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                        cancel_bid.agent = acfg.agent_id;
                        cancel_bid.source = acfg.scenario_id;
                        cancel_bid.side = Side::Buy;
                        in_flight_.push(InFlightAgentAction{
                            arrival_at_exchange, idx, cancel_bid, mid_x2});
                        m.resting_bid_order_id = 0;
                    }
                    if (gq.has_bid && m.resting_bid_order_id == 0 && gq.bid_qty > 0) {
                        const OrderId bid_id = next_order_id_++;
                        StrategyIntent submit_bid{};
                        submit_bid.seq = ++seq_;
                        submit_bid.ts = arrival_at_exchange;
                        submit_bid.kind = StrategyIntentKind::SubmitLimit;
                        submit_bid.order_id = bid_id;
                        submit_bid.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                        submit_bid.agent = acfg.agent_id;
                        submit_bid.source = acfg.scenario_id;
                        submit_bid.side = Side::Buy;
                        submit_bid.price = bid_px;
                        submit_bid.qty = gq.bid_qty;
                        in_flight_.push(InFlightAgentAction{
                            arrival_at_exchange, idx, submit_bid, mid_x2});
                        m.resting_bid_order_id = bid_id;
                        m.resting_bid_price = bid_px;
                    }

                    if (m.resting_ask_order_id != 0 && (!gq.has_ask || m.resting_ask_price != ask_px)) {
                        StrategyIntent cancel_ask{};
                        cancel_ask.seq = ++seq_;
                        cancel_ask.ts = arrival_at_exchange;
                        cancel_ask.kind = StrategyIntentKind::Cancel;
                        cancel_ask.order_id = m.resting_ask_order_id;
                        cancel_ask.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                        cancel_ask.agent = acfg.agent_id;
                        cancel_ask.source = acfg.scenario_id;
                        cancel_ask.side = Side::Sell;
                        in_flight_.push(InFlightAgentAction{
                            arrival_at_exchange, idx, cancel_ask, mid_x2});
                        m.resting_ask_order_id = 0;
                    }
                    if (gq.has_ask && m.resting_ask_order_id == 0 && gq.ask_qty > 0) {
                        const OrderId ask_id = next_order_id_++;
                        StrategyIntent submit_ask{};
                        submit_ask.seq = ++seq_;
                        submit_ask.ts = arrival_at_exchange;
                        submit_ask.kind = StrategyIntentKind::SubmitLimit;
                        submit_ask.order_id = ask_id;
                        submit_ask.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                        submit_ask.agent = acfg.agent_id;
                        submit_ask.source = acfg.scenario_id;
                        submit_ask.side = Side::Sell;
                        submit_ask.price = ask_px;
                        submit_ask.qty = gq.ask_qty;
                        in_flight_.push(InFlightAgentAction{
                            arrival_at_exchange, idx, submit_ask, mid_x2});
                        m.resting_ask_order_id = ask_id;
                        m.resting_ask_price = ask_px;
                    }
                }
            } else if (acfg.kind == StrategicAgentKind::LatencyTaker) {
                // Opportunistic sniper triggered by strong OFI or book imbalance
                if (std::fabs(md.ofi_signal) > 0.15) {
                    const Side take_side = (md.ofi_signal > 0.0) ? Side::Buy : Side::Sell;
                    const Qty take_qty = acfg.quote_params.quote_qty;
                    StrategyIntent take{};
                    take.seq = ++seq_;
                    take.ts = arrival_at_exchange;
                    take.kind = StrategyIntentKind::SubmitMarket;
                    take.order_id = next_order_id_++;
                    take.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                    take.agent = acfg.agent_id;
                    take.source = acfg.scenario_id;
                    take.side = take_side;
                    take.price = 0;
                    take.qty = take_qty;
                    in_flight_.push(InFlightAgentAction{
                        arrival_at_exchange, idx, take, mid_x2});
                }
            } else {
                // Execution schedules: TWAP / VWAP / POV
                const std::uint32_t slices = std::max<std::uint32_t>(1, acfg.execution_slices);
                const std::size_t slice_stride = std::max<std::size_t>(1, cfg_.num_events / slices);
                if (m.exec_slices_sent < slices && (step_idx / slice_stride) >= m.exec_slices_sent) {
                    Qty slice_qty = schedule_slice_qty(
                        acfg.execution_target_qty, m.exec_slices_sent, slices);
                    if (acfg.kind == StrategicAgentKind::VwapExecution) {
                        const double depth_ratio = static_cast<double>(md.best_bid_qty + md.best_ask_qty) /
                            std::max<double>(1.0, static_cast<double>(2 * cfg_.initial_depth_per_level));
                        slice_qty = std::max<Qty>(
                            1, static_cast<Qty>(std::llround(static_cast<double>(slice_qty) *
                                                             std::clamp(depth_ratio, 0.5, 1.8))));
                    } else if (acfg.kind == StrategicAgentKind::PovExecution) {
                        const Qty vis = (acfg.execution_side == Side::Buy)
                            ? md.best_ask_qty
                            : md.best_bid_qty;
                        slice_qty = std::max<Qty>(
                            1, static_cast<Qty>(std::llround(acfg.pov_participation *
                                                             static_cast<double>(vis))));
                    }
                    StrategyIntent exec_intent{};
                    exec_intent.seq = ++seq_;
                    exec_intent.ts = arrival_at_exchange;
                    exec_intent.kind = StrategyIntentKind::SubmitMarket;
                    exec_intent.order_id = next_order_id_++;
                    exec_intent.parent_id = static_cast<ParentOrderId>(acfg.agent_id);
                    exec_intent.agent = acfg.agent_id;
                    exec_intent.source = acfg.scenario_id;
                    exec_intent.side = acfg.execution_side;
                    exec_intent.price = 0;
                    exec_intent.qty = slice_qty;
                    in_flight_.push(InFlightAgentAction{
                        arrival_at_exchange, idx, exec_intent, mid_x2});
                    ++m.exec_slices_sent;
                }
            }
        }
    }

    void drain_in_flight_actions(TimeNs up_to_ns) {
        while (!in_flight_.empty() && in_flight_.top().arrival_at_exchange_ns <= up_to_ns) {
            const InFlightAgentAction act = in_flight_.top();
            in_flight_.pop();

            auto& m = agent_metrics_[act.agent_idx];
            const auto& acfg = agent_cfgs_[act.agent_idx];
            ++m.intents_submitted;

            const InventoryState* inv = pipeline_.inventory().find(
                InventoryKey{.agent = acfg.agent_id, .source = acfg.scenario_id});
            RiskState rstate{};
            rstate.inventory = inv ? inv->position : 0;
            rstate.realized_loss = 0;

            RiskGuard agent_guard(acfg.risk_limits);
            const RiskDecision rdec = agent_guard.evaluate(act.intent, rstate);
            if (!rdec.accepted) {
                ++m.intents_rejected;
                reward_ledger_.record_reward(
                    RewardKey{act.intent.parent_id, acfg.agent_id, acfg.scenario_id},
                    -50000);
                continue;
            }

            // Compute queue-risk latency slippage between decision mid and exchange arrival mid
            Tick ex_bb = cfg_.initial_bid, ex_ba = cfg_.initial_ask;
            Qty ex_bv = 0, ex_av = 0;
            pipeline_.replay().book().best_bid(ex_bb, ex_bv);
            pipeline_.replay().book().best_ask(ex_ba, ex_av);
            const Tick ex_mid_x2 = ex_bb + ex_ba;
            const Tick queue_drift_x2 = std::abs(ex_mid_x2 - act.observed_mid_x2_at_decision);
            if (act.intent.kind != StrategyIntentKind::Cancel && act.intent.qty > 0) {
                m.risk_triplet.queue_risk_x2 += queue_drift_x2;
                m.risk_triplet.queue_risk_qty += act.intent.qty;
            }

            const auto step = pipeline_.apply(act.intent, rstate);
            record_step_fills_and_impact(step, act.intent.side, act.intent.qty, act.arrival_at_exchange_ns);
        }
    }

    void record_step_fills_and_impact(const ReplayPipelineStep& step,
                                      Side initiator_side,
                                      Qty submitted_qty,
                                      TimeNs ts_ns) {
        if (!step.replayed) return;
        total_trades_ += step.replay.fills.size();
        for (const FillRecord& fill : step.replay.fills) {
            total_volume_ += static_cast<std::uint64_t>(fill.qty);
            queue_observer_.on_own_fill(fill.maker_id, fill.qty);
            notify_survival_fill(fill.maker_id, fill.qty, fill.seq, fill.ts);

            // Attribute fills & rewards to any strategic maker or taker agent
            for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
                const AgentId aid = agent_cfgs_[idx].agent_id;
                const ScenarioId sid = agent_cfgs_[idx].scenario_id;
                auto& m = agent_metrics_[idx];
                if (fill.maker == aid) {
                    ++m.passive_fills;
                    m.total_filled_qty += fill.qty;
                    if (fill.maker_id == m.resting_bid_order_id) m.resting_bid_order_id = 0;
                    if (fill.maker_id == m.resting_ask_order_id) m.resting_ask_order_id = 0;
                    reward_ledger_.record_fill(
                        RewardKey{fill.maker_parent_id, aid, sid}, fill.qty);
                    reward_ledger_.record_reward(
                        RewardKey{fill.maker_parent_id, aid, sid},
                        static_cast<std::int64_t>(fill.qty) * 150000LL);
                }
                if (fill.taker == aid) {
                    ++m.aggressive_fills;
                    m.total_filled_qty += fill.qty;
                    reward_ledger_.record_fill(
                        RewardKey{fill.taker_parent_id, aid, sid}, fill.qty);
                    reward_ledger_.record_reward(
                        RewardKey{fill.taker_parent_id, aid, sid},
                        static_cast<std::int64_t>(fill.qty) * 50000LL);
                }
            }
        }

        // Record markouts into ThreeRiskTriplet & ImpactScore
        for (const MarkoutRecord& mo : step.markouts) {
            if (mo.has_mid_after && mo.fill.has_mid_at_fill) {
                ImpactObservation iobs{};
                iobs.seq = mo.fill.seq;
                iobs.ts = mo.fill.ts;
                iobs.side = mo.fill.taker_side;
                iobs.participation = std::clamp(
                    static_cast<double>(mo.fill.qty) /
                        std::max<double>(1.0, static_cast<double>(2 * cfg_.initial_depth_per_level)),
                    0.001, 1.0);
                iobs.pre_mid_x2 = mo.fill.mid_x2_at_fill;
                iobs.post_mid_x2 = mo.mid_x2_after;
                iobs.has_pre_mid = true;
                iobs.has_post_mid = true;
                iobs.availability = LabelAvailability::Available;
                impact_score_.add(iobs, cfg_.impact_law_coef);
            }

            for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
                const AgentId aid = agent_cfgs_[idx].agent_id;
                auto& m = agent_metrics_[idx];
                if (mo.fill.maker == aid && mo.has_mid_after) {
                    if (mo.maker_markout_x2 < 0) {
                        m.risk_triplet.maker_adversity_x2 += (-mo.maker_markout_x2);
                        m.risk_triplet.maker_adversity_qty += mo.fill.qty;
                    }
                }
                if (mo.fill.taker == aid && mo.has_mid_after) {
                    if (mo.taker_markout_x2 < 0) {
                        m.risk_triplet.taker_adversity_x2 += (-mo.taker_markout_x2);
                        m.risk_triplet.taker_adversity_qty += mo.fill.qty;
                    }
                }
            }
        }

        // Update agent mark-to-mid PnL trajectories
        if (step.replay.after.has_mid) {
            const Tick mid_x2 = step.replay.after.mid_x2;
            for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
                const AgentId aid = agent_cfgs_[idx].agent_id;
                const ScenarioId sid = agent_cfgs_[idx].scenario_id;
                const InventoryState* inv = pipeline_.inventory().find(
                    InventoryKey{.agent = aid, .source = sid});
                if (!inv) continue;
                auto& m = agent_metrics_[idx];
                const double prev_pnl = m.mark_to_mid_pnl;
                const double cur_pnl = mark_to_mid_value(*inv, mid_x2);
                const double dpnl = cur_pnl - prev_pnl;
                m.mark_to_mid_pnl = cur_pnl;
                m.pnl_sum += dpnl;
                m.pnl_sq_sum += dpnl * dpnl;
                ++m.pnl_samples;
                if (cur_pnl > m.peak_pnl) m.peak_pnl = cur_pnl;
                const double dd = m.peak_pnl - cur_pnl;
                if (dd > m.max_drawdown_ticks) m.max_drawdown_ticks = dd;
            }
        }
        (void)initiator_side;
        (void)submitted_qty;
        (void)ts_ns;
    }

    void notify_survival_fill(OrderId id, Qty qty, EventSeq seq, TimeNs ts) noexcept {
        for (std::size_t i = 0; i < survival_order_ids_.size(); ++i) {
            if (survival_order_ids_[i] == id) {
                survival_trackers_[i].on_fill(qty, seq, ts);
                break;
            }
        }
    }

    void notify_survival_cancel(OrderId id, EventSeq seq, TimeNs ts) noexcept {
        for (std::size_t i = 0; i < survival_order_ids_.size(); ++i) {
            if (survival_order_ids_[i] == id) {
                survival_trackers_[i].on_cancel(seq, ts);
                break;
            }
        }
    }

    SimulationSessionResult build_result() {
        SimulationSessionResult res{};
        res.events_processed = events_processed_;
        res.total_trades = total_trades_;
        res.total_volume = total_volume_;
        res.background_cancels = background_cancels_;
        res.price_shifts = price_shifts_;
        res.final_replay_hash = pipeline_.replay().book().state_digest();
        res.invariants_ok = pipeline_.replay().book().check_invariants_report().ok;
        res.export_parity_ok = export_sink_.accepted_frames() == 0 ||
            (export_sink_.seq_monotone() &&
             export_sink_.last_hash() <= res.final_replay_hash);
        res.join_manifest_clean = !join_hook_.any_leak();
        res.join_summary = join_hook_.summary();
        res.impact_score = impact_score_;

        Tick bb = cfg_.initial_bid, ba = cfg_.initial_ask;
        Qty bv = 0, av = 0;
        pipeline_.replay().book().best_bid(bb, bv);
        pipeline_.replay().book().best_ask(ba, av);
        const Tick final_mid_x2 = bb + ba;

        std::vector<double> payoffs{};
        for (std::size_t idx = 0; idx < agent_cfgs_.size(); ++idx) {
            const AgentId aid = agent_cfgs_[idx].agent_id;
            const ScenarioId sid = agent_cfgs_[idx].scenario_id;
            auto& m = agent_metrics_[idx];
            const InventoryState* inv = pipeline_.inventory().find(
                InventoryKey{.agent = aid, .source = sid});
            if (inv) {
                m.final_inventory = inv->position;
                m.final_cash = inv->cash;
                m.mark_to_mid_pnl = mark_to_mid_value(*inv, final_mid_x2);
            }
            if (m.pnl_samples > 1) {
                const double mean_d = m.pnl_sum / static_cast<double>(m.pnl_samples);
                const double var_d = std::max(
                    1e-12, (m.pnl_sq_sum / static_cast<double>(m.pnl_samples)) - mean_d * mean_d);
                m.sharpe_proxy = (mean_d / std::sqrt(var_d)) * std::sqrt(252.0);
            }
            const TcaSummary* tca = pipeline_.tca().find(TcaKey{
                .parent_id = static_cast<ParentOrderId>(aid),
                .agent = aid,
                .source = sid
            });
            if (tca) {
                m.mean_maker_markout_ticks = average_passive_markout_ticks(*tca);
                m.mean_taker_markout_ticks = average_aggressive_markout_ticks(*tca);
            }
            payoffs.push_back(m.mark_to_mid_pnl);
        }
        res.agent_metrics = agent_metrics_;

        if (!payoffs.empty()) {
            res.exploitability = probe_exploitability(
                payoffs.front(), payoffs.data(), payoffs.size());
        }

        res.survival_tracked_orders = survival_trackers_.size();
        for (const auto& st : survival_trackers_) {
            switch (st.state()) {
                case SurvivalState::Filled:
                    ++res.survival_filled_orders;
                    break;
                case SurvivalState::CensoredCancel:
                    ++res.survival_censored_cancel;
                    break;
                case SurvivalState::CensoredHorizon:
                    ++res.survival_censored_horizon;
                    break;
                case SurvivalState::CensoredEndOfData:
                case SurvivalState::Pending:
                    ++res.survival_censored_eod;
                    break;
            }
        }
        return res;
    }

    SimulationSessionConfig cfg_{};
    std::vector<StrategicAgentConfig> agent_cfgs_{};
    std::vector<AgentRuntimeMetrics> agent_metrics_{};
    ReplayPipeline pipeline_;
    QueueReactiveProcess qrp_;
    QueueStateObserver queue_observer_{};
    RewardLedger reward_ledger_{};
    SyncFeatureExportSink export_sink_{};
    JoinManifestExportHook join_hook_;
    ImpactScore impact_score_{};
    std::vector<SurvivalTracker> survival_trackers_{};
    std::vector<OrderId> survival_order_ids_{};
    std::vector<std::priority_queue<
        DelayedMdObservation,
        std::vector<DelayedMdObservation>,
        std::greater<DelayedMdObservation>>> md_queues_{};
    std::priority_queue<
        InFlightAgentAction,
        std::vector<InFlightAgentAction>,
        std::greater<InFlightAgentAction>> in_flight_{};
    EventSeq seq_ = 0;
    OrderId next_order_id_ = 1;
    std::size_t events_processed_ = 0;
    std::size_t total_trades_ = 0;
    std::uint64_t total_volume_ = 0;
    std::uint64_t background_cancels_ = 0;
    std::uint64_t price_shifts_ = 0;
};

using SimulationCoordinator = SimulationSession;

} // namespace sim
