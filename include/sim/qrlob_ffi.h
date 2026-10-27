#pragma once

// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// ============================================================================
// C-ABI Shared-Library Foreign Function Interface (libqrlob_ffi.so)
//
// Exposes high-performance C-callable entrypoints for Python ctypes/cffi and
// external analytics notebooks over:
//   - BitwiseVeb4096 & EytzingerTickGrid microbenchmarks / lookups
//   - FlatLadderBook vs LimitOrderBook parity and throughput
//   - QueueReactiveProcess (HLR Models I/IIb/III, Hawkes, 3D Noble-Rosenbaum,
//     Ogata exact thinning residuals)
//   - SimulationSession multi-agent market-making & execution tournament
//   - OfflineHawkesCalibrator (moment & exact Ozaki + Queue-Reactive MLE) & VpinToxicityReplay
//   - Unified SPSC Shared Memory Ring ABI (0x434E49524E41454C) + 16B QrlobShmMicrostructureExt
// ============================================================================

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QrlobBenchSummary {
    double flat_ladder_ns_per_op;
    double ref_lob_ns_per_op;
    double veb_pred_ns_per_op;
    double eytzinger_lb_ns_per_op;
    double std_lower_bound_ns_per_op;
    double hawkes_q16_ns_per_op;
    double spsc_ring_ns_per_op;
    uint64_t parity_events_replayed;
    uint64_t parity_mismatches;
    uint64_t flat_digest;
    uint64_t ref_digest;
} QrlobBenchSummary;

typedef struct QrlobStylizedStep {
    int64_t time_us;
    int64_t best_bid;
    int64_t best_ask;
    int64_t bid_vol_l1;
    int64_t ask_vol_l1;
    int64_t total_bid_vol_5;
    int64_t total_ask_vol_5;
    int32_t event_kind;
    int32_t event_level;
    int32_t event_side;
    int64_t event_qty;
    int64_t price_shift;
    double total_intensity;
    double ogata_residual;
} QrlobStylizedStep;

typedef struct QrlobAgentSummary {
    uint32_t agent_id;
    uint32_t kind;
    int64_t total_latency_ns;
    uint64_t quotes_emitted;
    uint64_t intents_submitted;
    uint64_t intents_rejected;
    uint64_t passive_fills;
    uint64_t aggressive_fills;
    int64_t total_filled_qty;
    int64_t final_inventory;
    int64_t final_cash;
    double mark_to_mid_pnl;
    double sharpe_proxy;
    double max_drawdown_ticks;
    double mean_maker_markout_ticks;
    double mean_taker_markout_ticks;
    int64_t maker_adversity_x2;
    int64_t taker_adversity_x2;
    int64_t queue_risk_x2;
} QrlobAgentSummary;

typedef struct QrlobMleCalibrationReport {
    uint64_t n_total;
    uint64_t n_train;
    uint64_t n_oos;
    uint64_t n_trades;
    double mean_queue_depth;
    double mean_spread_ticks;
    double empirical_rate_hz;
    double mu_hz;
    double alpha_hz;
    double beta_hz;
    double branching_ratio;
    double gamma_qr;
    double se_mu;
    double se_alpha;
    double se_beta;
    double se_gamma;
    double log_likelihood_init;
    double log_likelihood_train;
    double log_likelihood_poisson_train;
    double aic_hawkes;
    double bic_hawkes;
    double aic_poisson;
    double log_likelihood_oos;
    double log_likelihood_poisson_oos;
    double oos_llr_per_event;
    double ogata_residual_mean;
    double ogata_residual_std;
    double ks_statistic;
    double ks_pvalue;
    double vpin;
    uint64_t vpin_buckets;
    uint64_t calibration_hash;
} QrlobMleCalibrationReport;

// Run C++ microbenchmarks + FlatLadderParityHarness in-process and populate summary
int qrlob_ffi_run_microbenchmarks(
    uint64_t num_ops,
    uint64_t seed,
    QrlobBenchSummary* out_summary);

// Generate a multi-level Queue-Reactive + Hawkes trajectory (with optional CSV 3D table)
// and fill `out_steps` (up to `max_steps`) plus `out_ogata_residuals` (up to `max_residuals`).
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
    size_t* out_num_steps);

// Evaluate analytical birth-death stationary queue distribution PMF at queue size q
int qrlob_ffi_stationary_queue_pmf(
    size_t max_q,
    double* out_pmf);

// Run a latency sweep for a chosen market-making model (0=AS, 1=GLFT, 2=CJP, 3=Passive)
int qrlob_ffi_run_latency_sweep(
    uint32_t agent_kind,
    const int64_t* latency_ns_grid,
    size_t num_latencies,
    uint64_t events_per_run,
    uint64_t seed,
    QrlobAgentSummary* out_summaries);

// Run a head-to-head multi-agent tournament (AS, GLFT, CJP, PassiveMaker, LatencyTaker)
int qrlob_ffi_run_mm_tournament(
    uint64_t num_events,
    uint64_t seed,
    QrlobAgentSummary* out_summaries,
    size_t max_agents,
    size_t* out_num_agents,
    double* out_exploitability_gain_lb,
    double* out_impact_rmse);

// Calibrate StateDependentHawkesCalibrator & VPIN on a LOBSTER message CSV file
// (co-reading companion _orderbook_5.csv when present).
// Returns 0 on success, negative on error (e.g. non-existent or empty CSV).
int qrlob_ffi_calibrate_lobster_csv(
    const char* csv_path,
    double* out_empirical_rate_hz,
    double* out_branching_ratio,
    double* out_baseline_mu_hz,
    double* out_vpin);

// Calibrate exact O(N) Ozaki + Queue-Reactive MLE on a LOBSTER paired CSV file.
// Optionally writes up to `max_residuals` out-of-sample Ogata time-rescaled residuals.
int qrlob_ffi_calibrate_lobster_mle(
    const char* csv_path,
    double train_fraction,
    QrlobMleCalibrationReport* out_report,
    double* out_oos_residuals,
    size_t max_residuals,
    size_t* out_num_residuals);

// Generate a synthetic Queue-Reactive + Hawkes L3 microstructure stream directly into
// a POSIX shared-memory SPSC ring buffer using the unified 0x434E49524E41454C ABI
// and 16-byte QrlobShmMicrostructureExt telemetry payload.
int qrlob_ffi_generate_shm_microstructure_stream(
    const char* shm_path,
    uint64_t num_events,
    uint64_t seed,
    int use_leangrph_abi,
    uint64_t* out_events_published,
    uint64_t* out_final_hash);

#ifdef __cplusplus
}
#endif
