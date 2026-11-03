# PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
# Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

"""
Shared C-ABI FFI wrapper (`libqrlob_ffi.so`), CLI runner, and publication-grade
plotting utilities for the Queue-Reactive Limit Order Book notebooks.
"""

import ctypes
import json
import math
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Publication Aesthetics
# ---------------------------------------------------------------------------
PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "amber": "#D97706",
    "crimson": "#B91C1C",
    "slate": "#475569",
    "purple": "#6D28D9",
    "emerald": "#047857",
    "gold": "#B45309",
}

def apply_publication_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9.5,
    })

# ---------------------------------------------------------------------------
# Ctypes Structures matching include/sim/qrlob_ffi.h
# ---------------------------------------------------------------------------
class QrlobBenchSummary(ctypes.Structure):
    _fields_ = [
        ("flat_ladder_ns_per_op", ctypes.c_double),
        ("ref_lob_ns_per_op", ctypes.c_double),
        ("veb_pred_ns_per_op", ctypes.c_double),
        ("eytzinger_lb_ns_per_op", ctypes.c_double),
        ("std_lower_bound_ns_per_op", ctypes.c_double),
        ("hawkes_q16_ns_per_op", ctypes.c_double),
        ("spsc_ring_ns_per_op", ctypes.c_double),
        ("parity_events_replayed", ctypes.c_uint64),
        ("parity_mismatches", ctypes.c_uint64),
        ("flat_digest", ctypes.c_uint64),
        ("ref_digest", ctypes.c_uint64),
    ]

class QrlobStylizedStep(ctypes.Structure):
    _fields_ = [
        ("time_us", ctypes.c_int64),
        ("best_bid", ctypes.c_int64),
        ("best_ask", ctypes.c_int64),
        ("bid_vol_l1", ctypes.c_int64),
        ("ask_vol_l1", ctypes.c_int64),
        ("total_bid_vol_5", ctypes.c_int64),
        ("total_ask_vol_5", ctypes.c_int64),
        ("event_kind", ctypes.c_int32),
        ("event_level", ctypes.c_int32),
        ("event_side", ctypes.c_int32),
        ("event_qty", ctypes.c_int64),
        ("price_shift", ctypes.c_int64),
        ("total_intensity", ctypes.c_double),
        ("ogata_residual", ctypes.c_double),
    ]

class QrlobAgentSummary(ctypes.Structure):
    _fields_ = [
        ("agent_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("total_latency_ns", ctypes.c_int64),
        ("quotes_emitted", ctypes.c_uint64),
        ("intents_submitted", ctypes.c_uint64),
        ("intents_rejected", ctypes.c_uint64),
        ("passive_fills", ctypes.c_uint64),
        ("aggressive_fills", ctypes.c_uint64),
        ("total_filled_qty", ctypes.c_int64),
        ("final_inventory", ctypes.c_int64),
        ("final_cash", ctypes.c_int64),
        ("mark_to_mid_pnl", ctypes.c_double),
        ("sharpe_proxy", ctypes.c_double),
        ("max_drawdown_ticks", ctypes.c_double),
        ("mean_maker_markout_ticks", ctypes.c_double),
        ("mean_taker_markout_ticks", ctypes.c_double),
        ("maker_adversity_x2", ctypes.c_int64),
        ("taker_adversity_x2", ctypes.c_int64),
        ("queue_risk_x2", ctypes.c_int64),
    ]

class QrlobMleCalibrationReport(ctypes.Structure):
    _fields_ = [
        ("n_total", ctypes.c_uint64),
        ("n_train", ctypes.c_uint64),
        ("n_oos", ctypes.c_uint64),
        ("n_trades", ctypes.c_uint64),
        ("mean_queue_depth", ctypes.c_double),
        ("mean_spread_ticks", ctypes.c_double),
        ("empirical_rate_hz", ctypes.c_double),
        ("mu_hz", ctypes.c_double),
        ("alpha_hz", ctypes.c_double),
        ("beta_hz", ctypes.c_double),
        ("branching_ratio", ctypes.c_double),
        ("gamma_qr", ctypes.c_double),
        ("se_mu", ctypes.c_double),
        ("se_alpha", ctypes.c_double),
        ("se_beta", ctypes.c_double),
        ("se_gamma", ctypes.c_double),
        ("log_likelihood_init", ctypes.c_double),
        ("log_likelihood_train", ctypes.c_double),
        ("log_likelihood_poisson_train", ctypes.c_double),
        ("aic_hawkes", ctypes.c_double),
        ("bic_hawkes", ctypes.c_double),
        ("aic_poisson", ctypes.c_double),
        ("log_likelihood_oos", ctypes.c_double),
        ("log_likelihood_poisson_oos", ctypes.c_double),
        ("oos_llr_per_event", ctypes.c_double),
        ("ogata_residual_mean", ctypes.c_double),
        ("ogata_residual_std", ctypes.c_double),
        ("ks_statistic", ctypes.c_double),
        ("ks_pvalue", ctypes.c_double),
        ("vpin", ctypes.c_double),
        ("vpin_buckets", ctypes.c_uint64),
        ("calibration_hash", ctypes.c_uint64),
    ]

def load_ffi_library() -> ctypes.CDLL:
    so_path = BUILD_DIR / "libqrlob_ffi.so"
    if not so_path.exists():
        raise FileNotFoundError(f"Shared library not found at {so_path}. Build project first.")
    lib = ctypes.CDLL(str(so_path))

    lib.qrlob_ffi_run_microbenchmarks.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(QrlobBenchSummary)
    ]
    lib.qrlob_ffi_run_microbenchmarks.restype = ctypes.c_int

    lib.qrlob_ffi_simulate_qr_trajectory.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(QrlobStylizedStep), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.qrlob_ffi_simulate_qr_trajectory.restype = ctypes.c_int

    lib.qrlob_ffi_stationary_queue_pmf.argtypes = [
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)
    ]
    lib.qrlob_ffi_stationary_queue_pmf.restype = ctypes.c_int

    lib.qrlob_ffi_run_latency_sweep.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(QrlobAgentSummary)
    ]
    lib.qrlob_ffi_run_latency_sweep.restype = ctypes.c_int

    lib.qrlob_ffi_run_mm_tournament.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(QrlobAgentSummary),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    lib.qrlob_ffi_run_mm_tournament.restype = ctypes.c_int

    lib.qrlob_ffi_calibrate_lobster_csv.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.qrlob_ffi_calibrate_lobster_csv.restype = ctypes.c_int

    lib.qrlob_ffi_calibrate_lobster_mle.argtypes = [
        ctypes.c_char_p, ctypes.c_double, ctypes.POINTER(QrlobMleCalibrationReport),
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.qrlob_ffi_calibrate_lobster_mle.restype = ctypes.c_int

    return lib

def run_ffi_microbenchmarks(num_ops: int = 100000, seed: int = 42) -> Dict[str, object]:
    lib = load_ffi_library()
    out = QrlobBenchSummary()
    rc = lib.qrlob_ffi_run_microbenchmarks(num_ops, seed, ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_run_microbenchmarks failed with rc={rc}")
    return {
        "flat_ladder_ns_per_op": out.flat_ladder_ns_per_op,
        "ref_lob_ns_per_op": out.ref_lob_ns_per_op,
        "veb_pred_ns_per_op": out.veb_pred_ns_per_op,
        "eytzinger_lb_ns_per_op": out.eytzinger_lb_ns_per_op,
        "std_lower_bound_ns_per_op": out.std_lower_bound_ns_per_op,
        "hawkes_q16_ns_per_op": out.hawkes_q16_ns_per_op,
        "spsc_ring_ns_per_op": out.spsc_ring_ns_per_op,
        "parity_events_replayed": out.parity_events_replayed,
        "parity_mismatches": out.parity_mismatches,
        "flat_digest": f"0x{out.flat_digest:016x}",
        "ref_digest": f"0x{out.ref_digest:016x}",
    }

def run_sim_bench_json(filter_pattern: str = "") -> pd.DataFrame:
    bench_bin = BUILD_DIR / "sim_bench"
    json_out = BUILD_DIR / "bench_results.json"
    cmd = [str(bench_bin), f"--benchmark_out={json_out}"]
    if filter_pattern:
        cmd.append(f"--benchmark_filter={filter_pattern}")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = json.loads(json_out.read_text(encoding="ascii"))
    return pd.DataFrame(data["benchmarks"])

def simulate_qr_dataframe(
    num_events: int = 12000,
    seed: int = 20260621,
    use_3d_table: bool = False,
    table_csv_path: Optional[str] = None,
    use_ogata_thinning: bool = True,
    theta_ergodic: float = 0.03,
    theta_reprice: float = 0.65,
) -> pd.DataFrame:
    lib = load_ffi_library()
    buf = (QrlobStylizedStep * num_events)()
    out_n = ctypes.c_size_t(0)
    csv_bytes = table_csv_path.encode("ascii") if table_csv_path else None
    rc = lib.qrlob_ffi_simulate_qr_trajectory(
        num_events, seed, 1 if use_3d_table else 0, csv_bytes,
        1 if use_ogata_thinning else 0, theta_ergodic, theta_reprice,
        buf, num_events, ctypes.byref(out_n)
    )
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_simulate_qr_trajectory failed with rc={rc}")
    n = out_n.value
    rows = []
    for i in range(n):
        s = buf[i]
        rows.append({
            "step": i,
            "time_ns": s.time_us,
            "best_bid": s.best_bid,
            "best_ask": s.best_ask,
            "mid": 0.5 * (s.best_bid + s.best_ask),
            "spread": s.best_ask - s.best_bid,
            "bid_vol_l1": s.bid_vol_l1,
            "ask_vol_l1": s.ask_vol_l1,
            "total_bid_vol_5": s.total_bid_vol_5,
            "total_ask_vol_5": s.total_ask_vol_5,
            "event_kind": s.event_kind,
            "event_level": s.event_level,
            "event_side": s.event_side,
            "event_qty": s.event_qty,
            "price_shift": s.price_shift,
            "total_intensity": s.total_intensity,
            "ogata_residual": s.ogata_residual,
        })
    return pd.DataFrame(rows)

def get_stationary_queue_pmf(max_q: int = 64) -> np.ndarray:
    lib = load_ffi_library()
    arr = (ctypes.c_double * max_q)()
    rc = lib.qrlob_ffi_stationary_queue_pmf(max_q, arr)
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_stationary_queue_pmf failed with rc={rc}")
    return np.array([arr[i] for i in range(max_q)], dtype=np.float64)

def run_latency_sweep_dataframe(
    agent_kind: int = 0,
    latencies_ns: Optional[List[int]] = None,
    events_per_run: int = 4000,
    seed: int = 1337,
) -> pd.DataFrame:
    if latencies_ns is None:
        latencies_ns = [0, 250, 1000, 5000, 25000, 100000, 500000, 2000000]
    lib = load_ffi_library()
    k = len(latencies_ns)
    lat_arr = (ctypes.c_int64 * k)(*latencies_ns)
    out_arr = (QrlobAgentSummary * k)()
    rc = lib.qrlob_ffi_run_latency_sweep(
        agent_kind, lat_arr, k, events_per_run, seed, out_arr
    )
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_run_latency_sweep failed with rc={rc}")
    rows = []
    for i in range(k):
        s = out_arr[i]
        rows.append({
            "latency_ns": s.total_latency_ns,
            "latency_us": s.total_latency_ns / 1000.0,
            "quotes_emitted": s.quotes_emitted,
            "intents_submitted": s.intents_submitted,
            "intents_rejected": s.intents_rejected,
            "passive_fills": s.passive_fills,
            "aggressive_fills": s.aggressive_fills,
            "total_filled_qty": s.total_filled_qty,
            "final_inventory": s.final_inventory,
            "final_cash": s.final_cash,
            "mark_to_mid_pnl": s.mark_to_mid_pnl,
            "sharpe_proxy": s.sharpe_proxy,
            "max_drawdown_ticks": s.max_drawdown_ticks,
            "mean_maker_markout_ticks": s.mean_maker_markout_ticks,
            "mean_taker_markout_ticks": s.mean_taker_markout_ticks,
            "maker_adversity_x2": s.maker_adversity_x2,
            "taker_adversity_x2": s.taker_adversity_x2,
            "queue_risk_x2": s.queue_risk_x2,
        })
    return pd.DataFrame(rows)

def run_mm_tournament_dataframe(
    num_events: int = 6000,
    seed: int = 9001,
) -> Tuple[pd.DataFrame, float, float]:
    lib = load_ffi_library()
    max_agents = 8
    out_arr = (QrlobAgentSummary * max_agents)()
    out_n = ctypes.c_size_t(0)
    gain_lb = ctypes.c_double(0.0)
    impact_rmse = ctypes.c_double(0.0)
    rc = lib.qrlob_ffi_run_mm_tournament(
        num_events, seed, out_arr, max_agents,
        ctypes.byref(out_n), ctypes.byref(gain_lb), ctypes.byref(impact_rmse)
    )
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_run_mm_tournament failed with rc={rc}")
    names = {
        0: "Avellaneda-Stoikov (2008)",
        1: "GLFT (2013) Infinite-Horizon",
        2: "CJP (2015) OFI-Alpha Skew",
        3: "Passive BBO Maker",
        4: "Low-Latency OFI Sniper",
    }
    rows = []
    for i in range(out_n.value):
        s = out_arr[i]
        rows.append({
            "agent_id": s.agent_id,
            "strategy": names.get(s.kind, f"Agent_{s.kind}"),
            "total_latency_ns": s.total_latency_ns,
            "quotes_emitted": s.quotes_emitted,
            "passive_fills": s.passive_fills,
            "aggressive_fills": s.aggressive_fills,
            "total_filled_qty": s.total_filled_qty,
            "final_inventory": s.final_inventory,
            "mark_to_mid_pnl": s.mark_to_mid_pnl,
            "sharpe_proxy": s.sharpe_proxy,
            "max_drawdown_ticks": s.max_drawdown_ticks,
            "maker_adversity_x2": s.maker_adversity_x2,
            "taker_adversity_x2": s.taker_adversity_x2,
            "queue_risk_x2": s.queue_risk_x2,
        })
    return pd.DataFrame(rows), float(gain_lb.value), float(impact_rmse.value)

def ks_statistic_exp1(samples: np.ndarray) -> Tuple[float, float]:
    """Compute exact Kolmogorov-Smirnov D-statistic and asymptotic p-value against Exp(1)."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    x = x[x > 0]
    n = len(x)
    if n == 0:
        return 1.0, 0.0
    cdf = 1.0 - np.exp(-x)
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1.0) / n)
    d = float(max(d_plus, d_minus))
    # Kolmogorov distribution survival approximation
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    pval = 0.0
    for k in range(1, 25):
        pval += 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
    pval = max(0.0, min(1.0, pval))
    return d, pval


def calibrate_lobster_csv_ffi(csv_path: str) -> Dict[str, float]:
    lib = load_ffi_library()
    rate_hz = ctypes.c_double(0.0)
    br = ctypes.c_double(0.0)
    mu_hz = ctypes.c_double(0.0)
    vpin = ctypes.c_double(0.0)
    b_path = csv_path.encode("ascii") if csv_path else None
    rc = lib.qrlob_ffi_calibrate_lobster_csv(
        b_path, ctypes.byref(rate_hz), ctypes.byref(br), ctypes.byref(mu_hz), ctypes.byref(vpin)
    )
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_calibrate_lobster_csv failed with rc={rc}")
    return {
        "empirical_rate_hz": float(rate_hz.value),
        "branching_ratio": float(br.value),
        "baseline_mu_hz": float(mu_hz.value),
        "vpin": float(vpin.value),
    }


def calibrate_lobster_mle_ffi(csv_path: str, train_fraction: float = 0.70):
    lib = load_ffi_library()
    rep = QrlobMleCalibrationReport()
    max_res = 4096
    res_buf = (ctypes.c_double * max_res)()
    out_n = ctypes.c_size_t(0)
    b_path = csv_path.encode("ascii") if csv_path else None
    rc = lib.qrlob_ffi_calibrate_lobster_mle(
        b_path, train_fraction, ctypes.byref(rep), res_buf, max_res, ctypes.byref(out_n)
    )
    if rc != 0:
        raise RuntimeError(f"qrlob_ffi_calibrate_lobster_mle failed with rc={rc}")
    n_r = out_n.value
    residuals = np.array([res_buf[i] for i in range(n_r)], dtype=np.float64)
    summary = {
        "n_total": int(rep.n_total),
        "n_train": int(rep.n_train),
        "n_oos": int(rep.n_oos),
        "n_trades": int(rep.n_trades),
        "mean_queue_depth": float(rep.mean_queue_depth),
        "mean_spread_ticks": float(rep.mean_spread_ticks),
        "empirical_rate_hz": float(rep.empirical_rate_hz),
        "mu_hz": float(rep.mu_hz),
        "alpha_hz": float(rep.alpha_hz),
        "beta_hz": float(rep.beta_hz),
        "branching_ratio": float(rep.branching_ratio),
        "gamma_qr": float(rep.gamma_qr),
        "se_mu": float(rep.se_mu),
        "se_alpha": float(rep.se_alpha),
        "se_beta": float(rep.se_beta),
        "se_gamma": float(rep.se_gamma),
        "log_likelihood_init": float(rep.log_likelihood_init),
        "log_likelihood_train": float(rep.log_likelihood_train),
        "log_likelihood_poisson_train": float(rep.log_likelihood_poisson_train),
        "aic_hawkes": float(rep.aic_hawkes),
        "bic_hawkes": float(rep.bic_hawkes),
        "aic_poisson": float(rep.aic_poisson),
        "log_likelihood_oos": float(rep.log_likelihood_oos),
        "log_likelihood_poisson_oos": float(rep.log_likelihood_poisson_oos),
        "oos_llr_per_event": float(rep.oos_llr_per_event),
        "ogata_residual_mean": float(rep.ogata_residual_mean),
        "ogata_residual_std": float(rep.ogata_residual_std),
        "ks_statistic": float(rep.ks_statistic),
        "ks_pvalue": float(rep.ks_pvalue),
        "vpin": float(rep.vpin),
        "vpin_buckets": int(rep.vpin_buckets),
        "calibration_hash": f"0x{rep.calibration_hash:016x}",
    }
    return summary, residuals
