// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "rng.hpp"
#include "types.hpp"
#include "veb_static_search.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace sim {

// ----------------------------------------------------------------------------
// Branchless Bitwise MUX & SWAR Packed Queue-Reactive Regime LUTs
// (`family2_swar_lut_bounds` / `family4_branchless_mux`)
// ----------------------------------------------------------------------------

inline constexpr std::uint32_t kSwarAddLut    = 0x55865432U;
inline constexpr std::uint32_t kSwarCancelLut = 0x40432210U;
inline constexpr std::uint32_t kSwarExecLut   = 0xFF9F4F00U;

// Branchless bit-multiplexer (`chANF` from `/home/radu/code/sair/kernel/common/BitwiseSWAR.lean`):
// selects bits from `t` where `mask` is 1, and from `f` where `mask` is 0, with zero NOT gates.
constexpr std::uint64_t bit_mux(std::uint64_t mask, std::uint64_t t, std::uint64_t f) noexcept {
    return f ^ (mask & (t ^ f));
}

constexpr std::uint64_t branchless_bit_mux(std::uint64_t cond_mask,
                                           std::uint64_t a,
                                           std::uint64_t b) noexcept {
    return bit_mux(cond_mask, a, b);
}

// Branchless all-ones mask (0xFFFFFFFFFFFFFFFFULL) if x == 0, else 0ULL.
constexpr std::uint64_t is_zero_mask(std::uint64_t x) noexcept {
    return (((x | (0ULL - x)) >> 63ULL) - 1ULL);
}

// Branchless sign-mask ReLU (`family4_cfr_branchless_relu64_exact`).
constexpr std::int64_t verified_cfr_branchless_relu64(std::int64_t v) noexcept {
    const auto w = static_cast<std::uint64_t>(v);
    const auto sign_mask = static_cast<std::uint64_t>(v >> 63);
    return static_cast<std::int64_t>(w & (sign_mask ^ 0xFFFFFFFFFFFFFFFFULL));
}

// Evaluates 3-bit queue depth regime in [0, 7] via SWAR 4-bit nibble lookup
// tables in O(1) with zero floating-point division and zero L1D cache misses.
struct SwarQueueReactiveRegimeLut {
    static constexpr std::uint32_t kAddLut    = kSwarAddLut;
    static constexpr std::uint32_t kCancelLut = kSwarCancelLut;
    static constexpr std::uint32_t kExecLut   = kSwarExecLut;

    static constexpr std::uint32_t extract_nibble(std::uint32_t packed_lut,
                                                  std::uint32_t regime) noexcept {
        return (packed_lut >> ((regime & 7U) << 2U)) & 0xFU;
    }

    static constexpr std::uint32_t add_nibble(std::uint32_t regime) noexcept {
        return extract_nibble(kSwarAddLut, regime);
    }

    static constexpr std::uint32_t cancel_nibble(std::uint32_t regime) noexcept {
        return extract_nibble(kSwarCancelLut, regime);
    }

    static constexpr std::uint32_t exec_nibble(std::uint32_t regime) noexcept {
        return extract_nibble(kSwarExecLut, regime);
    }
    static constexpr std::uint8_t decode_add(std::uint32_t regime) noexcept {
        return static_cast<std::uint8_t>(add_nibble(regime));
    }
    static constexpr std::uint8_t decode_cancel(std::uint32_t regime) noexcept {
        return static_cast<std::uint8_t>(cancel_nibble(regime));
    }
    static constexpr std::uint8_t decode_exec(std::uint32_t regime) noexcept {
        return static_cast<std::uint8_t>(exec_nibble(regime));
    }

    static constexpr std::uint32_t exec_toxicity_bit(std::uint32_t adv_mode, std::uint32_t regime) noexcept {
        return (kSwarExecLut >> (((adv_mode & 3U) << 3U) | (regime & 7U))) & 1U;
    }

    // Classify queue depth into 3-bit regime in [0, 7] via leading-zero log2 bucket.
    static constexpr std::uint32_t classify_regime(Qty bid_vol, Qty ask_vol) noexcept {
        const auto total = static_cast<std::uint64_t>(
            (bid_vol > 0 ? bid_vol : 0) + (ask_vol > 0 ? ask_vol : 0));
        if (total == 0ULL) return 0U;
        const auto lg = static_cast<std::uint32_t>(63 - std::countl_zero(total));
        return lg > 7U ? 7U : lg;
    }
};

// ----------------------------------------------------------------------------
// Deterministic Zero-libm Fixed-Point Q48.16 Horner Exponential Decay &
// Online Hawkes / Queue-Reactive Intensity Evaluator (`OnlineHawkesQ16`)
// (`family3_hawkes_q16_decay_bounded`)
// ----------------------------------------------------------------------------

inline constexpr std::uint64_t kQ16One  = 65536ULL;
inline constexpr std::uint64_t kLn2Q16  = 45426ULL; // round(ln(2) * 65536)

// Raw Lean-verified Family 3 kernel (`verified_hawkes_q16_horner_decay`):
// evaluates 2nd-order Horner polynomial `65536 - r + ((r * r) >> 17)` for r in [0, 65535].
constexpr std::uint64_t verified_hawkes_q16_horner_step(std::uint64_t w) noexcept {
    const std::uint64_t r = w & 65535ULL;
    return (65536ULL - r) + ((r * r) >> 17ULL);
}

// Evaluates exp(-dt_scaled_q16 / 65536.0) in Q48.16 fixed-point [0, 65536]
// using integer dyadic range reduction `dt_scaled = k * ln2_q16 + r`,
// 8x sub-step Horner evaluation `65536 - r_sub + ((r_sub * r_sub) >> 17)` with
// 3 fixed-point squarings, and final dyadic shift `>> k`.
// Achieves < 0.10% max relative error vs std::exp across [0, 4 * ln(2)] with
// zero transcendental libm calls and zero floating-point non-associativity.
constexpr std::uint64_t eval_hawkes_q16_horner_decay(std::uint64_t dt_scaled_q16) noexcept {
    if (dt_scaled_q16 == 0ULL) return kQ16One;
    const std::uint64_t k = dt_scaled_q16 / kLn2Q16;
    if (k >= 32ULL) return 0ULL;
    const std::uint64_t r = dt_scaled_q16 - (k * kLn2Q16);
    const std::uint64_t r_sub = (r + 4ULL) >> 3ULL;
    const std::uint64_t h1 = verified_hawkes_q16_horner_step(r_sub);
    const std::uint64_t h2 = (h1 * h1 + 32768ULL) >> 16ULL;
    const std::uint64_t h4 = (h2 * h2 + 32768ULL) >> 16ULL;
    const std::uint64_t h8 = (h4 * h4 + 32768ULL) >> 16ULL;
    return h8 >> k;
}

// Safe Q48.16 negative natural logarithm `-ln(u)` for a 64-bit random word `raw`.
// Clamps `u_q16` to `[1, 65535]` (or returns `0` when `u_q16 >= 65536`) so
// `std::countl_zero(u_q16) >= 48` and `lz = countl_zero(u_q16) - 48` is strictly
// in `[0, 15]`, eliminating shift undefined behavior for every 64-bit input.
constexpr std::uint64_t eval_neg_ln_u_q16_from_raw(std::uint64_t raw) noexcept {
    const std::uint64_t u_q16 = std::min<std::uint64_t>(65535ULL, (raw & 65535ULL) + 1ULL);
    const auto lz = static_cast<std::uint64_t>(std::countl_zero(u_q16) - 48);
    const std::uint64_t rem_q16 = ((1ULL << (16ULL - lz)) - u_q16) << lz;
    return lz * kLn2Q16 + rem_q16;
}

class OnlineHawkesQ16 {
public:
    static constexpr std::uint64_t kScaleQ16 = kQ16One;

    constexpr explicit OnlineHawkesQ16(std::uint64_t mu_q16 = 10ULL * kQ16One,
                                       std::uint64_t alpha_q16 = 5ULL * kQ16One,
                                       std::uint64_t beta_per_us_q16 = kQ16One) noexcept
        : mu_q16_(mu_q16),
          alpha_q16_(alpha_q16),
          beta_per_us_q16_(beta_per_us_q16),
          excitation_q16_(0ULL),
          last_ts_ns_(0),
          event_count_(0ULL) {}

    constexpr void reset() noexcept {
        excitation_q16_ = 0ULL;
        last_ts_ns_ = 0;
        event_count_ = 0ULL;
    }

    // Decay excitation state to timestamp `ts_ns` without adding a jump.
    constexpr std::uint64_t decay_to(TimeNs ts_ns) noexcept {
        if (event_count_ > 0ULL && ts_ns > last_ts_ns_) {
            const auto dt_ns = static_cast<std::uint64_t>(ts_ns - last_ts_ns_);
            const std::uint64_t dt_scaled_q16 = (dt_ns * beta_per_us_q16_) / 1000ULL;
            const std::uint64_t decay_q16 = eval_hawkes_q16_horner_decay(dt_scaled_q16);
            excitation_q16_ = (excitation_q16_ * decay_q16 + 32768ULL) >> 16ULL;
        }
        if (ts_ns > last_ts_ns_ || event_count_ == 0ULL) {
            last_ts_ns_ = ts_ns;
        }
        return mu_q16_ + excitation_q16_;
    }

    // Process an event arrival at `ts_ns`: decays state, records pre-jump intensity,
    // and adds self-excitation jump `alpha_q16_`. Returns post-jump intensity in Q48.16.
    constexpr std::uint64_t record_event(TimeNs ts_ns, std::uint32_t regime = 0U) noexcept {
        (void)decay_to(ts_ns);
        const std::uint32_t add_mult = SwarQueueReactiveRegimeLut::add_nibble(regime);
        const std::uint64_t scaled_jump = (alpha_q16_ * (4ULL + add_mult)) >> 3ULL;
        excitation_q16_ += (regime == 0U) ? alpha_q16_ : scaled_jump;
        ++event_count_;
        return mu_q16_ + excitation_q16_;
    }

    // Evaluate queue-reactive intensity in Q48.16 combining SWAR regime LUT multipliers
    // with current fixed-point Hawkes excitation.
    constexpr std::uint64_t evaluate_queue_reactive_intensity_q16(
        Qty bid_vol, Qty ask_vol) const noexcept {
        const std::uint32_t regime = SwarQueueReactiveRegimeLut::classify_regime(bid_vol, ask_vol);
        const std::uint32_t add_w = SwarQueueReactiveRegimeLut::add_nibble(regime);
        const std::uint32_t can_w = SwarQueueReactiveRegimeLut::cancel_nibble(regime);
        const std::uint64_t base_mod = (mu_q16_ * (8ULL + add_w + can_w)) >> 4ULL;
        return base_mod + excitation_q16_;
    }

    constexpr std::uint64_t decay_dt(std::uint64_t dt_scaled_q16) noexcept {
        const std::uint64_t decay_q16 = eval_hawkes_q16_horner_decay(dt_scaled_q16);
        excitation_q16_ = (excitation_q16_ * decay_q16 + 32768ULL) >> 16ULL;
        return mu_q16_ + excitation_q16_;
    }
    static constexpr double q16_to_double(std::uint64_t v_q16) noexcept {
        return static_cast<double>(v_q16) / static_cast<double>(kQ16One);
    }
    constexpr std::uint64_t intensity_q16() const noexcept { return mu_q16_ + excitation_q16_; }
    constexpr double intensity_double() const noexcept {
        return static_cast<double>(intensity_q16()) / static_cast<double>(kQ16One);
    }
    constexpr std::uint64_t excitation_q16() const noexcept { return excitation_q16_; }
    constexpr std::uint64_t event_count() const noexcept { return event_count_; }
    constexpr TimeNs last_time_ns() const noexcept { return last_ts_ns_; }

private:
    std::uint64_t mu_q16_;
    std::uint64_t alpha_q16_;
    std::uint64_t beta_per_us_q16_;
    std::uint64_t excitation_q16_;
    TimeNs last_ts_ns_;
    std::uint64_t event_count_;
};

// ----------------------------------------------------------------------------
// Queue-Reactive Model (Huang, Lehalle & Rosenbaum, 2015 Model IIb / III),
// Noble-Rosenbaum-Souilmi (2026) 3D Empirical Table Lookup,
// Bodor-Carlier (2024) Order Size Models, and Ogata Thinning Hawkes Engine
// ----------------------------------------------------------------------------

enum class QueueEventType : std::uint8_t {
    BuyLimit    = 0,
    SellLimit   = 1,
    BuyCancel   = 2,
    SellCancel  = 3,
    BuyMarket   = 4,
    SellMarket  = 5,
    Count       = 6,
    CreateBid   = 6, // Inside-spread limit order improving best bid when spread > 1
    CreateAsk   = 7  // Inside-spread limit order improving best ask when spread > 1
};

inline constexpr std::size_t kEventTypes = static_cast<std::size_t>(QueueEventType::Count);
inline constexpr std::size_t kMaxQrLevels = 5;

enum class OrderSizeModel : std::uint8_t {
    Uniform            = 0, // Bounded uniform [min_qty, max_qty] (default backward-compatible)
    Geometric          = 1, // Shifted geometric distribution scaled by per-level MES
    LogBinnedEmpirical = 2  // Empirical dyadic log-binned distribution normalized by per-level MES
};

struct Empirical3DCell {
    double lambda_add_per_us    = 0.045;
    double lambda_cancel_per_us = 0.035;
    double lambda_exec_per_us   = 0.012;
    double lambda_create_per_us = 0.008;
    std::uint32_t sample_count  = 0;
};

// Noble, Rosenbaum & Souilmi (2026, arXiv:2603.24137) 3D Empirical Intensity Table
// indexed over `(imbalance_bin in [0..20], spread_bin in [0..4], depth_bin in [0..4])`
// using `EytzingerTickGrid` for branchless O(log B) bin lookup.
class Empirical3DIntensityTable {
public:
    static constexpr std::size_t kImbBins    = 21;
    static constexpr std::size_t kSpreadBins = 5;
    static constexpr std::size_t kDepthBins  = 5;
    static constexpr std::size_t kTotalCells = kImbBins * kSpreadBins * kDepthBins;

    Empirical3DIntensityTable() {
        // Scaled integer imbalance thresholds in [-1000, 1000] (width 100 = 0.10)
        std::vector<std::int64_t> imb_edges;
        imb_edges.reserve(kImbBins);
        for (int i = 0; i < static_cast<int>(kImbBins); ++i) {
            imb_edges.push_back(static_cast<std::int64_t>(-1000 + i * 100));
        }
        imb_grid_ = EytzingerTickGrid<std::int64_t>::from_sorted(std::move(imb_edges));

        // Spread bin edges in ticks: 1, 2, 3, 4, 5
        spread_grid_ = EytzingerTickGrid<std::int64_t>::from_sorted({1, 2, 3, 4, 5});

        // Total L1 depth bin edges in lots: 100, 250, 500, 1000, 2000
        depth_grid_ = EytzingerTickGrid<std::int64_t>::from_sorted({100, 250, 500, 1000, 2000});

        init_default_surface();
    }

    static Empirical3DIntensityTable default_calibrated() {
        return Empirical3DIntensityTable{};
    }

    void init_default_surface() noexcept {
        for (std::size_t ib = 0; ib < kImbBins; ++ib) {
            const double imb = -1.0 + 0.1 * static_cast<double>(ib);
            const double abs_imb = std::fabs(imb);
            for (std::size_t sb = 0; sb < kSpreadBins; ++sb) {
                for (std::size_t db = 0; db < kDepthBins; ++db) {
                    const double tot_rate = 0.08 * (1.0 + 0.65 * abs_imb) * (1.0 + 0.15 * static_cast<double>(db));
                    const double p_add    = 0.44 - 0.08 * abs_imb;
                    const double p_cancel = 0.40 + 0.04 * static_cast<double>(db);
                    const double p_exec   = 0.12 + 0.10 * abs_imb;
                    const double p_create = (sb > 0) ? (0.10 * static_cast<double>(sb)) : 0.0;
                    const double norm = p_add + p_cancel + p_exec + p_create;
                    Empirical3DCell cell{};
                    cell.lambda_add_per_us    = tot_rate * (p_add / norm);
                    cell.lambda_cancel_per_us = tot_rate * (p_cancel / norm);
                    cell.lambda_exec_per_us   = tot_rate * (p_exec / norm);
                    cell.lambda_create_per_us = tot_rate * (p_create / norm);
                    cell.sample_count = 100;
                    cells_[flat_index(ib, sb, db)] = cell;
                }
            }
        }
    }

    // Load calibrated 525-cell CSV (`hlr_calibrated_5level.csv`)
    bool load_csv(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return false;
        std::string line;
        std::size_t loaded = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#' || line[0] == 'i') continue;
            std::stringstream ss(line);
            std::string tok;
            std::vector<std::string> cols;
            while (std::getline(ss, tok, ',')) cols.push_back(tok);
            if (cols.size() < 10) continue;
            const auto ib = static_cast<std::size_t>(std::max(0, std::min(20, std::atoi(cols[0].c_str()))));
            const auto sb = static_cast<std::size_t>(std::max(0, std::min(4, std::atoi(cols[2].c_str()))));
            const auto db = static_cast<std::size_t>(std::max(0, std::min(4, std::atoi(cols[4].c_str()))));
            Empirical3DCell cell{};
            cell.sample_count         = static_cast<std::uint32_t>(std::max(0, std::atoi(cols[5].c_str())));
            cell.lambda_add_per_us    = std::max(1e-6, std::atof(cols[6].c_str()));
            cell.lambda_cancel_per_us = std::max(1e-6, std::atof(cols[7].c_str()));
            cell.lambda_exec_per_us   = std::max(1e-6, std::atof(cols[8].c_str()));
            cell.lambda_create_per_us = std::max(0.0, std::atof(cols[9].c_str()));
            cells_[flat_index(ib, sb, db)] = cell;
            ++loaded;
        }
        return loaded > 0;
    }

    // Branchless Eytzinger bin lookup for `(imbalance in [-1, 1], spread_ticks, total_depth)`
    std::size_t classify_imbalance_bin(Qty bid_vol, Qty ask_vol) const noexcept {
        const std::int64_t b = std::max<Qty>(0, bid_vol);
        const std::int64_t a = std::max<Qty>(0, ask_vol);
        const std::int64_t tot = b + a;
        if (tot <= 0 || b == a) return 10U; // Exact zero point bin per Noble-Rosenbaum (2026)
        const std::int64_t scaled = ((b - a) * 1000) / tot;
        const std::size_t r = imb_grid_.lower_bound_rank(scaled);
        if (scaled < 0) {
            return std::min<std::size_t>(9U, r > 0 ? r - 1 : 0U);
        }
        return std::max<std::size_t>(11U, std::min<std::size_t>(20U, r));
    }

    std::size_t classify_spread_bin(Tick spread_ticks) const noexcept {
        const std::int64_t s = std::max<Tick>(1, spread_ticks);
        const std::size_t r = spread_grid_.lower_bound_rank(s);
        return std::min<std::size_t>(kSpreadBins - 1, r);
    }

    std::size_t classify_depth_bin(Qty total_l1_depth) const noexcept {
        const std::int64_t d = std::max<Qty>(0, total_l1_depth);
        const std::size_t r = depth_grid_.lower_bound_rank(d);
        return std::min<std::size_t>(kDepthBins - 1, r);
    }

    const Empirical3DCell& lookup(Qty bid_vol, Qty ask_vol, Tick spread_ticks) const noexcept {
        const std::size_t ib = classify_imbalance_bin(bid_vol, ask_vol);
        const std::size_t sb = classify_spread_bin(spread_ticks);
        const std::size_t db = classify_depth_bin(bid_vol + ask_vol);
        return cells_[flat_index(ib, sb, db)];
    }

    const Empirical3DCell& cell_at(std::size_t ib, std::size_t sb, std::size_t db) const noexcept {
        return cells_[flat_index(
            std::min(ib, kImbBins - 1),
            std::min(sb, kSpreadBins - 1),
            std::min(db, kDepthBins - 1))];
    }

    void set_cell(std::size_t ib, std::size_t sb, std::size_t db, const Empirical3DCell& c) noexcept {
        if (ib < kImbBins && sb < kSpreadBins && db < kDepthBins) {
            cells_[flat_index(ib, sb, db)] = c;
        }
    }

    static constexpr std::size_t flat_index(std::size_t ib, std::size_t sb, std::size_t db) noexcept {
        return (ib * kSpreadBins + sb) * kDepthBins + db;
    }

private:
    EytzingerTickGrid<std::int64_t> imb_grid_{};
    EytzingerTickGrid<std::int64_t> spread_grid_{};
    EytzingerTickGrid<std::int64_t> depth_grid_{};
    std::array<Empirical3DCell, kTotalCells> cells_{};
};

struct QueueReactiveConfig {
    // Base exogenous intensities (events / microsecond)
    double base_limit_rate   = 0.05;
    double base_cancel_rate  = 0.002;
    double base_market_rate  = 0.01;

    // Queue depth saturation parameters: lambda_L(q) = base_L / (1 + c_L * q^gamma)
    double limit_depth_decay = 0.01;
    double limit_gamma       = 0.75;

    // Cancel sensitivity: lambda_C(q) = base_C * q^delta
    double cancel_delta      = 1.0;

    // Market order depth affinity: lambda_M(q) = base_M * (1 + c_M * ln(1 + q))
    double market_affinity   = 0.15;

    // Order size bounds and model (Bodor & Carlier 2024 / HLR Section 2.1)
    Qty min_qty              = 1;
    Qty max_qty              = 100;
    OrderSizeModel size_model = OrderSizeModel::Uniform;
    double geometric_p       = 0.25;
    std::array<Qty, kMaxQrLevels> level_mes = {20, 30, 40, 50, 50};
    // Empirical log-binned size weights for sizes [1, 2, 4, 8, 16, 32, 64, 128] in MES fractions
    std::array<double, 8> empirical_size_weights = {0.18, 0.24, 0.22, 0.16, 0.10, 0.06, 0.03, 0.01};

    // Multi-level HLR Model IIb / III configuration (K in [1, 5])
    std::size_t num_levels           = 1;     // Default 1 for 100% L1 backward compatibility
    double opposite_queue_beta       = 0.0;   // Model IIb coupling to opposite best queue
    double imbalance_sensitivity     = 0.0;   // Logistic L1 imbalance sensitivity kappa
    double theta_ergodic             = 0.02;  // HLR Theorem 2.1 ergodic damping parameter (> 0)
    double theta_reprice             = 0.65;  // HLR Model III reference-price shift probability on depletion
    double theta_reinit              = 0.80;  // Stationary invariant queue-depth redraw probability on shift
    bool enable_inside_spread_orders = false; // Allow CreateBid / CreateAsk when spread_ticks > 1
    bool use_empirical_3d_table      = false; // Noble-Rosenbaum (2026) 3D table mode
    bool record_ogata_residuals      = true;  // Record integrated conditional intensity residuals tau_k

    // Level decay multipliers for levels 1..5
    std::array<double, kMaxQrLevels> level_limit_multipliers  = {1.00, 0.62, 0.38, 0.24, 0.16};
    std::array<double, kMaxQrLevels> level_cancel_multipliers = {1.00, 0.78, 0.55, 0.40, 0.30};

    // Transient Power-Law Market Impact (Bouchaud-Farmer-Lillo 2009) & OU Alpha Drift
    bool enable_transient_impact = false;
    double impact_eta            = 0.045;
    double impact_delta          = 0.50;
    double impact_gamma          = 0.52;
    double impact_tau0_us        = 10.0;
    bool enable_ou_alpha_drift   = false;
    double ou_kappa_per_us       = 0.001;
    double ou_sigma_per_sqrt_us  = 0.0005;
};

using QueueReactiveRegimeConfig = QueueReactiveConfig;
using QRParams = QueueReactiveConfig;
inline constexpr std::size_t kMaxQRLevels = kMaxQrLevels;

struct HawkesMatrix {
    // Self and cross-excitation matrix alpha[i][j]: event j excites event i
    std::array<std::array<double, kEventTypes>, kEventTypes> alpha{};
    // Decay rates beta[i][j] (1 / nanosecond)
    std::array<std::array<double, kEventTypes>, kEventTypes> beta{};
    // Unit scaling factor converting dimensionless excitation jumps into baseline rate units (events / us).
    // With default `hawkes_scale = 0.01`, an alpha jump of 0.35 adds `0.0035 / us` (~1.2x-2.5x amplification
    // over baseline `mu in [0.002, 0.05] / us`) rather than overwhelming baseline rates by 10x.
    double hawkes_scale = 0.01;

    static HawkesMatrix default_microstructure() noexcept {
        HawkesMatrix hm{};
        hm.hawkes_scale = 0.01;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            for (std::size_t j = 0; j < kEventTypes; ++j) {
                hm.beta[i][j] = 0.001; // 1 us half-life scale
            }
        }
        // Market orders strongly excite same-direction market orders (momentum)
        hm.alpha[static_cast<std::size_t>(QueueEventType::BuyMarket)][static_cast<std::size_t>(QueueEventType::BuyMarket)] = 0.35;
        hm.alpha[static_cast<std::size_t>(QueueEventType::SellMarket)][static_cast<std::size_t>(QueueEventType::SellMarket)] = 0.35;
        // Large market orders excite opposite-side cancellations (fleeting quotes)
        hm.alpha[static_cast<std::size_t>(QueueEventType::SellCancel)][static_cast<std::size_t>(QueueEventType::BuyMarket)] = 0.25;
        hm.alpha[static_cast<std::size_t>(QueueEventType::BuyCancel)][static_cast<std::size_t>(QueueEventType::SellMarket)] = 0.25;
        // Cancellations excite further cancellations (cascading withdrawals)
        hm.alpha[static_cast<std::size_t>(QueueEventType::BuyCancel)][static_cast<std::size_t>(QueueEventType::BuyCancel)] = 0.20;
        hm.alpha[static_cast<std::size_t>(QueueEventType::SellCancel)][static_cast<std::size_t>(QueueEventType::SellCancel)] = 0.20;
        return hm;
    }
};

struct GeneratedEvent {
    QueueEventType type             = QueueEventType::BuyLimit;
    Side           side             = Side::Buy;
    OrderType      order_type       = OrderType::Limit;
    Tick           price            = 0;
    Qty            qty              = 1;
    TimeNs         inter_arrival_ns = 1000;
    std::uint8_t   level            = 1;     // Queue level 1..K (0 for inside-spread CreateBid/CreateAsk)
    bool           reprice_triggered = false; // True when HLR Model III reference-price shift fired
    std::int8_t    reprice_ticks    = 0;     // Reference price shift (+1 or -1 tick)
    double         rescaled_tau     = 0.0;   // Ogata time-rescaling residual tau_k = \int \Lambda(s) ds
};
using QREvent = GeneratedEvent;

struct MultiLevelEventRates {
    std::size_t num_levels = 1;
    std::array<double, kMaxQrLevels> buy_limit{};
    std::array<double, kMaxQrLevels> sell_limit{};
    std::array<double, kMaxQrLevels> buy_cancel{};
    std::array<double, kMaxQrLevels> sell_cancel{};
    double buy_market  = 0.0;
    double sell_market = 0.0;
    double create_bid  = 0.0;
    double create_ask  = 0.0;

    double total_rate() const noexcept {
        double s = buy_market + sell_market + create_bid + create_ask;
        for (std::size_t m = 0; m < num_levels; ++m) {
            s += buy_limit[m] + sell_limit[m] + buy_cancel[m] + sell_cancel[m];
        }
        return std::max(1e-9, s);
    }
};

struct TransientImpactKernelEvent {
    TimeNs ts_ns = 0;
    double signed_vol = 0.0;
};

class QueueReactiveProcess {
public:
    explicit QueueReactiveProcess(QueueReactiveConfig cfg = {},
                                 HawkesMatrix hawkes = HawkesMatrix::default_microstructure(),
                                 std::uint64_t seed = rng_constants::kDefaultSeed)
        : cfg_(cfg), hawkes_(hawkes), rng_(seed), current_time_ns_(0),
          hawkes_q16_(
              static_cast<std::uint64_t>(std::max(1.0, (2.0 * cfg.base_limit_rate + 2.0 * cfg.base_cancel_rate + 2.0 * cfg.base_market_rate) * static_cast<double>(kQ16One))),
              static_cast<std::uint64_t>(std::max(1.0, 0.25 * hawkes.hawkes_scale * static_cast<double>(kQ16One))),
              kQ16One) {
        cfg_.num_levels = std::max<std::size_t>(1, std::min<std::size_t>(kMaxQrLevels, cfg_.num_levels));
        for (auto& row : state_) {
            row.fill(0.0);
        }
        if (cfg_.record_ogata_residuals) {
            ogata_residuals_.reserve(65536U);
        }
        init_q16_tables();
        build_stationary_cdf_tables();
    }

    const QueueReactiveConfig& config() const noexcept { return cfg_; }
    void set_config(const QueueReactiveConfig& cfg) noexcept {
        cfg_ = cfg;
        cfg_.num_levels = std::max<std::size_t>(1, std::min<std::size_t>(kMaxQrLevels, cfg_.num_levels));
        init_q16_tables();
        build_stationary_cdf_tables();
    }

    const Empirical3DIntensityTable& empirical_table() const noexcept { return table_3d_; }
    Empirical3DIntensityTable& empirical_table() noexcept { return table_3d_; }

    // Compute instantaneous L1 event intensities given current best bid and ask volumes
    std::array<double, kEventTypes> compute_intensities(Qty bid_volume, Qty ask_volume,
                                                        Tick spread_ticks = 1) const noexcept {
        std::array<double, kEventTypes> lambda{};

        const double q_bid = static_cast<double>(std::max<Qty>(0, bid_volume));
        const double q_ask = static_cast<double>(std::max<Qty>(0, ask_volume));
        const double tot = q_bid + q_ask;
        const double imb = (tot > 0.0) ? ((q_bid - q_ask) / tot) : 0.0;

        if (cfg_.use_empirical_3d_table) {
            const Empirical3DCell& cell = table_3d_.lookup(bid_volume, ask_volume, spread_ticks);
            const double imb_skew_bid = std::exp(-0.35 * imb);
            const double imb_skew_ask = std::exp(+0.35 * imb);
            lambda[static_cast<std::size_t>(QueueEventType::BuyLimit)]   = 0.5 * cell.lambda_add_per_us * imb_skew_bid;
            lambda[static_cast<std::size_t>(QueueEventType::SellLimit)]  = 0.5 * cell.lambda_add_per_us * imb_skew_ask;
            lambda[static_cast<std::size_t>(QueueEventType::BuyCancel)]  = (q_bid > 0.0) ? (0.5 * cell.lambda_cancel_per_us * (1.0 + 0.2 * std::log1p(q_bid))) : 1e-9;
            lambda[static_cast<std::size_t>(QueueEventType::SellCancel)] = (q_ask > 0.0) ? (0.5 * cell.lambda_cancel_per_us * (1.0 + 0.2 * std::log1p(q_ask))) : 1e-9;
            lambda[static_cast<std::size_t>(QueueEventType::BuyMarket)]  = 0.5 * cell.lambda_exec_per_us * imb_skew_ask;
            lambda[static_cast<std::size_t>(QueueEventType::SellMarket)] = 0.5 * cell.lambda_exec_per_us * imb_skew_bid;
        } else {
            // HLR (2015) Model IIb queue-reactive baseline intensities with opposite-queue coupling
            // and logistic imbalance / ergodic damping (`theta_ergodic > 0` per HLR Theorem 2.1).
            const double opp_bid_mod = 1.0 + cfg_.opposite_queue_beta * std::tanh(q_ask / 200.0);
            const double opp_ask_mod = 1.0 + cfg_.opposite_queue_beta * std::tanh(q_bid / 200.0);
            const double imb_add_bid = std::exp(-cfg_.imbalance_sensitivity * imb);
            const double imb_add_ask = std::exp(+cfg_.imbalance_sensitivity * imb);
            const double drift_shift = ou_alpha_state_ + current_transient_impact_locked();
            const double drift_buy_mod  = std::clamp(1.0 + drift_shift, 0.2, 5.0);
            const double drift_sell_mod = std::clamp(1.0 - drift_shift, 0.2, 5.0);

            lambda[static_cast<std::size_t>(QueueEventType::BuyLimit)] =
                (cfg_.base_limit_rate * opp_bid_mod * imb_add_bid * drift_buy_mod) /
                (1.0 + cfg_.limit_depth_decay * std::pow(q_bid + 1.0, cfg_.limit_gamma) + cfg_.theta_ergodic * q_bid);
            lambda[static_cast<std::size_t>(QueueEventType::SellLimit)] =
                (cfg_.base_limit_rate * opp_ask_mod * imb_add_ask * drift_sell_mod) /
                (1.0 + cfg_.limit_depth_decay * std::pow(q_ask + 1.0, cfg_.limit_gamma) + cfg_.theta_ergodic * q_ask);

            lambda[static_cast<std::size_t>(QueueEventType::BuyCancel)] =
                cfg_.base_cancel_rate * (std::pow(q_bid, cfg_.cancel_delta) + cfg_.theta_ergodic * q_bid * q_bid * 0.01) * opp_bid_mod;
            lambda[static_cast<std::size_t>(QueueEventType::SellCancel)] =
                cfg_.base_cancel_rate * (std::pow(q_ask, cfg_.cancel_delta) + cfg_.theta_ergodic * q_ask * q_ask * 0.01) * opp_ask_mod;

            lambda[static_cast<std::size_t>(QueueEventType::BuyMarket)] =
                cfg_.base_market_rate * (1.0 + cfg_.market_affinity * std::log1p(q_ask)) * imb_add_ask * drift_buy_mod;
            lambda[static_cast<std::size_t>(QueueEventType::SellMarket)] =
                cfg_.base_market_rate * (1.0 + cfg_.market_affinity * std::log1p(q_bid)) * imb_add_bid * drift_sell_mod;
        }

        // Add Hawkes self- and cross-excitation state ( already scaled by hawkes_scale )
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            double exc = 0.0;
            for (std::size_t j = 0; j < kEventTypes; ++j) {
                exc += state_[i][j];
            }
            lambda[i] += exc;
            if (lambda[i] < 1e-9) lambda[i] = 1e-9;
        }

        return lambda;
    }

    // Compute multi-level (K <= 5) HLR Model IIb/III + Inside-Spread Create intensities
    MultiLevelEventRates compute_multilevel_intensities(
        const std::array<Qty, kMaxQrLevels>& bid_vols,
        const std::array<Qty, kMaxQrLevels>& ask_vols,
        Tick spread_ticks = 1) const noexcept {
        MultiLevelEventRates out{};
        out.num_levels = std::max<std::size_t>(1, std::min<std::size_t>(kMaxQrLevels, cfg_.num_levels));

        const auto l1 = compute_intensities(bid_vols[0], ask_vols[0], spread_ticks);
        out.buy_limit[0]   = l1[static_cast<std::size_t>(QueueEventType::BuyLimit)];
        out.sell_limit[0]  = l1[static_cast<std::size_t>(QueueEventType::SellLimit)];
        out.buy_cancel[0]  = l1[static_cast<std::size_t>(QueueEventType::BuyCancel)];
        out.sell_cancel[0] = l1[static_cast<std::size_t>(QueueEventType::SellCancel)];
        out.buy_market     = l1[static_cast<std::size_t>(QueueEventType::BuyMarket)];
        out.sell_market    = l1[static_cast<std::size_t>(QueueEventType::SellMarket)];

        const double q1_bid = static_cast<double>(std::max<Qty>(0, bid_vols[0]));
        const double q1_ask = static_cast<double>(std::max<Qty>(0, ask_vols[0]));
        const double tot_l1 = q1_bid + q1_ask;
        const double imb_l1 = (tot_l1 > 0.0) ? ((q1_bid - q1_ask) / tot_l1) : 0.0;
        const double opp_bid_mod = 1.0 + cfg_.opposite_queue_beta * std::tanh(q1_ask / 200.0);
        const double opp_ask_mod = 1.0 + cfg_.opposite_queue_beta * std::tanh(q1_bid / 200.0);
        const double imb_add_bid = std::exp(-cfg_.imbalance_sensitivity * imb_l1);
        const double imb_add_ask = std::exp(+cfg_.imbalance_sensitivity * imb_l1);

        for (std::size_t m = 1; m < out.num_levels; ++m) {
            const double qb = static_cast<double>(std::max<Qty>(0, bid_vols[m]));
            const double qa = static_cast<double>(std::max<Qty>(0, ask_vols[m]));
            const double lim_mult = cfg_.level_limit_multipliers[m];
            const double can_mult = cfg_.level_cancel_multipliers[m];

            if (cfg_.use_empirical_3d_table) {
                const Empirical3DCell& cell = table_3d_.lookup(bid_vols[0], ask_vols[0], spread_ticks);
                const double imb_skew_bid = std::exp(-0.25 * imb_l1);
                const double imb_skew_ask = std::exp(+0.25 * imb_l1);
                out.buy_limit[m] = std::max(1e-9, 0.5 * cell.lambda_add_per_us * lim_mult * imb_skew_bid);
                out.sell_limit[m] = std::max(1e-9, 0.5 * cell.lambda_add_per_us * lim_mult * imb_skew_ask);
                out.buy_cancel[m] = (qb > 0.0)
                    ? std::max(1e-9, 0.5 * cell.lambda_cancel_per_us * can_mult * (1.0 + 0.2 * std::log1p(qb)))
                    : 1e-9;
                out.sell_cancel[m] = (qa > 0.0)
                    ? std::max(1e-9, 0.5 * cell.lambda_cancel_per_us * can_mult * (1.0 + 0.2 * std::log1p(qa)))
                    : 1e-9;
            } else {
                out.buy_limit[m] = std::max(1e-9,
                    (cfg_.base_limit_rate * lim_mult * opp_bid_mod * imb_add_bid) /
                    (1.0 + cfg_.limit_depth_decay * std::pow(qb + 1.0, cfg_.limit_gamma) + cfg_.theta_ergodic * qb));
                out.sell_limit[m] = std::max(1e-9,
                    (cfg_.base_limit_rate * lim_mult * opp_ask_mod * imb_add_ask) /
                    (1.0 + cfg_.limit_depth_decay * std::pow(qa + 1.0, cfg_.limit_gamma) + cfg_.theta_ergodic * qa));

                out.buy_cancel[m]  = std::max(1e-9,
                    cfg_.base_cancel_rate * can_mult * (std::pow(qb, cfg_.cancel_delta) + cfg_.theta_ergodic * qb * qb * 0.01) * opp_bid_mod);
                out.sell_cancel[m] = std::max(1e-9,
                    cfg_.base_cancel_rate * can_mult * (std::pow(qa, cfg_.cancel_delta) + cfg_.theta_ergodic * qa * qa * 0.01) * opp_ask_mod);
            }
        }

        if (cfg_.enable_inside_spread_orders && spread_ticks > 1) {
            if (cfg_.use_empirical_3d_table) {
                const Empirical3DCell& cell = table_3d_.lookup(bid_vols[0], ask_vols[0], spread_ticks);
                out.create_bid = 0.5 * std::max(0.005, cell.lambda_create_per_us);
                out.create_ask = 0.5 * std::max(0.005, cell.lambda_create_per_us);
            } else {
                const double spread_boost = 0.45 * static_cast<double>(spread_ticks - 1);
                out.create_bid = cfg_.base_limit_rate * spread_boost;
                out.create_ask = cfg_.base_limit_rate * spread_boost;
            }
        }
        return out;
    }

    // Zero-libm Q48.16 + SWAR LUT intensity evaluation (`OnlineHawkesQ16` + `SwarQueueReactiveRegimeLut`).
    [[gnu::hot]] std::array<std::uint64_t, kEventTypes> compute_intensities_q16(
        Qty bid_volume, Qty ask_volume) const noexcept {
        std::array<std::uint64_t, kEventTypes> lambda_q16{};
        const std::uint32_t bid_reg = SwarQueueReactiveRegimeLut::classify_regime(bid_volume, 0);
        const std::uint32_t ask_reg = SwarQueueReactiveRegimeLut::classify_regime(0, ask_volume);

        const std::uint64_t add_b = SwarQueueReactiveRegimeLut::add_nibble(bid_reg);
        const std::uint64_t add_a = SwarQueueReactiveRegimeLut::add_nibble(ask_reg);
        const std::uint64_t can_b = SwarQueueReactiveRegimeLut::cancel_nibble(bid_reg);
        const std::uint64_t can_a = SwarQueueReactiveRegimeLut::cancel_nibble(ask_reg);
        const std::uint64_t exe_b = SwarQueueReactiveRegimeLut::exec_nibble(ask_reg);
        const std::uint64_t exe_a = SwarQueueReactiveRegimeLut::exec_nibble(bid_reg);

        lambda_q16[static_cast<std::size_t>(QueueEventType::BuyLimit)] =
            std::max<std::uint64_t>(1ULL, ((base_limit_q16_ * add_b) >> 2ULL) + state_q16_[0]);
        lambda_q16[static_cast<std::size_t>(QueueEventType::SellLimit)] =
            std::max<std::uint64_t>(1ULL, ((base_limit_q16_ * add_a) >> 2ULL) + state_q16_[1]);
        lambda_q16[static_cast<std::size_t>(QueueEventType::BuyCancel)] =
            std::max<std::uint64_t>(1ULL, (base_cancel_q16_ * (1ULL + can_b)) + state_q16_[2]);
        lambda_q16[static_cast<std::size_t>(QueueEventType::SellCancel)] =
            std::max<std::uint64_t>(1ULL, (base_cancel_q16_ * (1ULL + can_a)) + state_q16_[3]);
        lambda_q16[static_cast<std::size_t>(QueueEventType::BuyMarket)] =
            std::max<std::uint64_t>(1ULL, ((base_market_q16_ * (4ULL + exe_b)) >> 2ULL) + state_q16_[4]);
        lambda_q16[static_cast<std::size_t>(QueueEventType::SellMarket)] =
            std::max<std::uint64_t>(1ULL, ((base_market_q16_ * (4ULL + exe_a)) >> 2ULL) + state_q16_[5]);

        return lambda_q16;
    }

    // Zero-libm fast event generator powered by `OnlineHawkesQ16` and `SwarQueueReactiveRegimeLut`.
    [[gnu::hot]] GeneratedEvent next_event_q16(Tick best_bid_price, Qty best_bid_vol,
                                               Tick best_ask_price, Qty best_ask_vol) noexcept {
        const auto lambda_q16 = compute_intensities_q16(best_bid_vol, best_ask_vol);
        std::uint64_t total_q16 = 0ULL;
        for (std::uint64_t r : lambda_q16) total_q16 += r;
        if (total_q16 == 0ULL) total_q16 = 1ULL;

        // Deterministic integer inter-arrival sampling without std::log / std::exp:
        // sample u_q16 in [1, 65535] via eval_neg_ln_u_q16_from_raw (shift-UB free).
        const std::uint64_t raw = rng_.next_u64();
        const std::uint64_t neg_ln_u_q16 = eval_neg_ln_u_q16_from_raw(raw);
        const std::uint64_t dt_ns_u64 = std::max<std::uint64_t>(
            1ULL, (neg_ln_u_q16 * 1000ULL) / total_q16);
        const TimeNs dt_ns = static_cast<TimeNs>(dt_ns_u64);

        // Decay per-channel Q16 excitation state via Horner Q48.16 exponential kernel
        const std::uint64_t dt_scaled_q16 = (dt_ns_u64 * beta_per_us_q16_) / 1000ULL;
        const std::uint64_t decay_q16 = eval_hawkes_q16_horner_decay(dt_scaled_q16);
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            state_q16_[i] = (state_q16_[i] * decay_q16 + 32768ULL) >> 16ULL;
        }
        current_time_ns_ += dt_ns;

        // Categorical event selection in integer Q16 space
        const std::uint64_t pick_q16 = ((raw >> 16ULL) % total_q16);
        std::uint64_t accum_q16 = 0ULL;
        QueueEventType selected = QueueEventType::BuyLimit;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            accum_q16 += lambda_q16[i];
            if (pick_q16 < accum_q16 || i == kEventTypes - 1) {
                selected = static_cast<QueueEventType>(i);
                break;
            }
        }

        const std::size_t sel_idx = static_cast<std::size_t>(selected);
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            state_q16_[i] += alpha_q16_[i][sel_idx];
        }
        const std::uint32_t regime = SwarQueueReactiveRegimeLut::classify_regime(best_bid_vol, best_ask_vol);
        (void)hawkes_q16_.record_event(current_time_ns_, regime);

        const Qty qty = sample_order_qty(0);
        return build_generated_event(selected, qty, dt_ns, best_bid_price, best_ask_price, 1);
    }

    // Advance time and generate the next L1 market event
    GeneratedEvent next_event(Tick best_bid_price, Qty best_bid_vol,
                             Tick best_ask_price, Qty best_ask_vol) {
        const Tick spread_ticks = std::max<Tick>(1, best_ask_price - best_bid_price);
        if (cfg_.num_levels > 1 || cfg_.enable_inside_spread_orders) {
            std::array<Qty, kMaxQrLevels> b_vols{};
            std::array<Qty, kMaxQrLevels> a_vols{};
            b_vols[0] = best_bid_vol;
            a_vols[0] = best_ask_vol;
            for (std::size_t m = 1; m < cfg_.num_levels; ++m) {
                b_vols[m] = best_bid_vol;
                a_vols[m] = best_ask_vol;
            }
            return next_event_multilevel(best_bid_price, best_ask_price, b_vols, a_vols);
        }

        const auto lambda = compute_intensities(best_bid_vol, best_ask_vol, spread_ticks);
        double total_rate = 0.0;
        for (double r : lambda) total_rate += r;

        // Sample inter-arrival time: delta_t ~ Exp(total_rate)
        const double u = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
        const double dt_us = -std::log(u) / total_rate;
        const TimeNs dt_ns = std::max<TimeNs>(1, static_cast<TimeNs>(dt_us * 1000.0));

        // Exact analytical integration of Hawkes conditional intensity over [t, t + dt]
        // for Ogata's time-rescaling theorem residual:
        // \tau_k = \mu_0 * dt + \sum_{i,j} S_{i,j} * (1 - exp(-\beta_{i,j} * dt)) / \beta_{i,j}
        const double rescaled_tau = advance_and_integrate_state(lambda, dt_ns, dt_us, best_bid_vol, best_ask_vol);

        // Sample which event occurred according to post-decay intensity
        const auto post_lambda = compute_intensities(best_bid_vol, best_ask_vol, spread_ticks);
        double post_total = 0.0;
        for (double r : post_lambda) post_total += r;

        const double pick = distributions::next_unit_double(rng_) * post_total;
        double accum = 0.0;
        QueueEventType selected = QueueEventType::BuyLimit;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            accum += post_lambda[i];
            if (pick <= accum || i == kEventTypes - 1) {
                selected = static_cast<QueueEventType>(i);
                break;
            }
        }

        // Excite Hawkes state with the selected event (scaled by hawkes_.hawkes_scale)
        excite_hawkes(selected);

        const Qty qty = sample_order_qty(0);
        GeneratedEvent ev = build_generated_event(selected, qty, dt_ns, best_bid_price, best_ask_price, 1);
        ev.rescaled_tau = rescaled_tau;
        apply_hlr_model_iii_depletion_and_impact(ev, best_bid_vol, best_ask_vol);
        return ev;
    }

    // Exact Ogata Thinning Generator for State-Dependent Hawkes + Queue-Reactive Point Process.
    // Proposes candidate arrivals from upper bound `\Lambda^*(t)`, decays state to candidate time,
    // accepts with probability `\Lambda(t_cand) / \Lambda^*`, and accumulates the exact
    // integrated conditional intensity residual `\tau_k = \int_{t_{k-1}}^{t_k} \Lambda(s) ds ~ Exp(1)`.
    GeneratedEvent next_event_ogata_thinning(Tick best_bid_price, Qty best_bid_vol,
                                             Tick best_ask_price, Qty best_ask_vol) {
        const Tick spread_ticks = std::max<Tick>(1, best_ask_price - best_bid_price);
        double integrated_tau = 0.0;
        TimeNs total_dt_ns = 0;
        QueueEventType selected = QueueEventType::BuyLimit;

        for (int step = 0; step < 64; ++step) {
            const auto lambda_upper = compute_intensities(best_bid_vol, best_ask_vol, spread_ticks);
            double upper_rate = 0.0;
            for (double r : lambda_upper) upper_rate += r;
            upper_rate = std::max(1e-9, upper_rate);

            const double u1 = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
            const double dt_us = -std::log(u1) / upper_rate;
            const TimeNs dt_ns = std::max<TimeNs>(1, static_cast<TimeNs>(dt_us * 1000.0));

            integrated_tau += advance_and_integrate_state(lambda_upper, dt_ns, dt_us, best_bid_vol, best_ask_vol, false);
            total_dt_ns += dt_ns;

            const auto lambda_cand = compute_intensities(best_bid_vol, best_ask_vol, spread_ticks);
            double cand_rate = 0.0;
            for (double r : lambda_cand) cand_rate += r;

            const double u2 = distributions::next_unit_double(rng_) * upper_rate;
            if (u2 <= cand_rate || step == 63) {
                double accum = 0.0;
                for (std::size_t i = 0; i < kEventTypes; ++i) {
                    accum += lambda_cand[i];
                    if (u2 <= accum || i == kEventTypes - 1) {
                        selected = static_cast<QueueEventType>(i);
                        break;
                    }
                }
                break;
            }
        }

        if (cfg_.record_ogata_residuals && ogata_residuals_.size() < 100000U) {
            ogata_residuals_.push_back(integrated_tau);
        }
        excite_hawkes(selected);
        const Qty qty = sample_order_qty(0);
        GeneratedEvent ev = build_generated_event(selected, qty, total_dt_ns, best_bid_price, best_ask_price, 1);
        ev.rescaled_tau = integrated_tau;
        apply_hlr_model_iii_depletion_and_impact(ev, best_bid_vol, best_ask_vol);
        return ev;
    }

    // Multi-level (K <= 5) HLR Model IIb/III event generator with inside-spread CreateBid/CreateAsk
    // and stationary invariant distribution queue redraws.
    GeneratedEvent next_event_multilevel(Tick best_bid_price,
                                         Tick best_ask_price,
                                         const std::array<Qty, kMaxQrLevels>& bid_vols,
                                         const std::array<Qty, kMaxQrLevels>& ask_vols) {
        const Tick spread_ticks = std::max<Tick>(1, best_ask_price - best_bid_price);
        const MultiLevelEventRates rates = compute_multilevel_intensities(bid_vols, ask_vols, spread_ticks);
        const double total_rate = rates.total_rate();

        const double u = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
        const double dt_us = -std::log(u) / total_rate;
        const TimeNs dt_ns = std::max<TimeNs>(1, static_cast<TimeNs>(dt_us * 1000.0));

        const auto l1_lambda = compute_intensities(bid_vols[0], ask_vols[0], spread_ticks);
        const double l1_sum = l1_lambda[0] + l1_lambda[1] + l1_lambda[2] + l1_lambda[3] + l1_lambda[4] + l1_lambda[5];
        const double extra_rate = std::max(0.0, total_rate - l1_sum);
        const double rescaled_tau = advance_and_integrate_state(l1_lambda, dt_ns, dt_us, bid_vols[0], ask_vols[0])
                                  + extra_rate * dt_us;

        // Categorical draw across 4*K + 4 channels
        const double pick = distributions::next_unit_double(rng_) * total_rate;
        double accum = 0.0;
        QueueEventType selected = QueueEventType::BuyLimit;
        std::uint8_t sel_level = 1;

        bool chosen = false;
        for (std::size_t m = 0; m < rates.num_levels && !chosen; ++m) {
            const auto lvl = static_cast<std::uint8_t>(m + 1);
            accum += rates.buy_limit[m];
            if (pick <= accum) { selected = QueueEventType::BuyLimit; sel_level = lvl; chosen = true; break; }
            accum += rates.sell_limit[m];
            if (pick <= accum) { selected = QueueEventType::SellLimit; sel_level = lvl; chosen = true; break; }
            accum += rates.buy_cancel[m];
            if (pick <= accum) { selected = QueueEventType::BuyCancel; sel_level = lvl; chosen = true; break; }
            accum += rates.sell_cancel[m];
            if (pick <= accum) { selected = QueueEventType::SellCancel; sel_level = lvl; chosen = true; break; }
        }
        if (!chosen) {
            accum += rates.buy_market;
            if (pick <= accum) {
                selected = QueueEventType::BuyMarket;
                sel_level = 1;
            } else {
                accum += rates.sell_market;
                if (pick <= accum) {
                    selected = QueueEventType::SellMarket;
                    sel_level = 1;
                } else {
                    accum += rates.create_bid;
                    if (pick <= accum) {
                        selected = QueueEventType::CreateBid;
                        sel_level = 0;
                    } else {
                        selected = QueueEventType::CreateAsk;
                        sel_level = 0;
                    }
                }
            }
        }

        // Map CreateBid/CreateAsk to BuyLimit/SellLimit for 6x6 Hawkes excitation
        const QueueEventType hawkes_ev =
            (selected == QueueEventType::CreateBid) ? QueueEventType::BuyLimit :
            (selected == QueueEventType::CreateAsk) ? QueueEventType::SellLimit : selected;
        excite_hawkes(hawkes_ev);

        const std::size_t lvl_idx = (sel_level > 0) ? static_cast<std::size_t>(sel_level - 1) : 0U;
        const Qty qty = sample_order_qty(lvl_idx);
        GeneratedEvent ev = build_generated_event(selected, qty, dt_ns, best_bid_price, best_ask_price, sel_level);
        ev.rescaled_tau = rescaled_tau;
        apply_hlr_model_iii_depletion_and_impact(ev, bid_vols[0], ask_vols[0]);
        return ev;
    }

    // Sample a stationary queue depth `q_i` (in lots) from the HLR birth-death invariant
    // distribution `\pi_i(n) \propto \prod_{k=1}^n \frac{\lambda_i(k-1)}{\mu_i^{cancel}(k) + \mu_i^{exec}(k)}`
    // scaled by `level_mes[level_idx]`.
    Qty sample_stationary_queue_depth(std::size_t level_idx = 0) noexcept {
        const std::size_t m = std::min<std::size_t>(kMaxQrLevels - 1, level_idx);
        const double u = distributions::next_unit_double(rng_);
        const auto& cdf = stationary_cdf_[m];
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        const std::size_t units = static_cast<std::size_t>(std::distance(cdf.begin(), it)) + 1U;
        const Qty mes = std::max<Qty>(1, cfg_.level_mes[m]);
        return static_cast<Qty>(units) * mes;
    }

    // Analytical birth-death invariant PMF `\pi_i(n)` for `n in [1..kMaxStationaryUnits]` (in MES units)
    std::array<double, 64> stationary_queue_pmf(std::size_t level_idx = 0) const noexcept {
        const std::size_t m = std::min<std::size_t>(kMaxQrLevels - 1, level_idx);
        std::array<double, 64> pmf{};
        double prev = 0.0;
        for (std::size_t k = 0; k < 64; ++k) {
            pmf[k] = stationary_cdf_[m][k] - prev;
            prev = stationary_cdf_[m][k];
        }
        return pmf;
    }

    // Sample order size according to configured `OrderSizeModel`
    Qty sample_order_qty(std::size_t level_idx = 0) noexcept {
        const std::size_t m = std::min<std::size_t>(kMaxQrLevels - 1, level_idx);
        const Qty min_q = std::max<Qty>(1, cfg_.min_qty);
        const Qty max_q = std::max<Qty>(min_q, cfg_.max_qty);

        if (cfg_.size_model == OrderSizeModel::Uniform) {
            return static_cast<Qty>(min_q +
                (rng_.next_u64() % static_cast<std::uint64_t>(max_q - min_q + 1)));
        }
        if (cfg_.size_model == OrderSizeModel::Geometric) {
            const double p = std::clamp(cfg_.geometric_p, 0.02, 0.95);
            const double u = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
            const auto units = static_cast<Qty>(1 + std::floor(std::log(u) / std::log(1.0 - p)));
            const Qty unit_lot = std::max<Qty>(1, cfg_.level_mes[m] / 4);
            return std::clamp<Qty>(units * unit_lot, min_q, max_q * 5);
        }
        // OrderSizeModel::LogBinnedEmpirical
        const double u = distributions::next_unit_double(rng_);
        double accum = 0.0;
        std::size_t bin = 0;
        for (std::size_t b = 0; b < cfg_.empirical_size_weights.size(); ++b) {
            accum += cfg_.empirical_size_weights[b];
            if (u <= accum || b + 1 == cfg_.empirical_size_weights.size()) {
                bin = b;
                break;
            }
        }
        const Qty base_unit = std::max<Qty>(1, cfg_.level_mes[m] / 4);
        const Qty lo = static_cast<Qty>(base_unit * (1 << bin));
        const Qty hi = static_cast<Qty>(base_unit * (1 << (bin + 1)));
        const Qty span = std::max<Qty>(1, hi - lo);
        const Qty raw_sz = static_cast<Qty>(lo + static_cast<Qty>(rng_.next_u64() % static_cast<std::uint64_t>(span)));
        return std::clamp<Qty>(raw_sz, min_q, max_q * 10);
    }

    static constexpr std::uint64_t eval_neg_ln_u_q16_from_raw(std::uint64_t raw) noexcept {
        return ::sim::eval_neg_ln_u_q16_from_raw(raw);
    }

    // Overload accepting (best_bid_price, bid_vols, best_ask_price, ask_vols)
    GeneratedEvent next_event_multilevel(Tick best_bid_price,
                                         const std::array<Qty, kMaxQrLevels>& bid_vols,
                                         Tick best_ask_price,
                                         const std::array<Qty, kMaxQrLevels>& ask_vols) {
        return next_event_multilevel(best_bid_price, best_ask_price, bid_vols, ask_vols);
    }

    // Multi-level (K <= 5) exact Ogata thinning generator across all 4*K + 4 event channels
    GeneratedEvent next_event_ogata_thinning(Tick best_bid_price,
                                             const std::array<Qty, kMaxQrLevels>& bid_vols,
                                             Tick best_ask_price,
                                             const std::array<Qty, kMaxQrLevels>& ask_vols) {
        if (cfg_.num_levels <= 1 && !cfg_.enable_inside_spread_orders) {
            return next_event_ogata_thinning(best_bid_price, bid_vols[0], best_ask_price, ask_vols[0]);
        }
        const Tick spread_ticks = std::max<Tick>(1, best_ask_price - best_bid_price);
        double integrated_tau = 0.0;
        TimeNs total_dt_ns = 0;
        QueueEventType selected = QueueEventType::BuyLimit;
        std::uint8_t sel_level = 1;

        for (int step = 0; step < 64; ++step) {
            const MultiLevelEventRates rates_upper = compute_multilevel_intensities(bid_vols, ask_vols, spread_ticks);
            const double upper_rate = rates_upper.total_rate();
            const auto l1_upper = compute_intensities(bid_vols[0], ask_vols[0], spread_ticks);
            const double l1_upper_sum = l1_upper[0] + l1_upper[1] + l1_upper[2] + l1_upper[3] + l1_upper[4] + l1_upper[5];
            const double extra_upper = std::max(0.0, upper_rate - l1_upper_sum);

            const double u1 = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
            const double dt_us = -std::log(u1) / upper_rate;
            const TimeNs dt_ns = std::max<TimeNs>(1, static_cast<TimeNs>(dt_us * 1000.0));

            integrated_tau += advance_and_integrate_state(l1_upper, dt_ns, dt_us, bid_vols[0], ask_vols[0], false)
                            + extra_upper * dt_us;
            total_dt_ns += dt_ns;

            const MultiLevelEventRates rates_cand = compute_multilevel_intensities(bid_vols, ask_vols, spread_ticks);
            const double cand_rate = rates_cand.total_rate();

            const double u2 = distributions::next_unit_double(rng_) * upper_rate;
            if (u2 <= cand_rate || step == 63) {
                const double pick = (u2 <= cand_rate) ? u2 : (distributions::next_unit_double(rng_) * cand_rate);
                double accum = 0.0;
                bool chosen = false;
                for (std::size_t m = 0; m < rates_cand.num_levels && !chosen; ++m) {
                    const auto lvl = static_cast<std::uint8_t>(m + 1);
                    accum += rates_cand.buy_limit[m];
                    if (pick <= accum) { selected = QueueEventType::BuyLimit; sel_level = lvl; chosen = true; break; }
                    accum += rates_cand.sell_limit[m];
                    if (pick <= accum) { selected = QueueEventType::SellLimit; sel_level = lvl; chosen = true; break; }
                    accum += rates_cand.buy_cancel[m];
                    if (pick <= accum) { selected = QueueEventType::BuyCancel; sel_level = lvl; chosen = true; break; }
                    accum += rates_cand.sell_cancel[m];
                    if (pick <= accum) { selected = QueueEventType::SellCancel; sel_level = lvl; chosen = true; break; }
                }
                if (!chosen) {
                    accum += rates_cand.buy_market;
                    if (pick <= accum) {
                        selected = QueueEventType::BuyMarket;
                        sel_level = 1;
                    } else {
                        accum += rates_cand.sell_market;
                        if (pick <= accum) {
                            selected = QueueEventType::SellMarket;
                            sel_level = 1;
                        } else {
                            accum += rates_cand.create_bid;
                            if (pick <= accum) {
                                selected = QueueEventType::CreateBid;
                                sel_level = 0;
                            } else {
                                selected = QueueEventType::CreateAsk;
                                sel_level = 0;
                            }
                        }
                    }
                }
                break;
            }
        }

        if (cfg_.record_ogata_residuals && ogata_residuals_.size() < 100000U) {
            ogata_residuals_.push_back(integrated_tau);
        }
        const QueueEventType hawkes_ev =
            (selected == QueueEventType::CreateBid) ? QueueEventType::BuyLimit :
            (selected == QueueEventType::CreateAsk) ? QueueEventType::SellLimit : selected;
        excite_hawkes(hawkes_ev);

        const std::size_t lvl_idx = (sel_level > 0) ? static_cast<std::size_t>(sel_level - 1) : 0U;
        const Qty qty = sample_order_qty(lvl_idx);
        GeneratedEvent ev = build_generated_event(selected, qty, total_dt_ns, best_bid_price, best_ask_price, sel_level);
        ev.rescaled_tau = integrated_tau;
        apply_hlr_model_iii_depletion_and_impact(ev, bid_vols[0], ask_vols[0]);
        return ev;
    }

    void set_empirical_3d_table(const Empirical3DIntensityTable& table) noexcept {
        table_3d_ = table;
        cfg_.use_empirical_3d_table = true;
    }

    double last_total_intensity() const noexcept {
        const auto lam = compute_intensities(100, 100, 2);
        double s = 0.0;
        for (double v : lam) s += v;
        return s;
    }

    TimeNs current_time() const noexcept { return current_time_ns_; }
    const OnlineHawkesQ16& hawkes_q16() const noexcept { return hawkes_q16_; }
    OnlineHawkesQ16& hawkes_q16() noexcept { return hawkes_q16_; }
    const std::vector<double>& ogata_residuals() const noexcept { return ogata_residuals_; }
    void clear_ogata_residuals() noexcept { ogata_residuals_.clear(); }
    double current_ou_alpha() const noexcept { return ou_alpha_state_; }
    double current_transient_impact() const noexcept { return current_transient_impact_locked(); }
    std::uint64_t reprice_count() const noexcept { return reprice_count_; }

private:
    static constexpr std::size_t kMaxStationaryUnits = 64;
    static constexpr std::size_t kImpactWindowCap = 64;

    double current_transient_impact_locked() const noexcept {
        if (!cfg_.enable_transient_impact || impact_count_ == 0) return 0.0;
        const double now_us = static_cast<double>(current_time_ns_) * 1e-3;
        double sum = 0.0;
        const std::size_t n = std::min<std::size_t>(impact_count_, kImpactWindowCap);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& ev = impact_ring_[i];
            const double dt_us = std::max(0.0, ( static_cast<double>(current_time_ns_) - static_cast<double>(ev.ts_ns) ) * 1e-3);
            sum += ev.signed_vol * std::pow(dt_us + std::max(1e-6, cfg_.impact_tau0_us), -cfg_.impact_gamma);
        }
        return sum;
    }

    double advance_and_integrate_state(const std::array<double, kEventTypes>& pre_lambda,
                                       TimeNs dt_ns, double dt_us,
                                       Qty best_bid_vol, Qty best_ask_vol,
                                       bool record_residual = true) {
        double state_sum_pre = 0.0;
        double integrated_exc_us = 0.0;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            for (std::size_t j = 0; j < kEventTypes; ++j) {
                const double s_ij = state_[i][j];
                state_sum_pre += s_ij;
                const double beta_ns = hawkes_.beta[i][j];
                const double decay = (beta_ns > 0.0) ? std::exp(-beta_ns * static_cast<double>(dt_ns)) : 1.0;
                if (beta_ns > 1e-15) {
                    const double beta_us = beta_ns * 1000.0;
                    integrated_exc_us += s_ij * (1.0 - decay) / beta_us;
                } else {
                    integrated_exc_us += s_ij * dt_us;
                }
                state_[i][j] = s_ij * decay;
            }
        }
        double total_pre = 0.0;
        for (double r : pre_lambda) total_pre += r;
        const double base_sum = std::max(0.0, total_pre - state_sum_pre);
        const double tau_k = base_sum * dt_us + integrated_exc_us;

        if (cfg_.enable_ou_alpha_drift) {
            const double decay_ou = std::exp(-cfg_.ou_kappa_per_us * dt_us);
            // Box-Muller / Irwin-Hall normal approximation from PRNG
            const double u1 = distributions::next_unit_double(rng_);
            const double u2 = distributions::next_unit_double(rng_);
            const double z = (u1 + u2 - 1.0) * 2.449489742783178; // unit variance symmetric draw
            ou_alpha_state_ = ou_alpha_state_ * decay_ou + cfg_.ou_sigma_per_sqrt_us * std::sqrt(dt_us) * z;
        }

        current_time_ns_ += dt_ns;
        const std::uint32_t regime = SwarQueueReactiveRegimeLut::classify_regime(best_bid_vol, best_ask_vol);
        (void)hawkes_q16_.record_event(current_time_ns_, regime);

        if (record_residual && cfg_.record_ogata_residuals && ogata_residuals_.size() < 100000U) {
            ogata_residuals_.push_back(tau_k);
        }
        return tau_k;
    }

    void excite_hawkes(QueueEventType selected) noexcept {
        const std::size_t sel_idx = static_cast<std::size_t>(selected) % kEventTypes;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            state_[i][sel_idx] += hawkes_.hawkes_scale * hawkes_.alpha[i][sel_idx];
        }
    }

    void apply_hlr_model_iii_depletion_and_impact(GeneratedEvent& ev, Qty bid_vol, Qty ask_vol) noexcept {
        if (cfg_.enable_transient_impact && ev.order_type == OrderType::Market) {
            const double sign = is_buy(ev.side) ? +1.0 : -1.0;
            const double mes0 = static_cast<double>(std::max<Qty>(1, cfg_.level_mes[0]));
            const double impulse = sign * cfg_.impact_eta * std::pow(static_cast<double>(ev.qty) / mes0, cfg_.impact_delta);
            impact_ring_[impact_count_ % kImpactWindowCap] = TransientImpactKernelEvent{current_time_ns_, impulse};
            ++impact_count_;
        }

        // HLR Model III reference-price shift when an execution or cancellation depletes the best queue
        const bool depletes_bid =
            (ev.type == QueueEventType::SellMarket || (ev.type == QueueEventType::BuyCancel && ev.level == 1)) &&
            (bid_vol > 0 && ev.qty >= bid_vol);
        const bool depletes_ask =
            (ev.type == QueueEventType::BuyMarket || (ev.type == QueueEventType::SellCancel && ev.level == 1)) &&
            (ask_vol > 0 && ev.qty >= ask_vol);

        if ((depletes_bid || depletes_ask) && cfg_.theta_reprice > 0.0) {
            if (distributions::next_unit_double(rng_) < cfg_.theta_reprice) {
                ev.reprice_triggered = true;
                ev.reprice_ticks = depletes_ask ? static_cast<std::int8_t>(+1) : static_cast<std::int8_t>(-1);
                ++reprice_count_;
            }
        }
    }

    static constexpr GeneratedEvent build_generated_event(
        QueueEventType selected, Qty qty, TimeNs dt_ns,
        Tick best_bid_price, Tick best_ask_price, std::uint8_t level = 1) noexcept {
        GeneratedEvent ev{};
        ev.type = selected;
        ev.qty = qty;
        ev.inter_arrival_ns = dt_ns;
        ev.level = level;
        const Tick offset = (level > 1) ? static_cast<Tick>(level - 1) : 0;

        switch (selected) {
            case QueueEventType::BuyLimit:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Limit;
                ev.price = best_bid_price - offset;
                break;
            case QueueEventType::SellLimit:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Limit;
                ev.price = best_ask_price + offset;
                break;
            case QueueEventType::BuyCancel:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Cancel;
                ev.price = best_bid_price - offset;
                break;
            case QueueEventType::SellCancel:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Cancel;
                ev.price = best_ask_price + offset;
                break;
            case QueueEventType::BuyMarket:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Market;
                ev.price = 0; // sweeps ask
                break;
            case QueueEventType::SellMarket:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Market;
                ev.price = 0; // sweeps bid
                break;
            case QueueEventType::CreateBid:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Limit;
                ev.price = (best_ask_price > best_bid_price + 1) ? (best_bid_price + 1) : best_bid_price;
                ev.level = 0;
                break;
            case QueueEventType::CreateAsk:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Limit;
                ev.price = (best_ask_price > best_bid_price + 1) ? (best_ask_price - 1) : best_ask_price;
                ev.level = 0;
                break;
            default:
                break;
        }
        return ev;
    }

    void init_q16_tables() noexcept {
        base_limit_q16_ = static_cast<std::uint64_t>(
            std::max<long long>(1LL, std::llround(cfg_.base_limit_rate * static_cast<double>(kQ16One))));
        base_cancel_q16_ = static_cast<std::uint64_t>(
            std::max<long long>(1LL, std::llround(cfg_.base_cancel_rate * static_cast<double>(kQ16One))));
        base_market_q16_ = static_cast<std::uint64_t>(
            std::max<long long>(1LL, std::llround(cfg_.base_market_rate * static_cast<double>(kQ16One))));
        beta_per_us_q16_ = static_cast<std::uint64_t>(
            std::max<long long>(1LL, std::llround(hawkes_.beta[0][0] * 1000.0 * static_cast<double>(kQ16One))));
        state_q16_.fill(0ULL);
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            for (std::size_t j = 0; j < kEventTypes; ++j) {
                alpha_q16_[i][j] = static_cast<std::uint64_t>(
                    std::max<long long>(0LL, std::llround(hawkes_.hawkes_scale * hawkes_.alpha[i][j] * static_cast<double>(kQ16One))));
            }
        }
    }

    // Build birth-death stationary invariant distribution CDF `\pi_i(n)` per HLR (2015) Section 2.2
    void build_stationary_cdf_tables() noexcept {
        for (std::size_t m = 0; m < kMaxQrLevels; ++m) {
            const double mes = static_cast<double>(std::max<Qty>(1, cfg_.level_mes[m]));
            const double lim_mult = cfg_.level_limit_multipliers[m];
            const double can_mult = cfg_.level_cancel_multipliers[m];
            std::array<double, kMaxStationaryUnits> weights{};
            double log_w = 0.0;
            double max_log_w = -1e30;
            for (std::size_t n = 1; n <= kMaxStationaryUnits; ++n) {
                const double q_prev = static_cast<double>(n - 1) * mes;
                const double q_curr = static_cast<double>(n) * mes;
                const double lam_birth = (cfg_.base_limit_rate * lim_mult) /
                    (1.0 + cfg_.limit_depth_decay * std::pow(q_prev + 1.0, cfg_.limit_gamma) + cfg_.theta_ergodic * q_prev);
                const double mu_death = cfg_.base_cancel_rate * can_mult * std::pow(q_curr, cfg_.cancel_delta)
                    + ((m == 0) ? cfg_.base_market_rate * (1.0 + cfg_.market_affinity * std::log1p(q_curr)) : 0.0)
                    + cfg_.theta_ergodic * q_curr * 0.005;
                const double ratio = std::max(1e-6, lam_birth / std::max(1e-6, mu_death));
                log_w += std::log(ratio);
                weights[n - 1] = log_w;
                if (log_w > max_log_w) max_log_w = log_w;
            }
            double sum = 0.0;
            for (std::size_t k = 0; k < kMaxStationaryUnits; ++k) {
                weights[k] = std::exp(weights[k] - max_log_w);
                sum += weights[k];
            }
            double accum = 0.0;
            for (std::size_t k = 0; k < kMaxStationaryUnits; ++k) {
                accum += weights[k] / std::max(1e-12, sum);
                stationary_cdf_[m][k] = accum;
            }
            stationary_cdf_[m][kMaxStationaryUnits - 1] = 1.0;
        }
    }

    QueueReactiveConfig cfg_;
    HawkesMatrix hawkes_;
    Xoshiro256StarStar rng_;
    TimeNs current_time_ns_ = 0;
    // state_[i][j] = current excitation on event i due to past events of type j
    std::array<std::array<double, kEventTypes>, kEventTypes> state_{};
    OnlineHawkesQ16 hawkes_q16_;
    std::uint64_t base_limit_q16_ = 3277ULL;
    std::uint64_t base_cancel_q16_ = 131ULL;
    std::uint64_t base_market_q16_ = 655ULL;
    std::uint64_t beta_per_us_q16_ = kQ16One;
    std::array<std::uint64_t, kEventTypes> state_q16_{};
    std::array<std::array<std::uint64_t, kEventTypes>, kEventTypes> alpha_q16_{};

    Empirical3DIntensityTable table_3d_{};
    std::array<std::array<double, kMaxStationaryUnits>, kMaxQrLevels> stationary_cdf_{};
    std::vector<double> ogata_residuals_{};
    std::array<TransientImpactKernelEvent, kImpactWindowCap> impact_ring_{};
    std::size_t impact_count_ = 0;
    double ou_alpha_state_ = 0.0;
    std::uint64_t reprice_count_ = 0;
};

} // namespace sim
