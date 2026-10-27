#pragma once

#include "types.hpp"
#include "rng.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>
#include <optional>

namespace sim {

// ----------------------------------------------------------------------------
// Queue-Reactive Model (Huang, Lehalle & Rosenbaum, 2015)
// and Hawkes Point-Process Arrival Generator
//
// Models limit order book event intensities as state-dependent functions of
// the resting queue depth at the best quotes, coupled with mutually-exciting
// Hawkes feedback dynamics.
// ----------------------------------------------------------------------------

enum class QueueEventType : std::uint8_t {
    BuyLimit    = 0,
    SellLimit   = 1,
    BuyCancel   = 2,
    SellCancel  = 3,
    BuyMarket   = 4,
    SellMarket  = 5,
    Count       = 6
};

inline constexpr std::size_t kEventTypes = static_cast<std::size_t>(QueueEventType::Count);

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

    // Mean order size and standard deviation (log-normal or bounded uniform)
    Qty min_qty              = 1;
    Qty max_qty              = 100;
};

struct HawkesMatrix {
    // Self and cross-excitation matrix alpha[i][j]: event j excites event i
    std::array<std::array<double, kEventTypes>, kEventTypes> alpha{};
    // Decay rates beta[i][j] (1 / nanosecond)
    std::array<std::array<double, kEventTypes>, kEventTypes> beta{};

    static HawkesMatrix default_microstructure() noexcept {
        HawkesMatrix hm{};
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
    QueueEventType type = QueueEventType::BuyLimit;
    Side           side = Side::Buy;
    OrderType      order_type = OrderType::Limit;
    Tick           price = 0;
    Qty            qty = 1;
    TimeNs         inter_arrival_ns = 1000;
};

class QueueReactiveProcess {
public:
    explicit QueueReactiveProcess(QueueReactiveConfig cfg = {},
                                 HawkesMatrix hawkes = HawkesMatrix::default_microstructure(),
                                 std::uint64_t seed = rng_constants::kDefaultSeed)
        : cfg_(cfg), hawkes_(hawkes), rng_(seed), current_time_ns_(0) {
        for (auto& row : state_) {
            row.fill(0.0);
        }
    }

    // Compute instantaneous event intensities given current best bid and ask volumes
    std::array<double, kEventTypes> compute_intensities(Qty bid_volume, Qty ask_volume) const noexcept {
        std::array<double, kEventTypes> lambda{};

        const double q_bid = static_cast<double>(std::max<Qty>(0, bid_volume));
        const double q_ask = static_cast<double>(std::max<Qty>(0, ask_volume));

        // Queue-reactive baseline intensities (Huang et al. 2015)
        lambda[static_cast<std::size_t>(QueueEventType::BuyLimit)] =
            cfg_.base_limit_rate / (1.0 + cfg_.limit_depth_decay * std::pow(q_bid + 1.0, cfg_.limit_gamma));
        lambda[static_cast<std::size_t>(QueueEventType::SellLimit)] =
            cfg_.base_limit_rate / (1.0 + cfg_.limit_depth_decay * std::pow(q_ask + 1.0, cfg_.limit_gamma));

        lambda[static_cast<std::size_t>(QueueEventType::BuyCancel)] =
            cfg_.base_cancel_rate * std::pow(q_bid, cfg_.cancel_delta);
        lambda[static_cast<std::size_t>(QueueEventType::SellCancel)] =
            cfg_.base_cancel_rate * std::pow(q_ask, cfg_.cancel_delta);

        lambda[static_cast<std::size_t>(QueueEventType::BuyMarket)] =
            cfg_.base_market_rate * (1.0 + cfg_.market_affinity * std::log1p(q_ask));
        lambda[static_cast<std::size_t>(QueueEventType::SellMarket)] =
            cfg_.base_market_rate * (1.0 + cfg_.market_affinity * std::log1p(q_bid));

        // Add Hawkes self- and cross-excitation state
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

    // Advance time and generate the next market event
    GeneratedEvent next_event(Tick best_bid_price, Qty best_bid_vol,
                             Tick best_ask_price, Qty best_ask_vol) {
        const auto lambda = compute_intensities(best_bid_vol, best_ask_vol);
        double total_rate = 0.0;
        for (double r : lambda) total_rate += r;

        // Sample inter-arrival time: delta_t ~ Exp(total_rate)
        const double u = std::max(1e-12, 1.0 - distributions::next_unit_double(rng_));
        const double dt_us = -std::log(u) / total_rate;
        const TimeNs dt_ns = std::max<TimeNs>(1, static_cast<TimeNs>(dt_us * 1000.0));

        // Decay Hawkes excitation state over dt
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            for (std::size_t j = 0; j < kEventTypes; ++j) {
                state_[i][j] *= std::exp(-hawkes_.beta[i][j] * static_cast<double>(dt_ns));
            }
        }
        current_time_ns_ += dt_ns;

        // Sample which event occurred according to relative intensity
        const double pick = distributions::next_unit_double(rng_) * total_rate;
        double accum = 0.0;
        QueueEventType selected = QueueEventType::BuyLimit;
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            accum += lambda[i];
            if (pick <= accum || i == kEventTypes - 1) {
                selected = static_cast<QueueEventType>(i);
                break;
            }
        }

        // Excite Hawkes state with the selected event
        const std::size_t sel_idx = static_cast<std::size_t>(selected);
        for (std::size_t i = 0; i < kEventTypes; ++i) {
            state_[i][sel_idx] += hawkes_.alpha[i][sel_idx];
        }

        // Sample order quantity
        const Qty qty = static_cast<Qty>(cfg_.min_qty +
            (rng_.next_u64() % static_cast<std::uint64_t>(std::max<Qty>(1, cfg_.max_qty - cfg_.min_qty + 1))));

        GeneratedEvent ev{};
        ev.type = selected;
        ev.qty = qty;
        ev.inter_arrival_ns = dt_ns;

        switch (selected) {
            case QueueEventType::BuyLimit:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Limit;
                ev.price = best_bid_price;
                break;
            case QueueEventType::SellLimit:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Limit;
                ev.price = best_ask_price;
                break;
            case QueueEventType::BuyCancel:
                ev.side = Side::Buy;
                ev.order_type = OrderType::Cancel;
                ev.price = best_bid_price;
                break;
            case QueueEventType::SellCancel:
                ev.side = Side::Sell;
                ev.order_type = OrderType::Cancel;
                ev.price = best_ask_price;
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
            default:
                break;
        }

        return ev;
    }

    TimeNs current_time() const noexcept { return current_time_ns_; }

private:
    QueueReactiveConfig cfg_;
    HawkesMatrix hawkes_;
    Xoshiro256StarStar rng_;
    TimeNs current_time_ns_ = 0;
    // state_[i][j] = current excitation on event i due to past events of type j
    std::array<std::array<double, kEventTypes>, kEventTypes> state_{};
};

// ----------------------------------------------------------------------------
// Lean-Verified SWAR 4-Bit Nibble LUTs & Branchless Primitives
// (`generated_verified_kernels.rs`, `Rules.MarketSubgameGpu`, `Common.BitwiseSWAR`)
// ----------------------------------------------------------------------------

inline constexpr std::uint32_t kSwarAddLut    = 0x55865432U;
inline constexpr std::uint32_t kSwarCancelLut = 0x40432210U;
inline constexpr std::uint32_t kSwarExecLut   = 0xFF9F4F00U;

// Branchless bit-multiplexer (`chANF` from `/home/radu/code/sair/kernel/common/BitwiseSWAR.lean`):
// selects bits from `t` where `mask` is 1, and from `f` where `mask` is 0, with zero NOT gates.
constexpr std::uint64_t bit_mux(std::uint64_t mask, std::uint64_t t, std::uint64_t f) noexcept {
    return f ^ (mask & (t ^ f));
}

// Branchless all-ones mask (0xFFFFFFFFFFFFFFFFULL) if x == 0, else 0ULL.
constexpr std::uint64_t is_zero_mask(std::uint64_t x) noexcept {
    return (((x | (0ULL - x)) >> 63ULL) - 1ULL);
}

// Branchless sign-mask ReLU (`family4_cfr_branchless_relu64_exact`).
constexpr std::uint64_t verified_cfr_branchless_relu64(std::uint64_t w) noexcept {
    const auto sign_mask = static_cast<std::uint64_t>(static_cast<std::int64_t>(w) >> 63);
    return w & (sign_mask ^ 0xFFFFFFFFFFFFFFFFULL);
}

struct SwarQueueReactiveRegimeLut {
    static constexpr std::uint32_t kAddLut    = kSwarAddLut;
    static constexpr std::uint32_t kCancelLut = kSwarCancelLut;
    static constexpr std::uint32_t kExecLut   = kSwarExecLut;

    static constexpr std::uint32_t extract_nibble(std::uint32_t lut, std::uint32_t regime) noexcept {
        return (lut >> ((regime & 7U) << 2U)) & 0xFU;
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

} // namespace sim
