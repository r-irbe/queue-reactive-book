#pragma once

#include "types.hpp"
#include "rng.hpp"

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

} // namespace sim
