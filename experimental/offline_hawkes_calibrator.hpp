// Copyright (c) 2026 Open-WebUI & Project Q Systems Engineering
// SPDX-License-Identifier: Apache-2.0
//
// Experimental offline Hawkes/queue-reactive calibrator and VPIN toxicity
// replay surface absorbed from /home/radu/code/q (commit f190d0a4).
// Isolated from the -fno-exceptions matching core behind QRLOB_BUILD_EXPERIMENTAL.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sim {
namespace experimental {

using ReplayHash = std::uint64_t;
using EventTime = std::int64_t;

enum class PointProcessKind : std::uint8_t {
    SelfExcitingHawkes = 1,
};

struct ArrivalEvent {
    std::uint64_t seq = 0;
    EventTime time = 0;
    std::int32_t mark = 0;
    std::int64_t queue_depth = 0;
};

struct HawkesParams {
    std::int64_t base_mu = 0;
    std::int64_t excite_alpha = 0;
    std::uint32_t decay_shift = 1;
    double depth_decay = 0.0;
};

struct EventStat {
    std::size_t index = 0;
    EventTime time = 0;
    EventTime interarrival = 0;
    std::size_t count_through = 0;
    double excitation = 0.0;
    double intensity = 0.0;
};

struct ArrivalDiagnostics {
    PointProcessKind process = PointProcessKind::SelfExcitingHawkes;
    std::size_t n = 0;
    EventTime first_time = 0;
    EventTime last_time = 0;
    EventTime total_span = 0;
    std::int64_t total_interarrival = 0;
    EventTime min_interarrival = 0;
    EventTime max_interarrival = 0;
    double mean_interarrival = 0.0;
    double empirical_intensity = 0.0;
    double max_intensity = 0.0;
    double final_intensity = 0.0;
    ReplayHash replay_hash = 0;

    static constexpr bool is_matching_kernel_input = false;
};

struct ArrivalReplayResult {
    ArrivalDiagnostics diagnostics{};
    std::vector<EventStat> events;

    static constexpr bool is_matching_kernel_input = false;
};

namespace detail {

inline constexpr ReplayHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr ReplayHash kFnvPrime = 1099511628211ULL;

inline constexpr ReplayHash fnv1a_byte(ReplayHash h, std::uint8_t b) noexcept {
    h ^= static_cast<ReplayHash>(b);
    h *= kFnvPrime;
    return h;
}

inline constexpr ReplayHash fnv1a_u64(ReplayHash h, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = fnv1a_byte(h, static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
    return h;
}

inline constexpr ReplayHash fnv1a_i64(ReplayHash h, std::int64_t v) noexcept {
    return fnv1a_u64(h, static_cast<std::uint64_t>(v));
}

inline constexpr std::int64_t iabs(std::int64_t x) noexcept {
    return x < 0 ? -x : x;
}

inline std::int64_t quantize_fp(double x) noexcept {
    if (!std::isfinite(x)) return 0;
    return static_cast<std::int64_t>(std::llround(x * 1000000.0));
}

inline double dyadic_decay(std::uint32_t shift, EventTime dt) noexcept {
    if (dt <= 0 || shift == 0) return 1.0;
    const std::uint64_t total_shift =
        static_cast<std::uint64_t>(shift) * static_cast<std::uint64_t>(dt);
    if (total_shift >= 62ULL) return 0.0;
    return 1.0 / static_cast<double>(1ULL << total_shift);
}

} // namespace detail

inline ArrivalReplayResult evaluate_arrival_point_process(
    const std::vector<ArrivalEvent>& stream,
    const HawkesParams& params) {
    ArrivalReplayResult out{};
    out.diagnostics.process = PointProcessKind::SelfExcitingHawkes;
    out.diagnostics.n = stream.size();

    ReplayHash h = detail::kFnvOffsetBasis;
    h = detail::fnv1a_i64(h, params.base_mu);
    h = detail::fnv1a_i64(h, params.excite_alpha);
    h = detail::fnv1a_u64(h, params.decay_shift);

    if (stream.empty()) {
        out.diagnostics.replay_hash = h;
        return out;
    }

    out.events.reserve(stream.size());
    out.diagnostics.first_time = stream.front().time;
    out.diagnostics.last_time = stream.back().time;
    out.diagnostics.total_span = out.diagnostics.last_time - out.diagnostics.first_time;

    double r_prev = 0.0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const ArrivalEvent& ev = stream[i];
        EventStat st{};
        st.index = i;
        st.time = ev.time;
        st.count_through = i + 1;

        if (i == 0) {
            st.interarrival = 0;
            st.excitation = 0.0;
        } else {
            const EventTime dt = ev.time - stream[i - 1].time;
            st.interarrival = dt;
            out.diagnostics.total_interarrival += dt;
            if (i == 1 || dt < out.diagnostics.min_interarrival) {
                out.diagnostics.min_interarrival = dt;
            }
            if (i == 1 || dt > out.diagnostics.max_interarrival) {
                out.diagnostics.max_interarrival = dt;
            }
            const double k = detail::dyadic_decay(params.decay_shift, dt);
            st.excitation = k * (r_prev + 1.0);
        }
        r_prev = st.excitation;

        const double depth_factor =
            (params.depth_decay > 0.0 && ev.queue_depth > 0)
                ? 1.0 / (1.0 + params.depth_decay * static_cast<double>(ev.queue_depth))
                : 1.0;
        st.intensity =
            static_cast<double>(params.base_mu) * depth_factor +
            static_cast<double>(params.excite_alpha) * st.excitation;

        if (i == 0 || st.intensity > out.diagnostics.max_intensity) {
            out.diagnostics.max_intensity = st.intensity;
        }
        out.diagnostics.final_intensity = st.intensity;
        out.events.push_back(st);

        h = detail::fnv1a_u64(h, ev.seq);
        h = detail::fnv1a_i64(h, ev.time);
        h = detail::fnv1a_i64(h, ev.mark);
        h = detail::fnv1a_i64(h, detail::quantize_fp(st.intensity));
    }

    if (stream.size() >= 2) {
        out.diagnostics.mean_interarrival =
            static_cast<double>(out.diagnostics.total_interarrival) /
            static_cast<double>(stream.size() - 1);
    }
    if (out.diagnostics.total_span > 0) {
        out.diagnostics.empirical_intensity =
            static_cast<double>(stream.size()) /
            static_cast<double>(out.diagnostics.total_span);
    }

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(out.diagnostics.n));
    h = detail::fnv1a_i64(h, detail::quantize_fp(out.diagnostics.max_intensity));
    out.diagnostics.replay_hash = h;
    return out;
}

class ArrivalPointProcessReplay {
public:
    ArrivalPointProcessReplay(const std::vector<ArrivalEvent>& stream,
                              HawkesParams params) noexcept
        : stream_(&stream), params_(params) {}

    ArrivalReplayResult analyze() const {
        return evaluate_arrival_point_process(*stream_, params_);
    }

    std::size_t size() const noexcept { return stream_->size(); }
    bool empty() const noexcept { return stream_->empty(); }

private:
    const std::vector<ArrivalEvent>* stream_;
    HawkesParams params_;
};

// State-dependent Hawkes + Queue-Reactive empirical calibrator (offline only).
struct CalibratedHawkesSummary {
    double empirical_rate = 0.0;
    double branching_ratio = 0.0;
    double baseline_mu = 0.0;
    double alpha_estimate = 0.0;
    double depth_sensitivity = 0.0;
    ReplayHash calibration_hash = 0;

    static constexpr bool is_matching_kernel_input = false;
};

class StateDependentHawkesCalibrator {
public:
    static CalibratedHawkesSummary calibrate(
        const std::vector<ArrivalEvent>& stream,
        std::uint32_t decay_shift = 1) noexcept {
        CalibratedHawkesSummary s{};
        if (stream.size() < 2) {
            s.calibration_hash = detail::kFnvOffsetBasis;
            return s;
        }
        const EventTime span = stream.back().time - stream.front().time;
        if (span <= 0) {
            s.calibration_hash = detail::kFnvOffsetBasis;
            return s;
        }
        s.empirical_rate =
            static_cast<double>(stream.size()) / static_cast<double>(span);

        double sum_dt = 0.0;
        double sum_dt_sq = 0.0;
        double sum_depth = 0.0;
        const std::size_t gaps = stream.size() - 1;
        for (std::size_t i = 1; i < stream.size(); ++i) {
            const double dt = static_cast<double>(
                std::max<EventTime>(1, stream[i].time - stream[i - 1].time));
            sum_dt += dt;
            sum_dt_sq += dt * dt;
            sum_depth += static_cast<double>(std::max<std::int64_t>(0, stream[i].queue_depth));
        }
        const double mean_dt = sum_dt / static_cast<double>(gaps);
        const double var_dt =
            std::max(0.0, (sum_dt_sq / static_cast<double>(gaps)) - (mean_dt * mean_dt));
        const double cv_sq = (mean_dt > 0.0) ? (var_dt / (mean_dt * mean_dt)) : 1.0;

        // Moment-based branching ratio estimate n_hat in [0, 0.95] from overdispersion
        if (cv_sq > 1.0) {
            s.branching_ratio = std::min(0.95, 1.0 - (1.0 / std::sqrt(cv_sq)));
        } else {
            s.branching_ratio = 0.0;
        }
        s.baseline_mu = s.empirical_rate * (1.0 - s.branching_ratio);
        const double beta_equiv = static_cast<double>(std::max<std::uint32_t>(1, decay_shift)) * 0.6931471805599453;
        s.alpha_estimate = s.branching_ratio * beta_equiv;
        const double mean_depth = sum_depth / static_cast<double>(gaps);
        s.depth_sensitivity = (mean_depth > 0.0) ? (1.0 / (1.0 + mean_depth)) : 0.0;

        ReplayHash h = detail::kFnvOffsetBasis;
        h = detail::fnv1a_i64(h, detail::quantize_fp(s.empirical_rate));
        h = detail::fnv1a_i64(h, detail::quantize_fp(s.branching_ratio));
        h = detail::fnv1a_i64(h, detail::quantize_fp(s.baseline_mu));
        h = detail::fnv1a_i64(h, detail::quantize_fp(s.alpha_estimate));
        h = detail::fnv1a_i64(h, detail::quantize_fp(s.depth_sensitivity));
        s.calibration_hash = h;
        return s;
    }
};

// --- Offline VPIN / Flow-Toxicity Replay Surface -----------------------------

inline constexpr std::int64_t kVpinScale = 1000000;

struct SignedTrade {
    std::uint64_t seq = 0;
    std::uint64_t ts_ns = 0;
    std::int64_t price = 0;
    std::int64_t volume = 0;
    std::int8_t side = 0; // +1 buy-classified, -1 sell-classified
};

struct VolumeBucket {
    std::size_t index = 0;
    std::int64_t buy_vol = 0;
    std::int64_t sell_vol = 0;
    std::int64_t total_vol = 0;
    std::int64_t imbalance = 0;
    std::int64_t abs_imbalance = 0;
    std::int64_t toxicity_scaled = 0;

    static constexpr bool is_matching_kernel_input = false;
};

struct VpinResult {
    std::int64_t bucket_size = 0;
    std::size_t n_trades = 0;
    std::size_t n_buckets = 0;
    std::int64_t bucketed_volume = 0;
    std::int64_t residual_volume = 0;
    std::int64_t sum_abs_imbalance = 0;
    std::int64_t vpin_scaled = 0;
    std::vector<VolumeBucket> buckets;
    ReplayHash replay_hash = 0;

    static constexpr bool is_matching_kernel_input = false;
};

inline VolumeBucket finalize_vpin_bucket(std::size_t index,
                                         std::int64_t buy_vol,
                                         std::int64_t sell_vol,
                                         std::int64_t bucket_size) noexcept {
    VolumeBucket b{};
    b.index = index;
    b.buy_vol = buy_vol;
    b.sell_vol = sell_vol;
    b.total_vol = buy_vol + sell_vol;
    b.imbalance = buy_vol - sell_vol;
    b.abs_imbalance = detail::iabs(b.imbalance);
    b.toxicity_scaled =
        bucket_size > 0 ? (b.abs_imbalance * kVpinScale / bucket_size) : 0;
    return b;
}

inline VpinResult evaluate_vpin(const std::vector<SignedTrade>& trades,
                                std::int64_t bucket_size) {
    VpinResult res{};
    res.bucket_size = bucket_size;
    res.n_trades = trades.size();

    ReplayHash h = detail::kFnvOffsetBasis;
    h = detail::fnv1a_i64(h, bucket_size);
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(trades.size()));

    if (bucket_size <= 0 || trades.empty()) {
        res.replay_hash = h;
        return res;
    }

    std::int64_t cur_buy = 0;
    std::int64_t cur_sell = 0;
    std::int64_t cur_total = 0;
    std::size_t bucket_index = 0;

    for (const SignedTrade& t : trades) {
        h = detail::fnv1a_u64(h, t.seq);
        h = detail::fnv1a_u64(h, t.ts_ns);
        h = detail::fnv1a_i64(h, t.price);
        h = detail::fnv1a_i64(h, t.volume);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(t.side));

        std::int64_t remaining = t.volume > 0 ? t.volume : 0;
        while (remaining > 0) {
            const std::int64_t room = bucket_size - cur_total;
            const std::int64_t take = remaining < room ? remaining : room;
            if (t.side >= 0) {
                cur_buy += take;
            } else {
                cur_sell += take;
            }
            cur_total += take;
            remaining -= take;

            if (cur_total == bucket_size) {
                VolumeBucket b =
                    finalize_vpin_bucket(bucket_index, cur_buy, cur_sell, bucket_size);
                res.sum_abs_imbalance += b.abs_imbalance;
                res.buckets.push_back(b);
                h = detail::fnv1a_i64(h, b.imbalance);
                ++bucket_index;
                cur_buy = 0;
                cur_sell = 0;
                cur_total = 0;
            }
        }
    }

    res.n_buckets = res.buckets.size();
    res.bucketed_volume = static_cast<std::int64_t>(res.n_buckets) * bucket_size;
    res.residual_volume = cur_total;
    if (res.bucketed_volume > 0) {
        res.vpin_scaled = res.sum_abs_imbalance * kVpinScale / res.bucketed_volume;
    }
    h = detail::fnv1a_i64(h, res.vpin_scaled);
    res.replay_hash = h;
    return res;
}

inline std::int64_t vpin_scaled_over_window(const VpinResult& res,
                                            std::size_t window) noexcept {
    if (res.n_buckets == 0 || res.bucket_size <= 0) return 0;
    const std::size_t w = window < res.n_buckets ? window : res.n_buckets;
    if (w == 0) return 0;
    std::int64_t sum_abs = 0;
    for (std::size_t k = res.n_buckets - w; k < res.n_buckets; ++k) {
        sum_abs += res.buckets[k].abs_imbalance;
    }
    const std::int64_t denom = static_cast<std::int64_t>(w) * res.bucket_size;
    return denom > 0 ? sum_abs * kVpinScale / denom : 0;
}

class VpinToxicityReplay {
public:
    explicit VpinToxicityReplay(const std::vector<SignedTrade>& trades) noexcept
        : trades_(&trades) {}

    VpinResult evaluate(std::int64_t bucket_size) const {
        return evaluate_vpin(*trades_, bucket_size);
    }
    ReplayHash hash(std::int64_t bucket_size) const {
        return evaluate_vpin(*trades_, bucket_size).replay_hash;
    }
    std::size_t size() const noexcept { return trades_->size(); }
    bool empty() const noexcept { return trades_->empty(); }

private:
    const std::vector<SignedTrade>* trades_;
};

// Compile-time non-feedback / non-coupling traits
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

template <typename T, typename = void>
struct is_matching_kernel_input : std::false_type {};

template <typename T>
struct is_matching_kernel_input<
    T, std::void_t<decltype(T::is_matching_kernel_input)>>
    : std::bool_constant<T::is_matching_kernel_input> {};

template <typename T>
inline constexpr bool is_matching_kernel_input_v =
    is_matching_kernel_input<T>::value;

static_assert(std::is_trivially_copyable_v<ArrivalEvent>,
              "ArrivalEvent must be pure data");
static_assert(std::is_trivially_copyable_v<SignedTrade>,
              "SignedTrade must be pure data");
static_assert(std::is_trivially_copyable_v<VolumeBucket>,
              "VolumeBucket must be pure data");
static_assert(!has_matching_feedback_v<ArrivalDiagnostics>,
              "ArrivalDiagnostics must not expose a matching-feedback member");
static_assert(!has_matching_feedback_v<VpinResult>,
              "VpinResult must not expose a matching-feedback member");
static_assert(!is_matching_kernel_input_v<VolumeBucket>,
              "VolumeBucket is an offline diagnostic, never a kernel input");
static_assert(!is_matching_kernel_input_v<VpinResult>,
              "VpinResult is an offline diagnostic, never a kernel input");

} // namespace experimental
} // namespace sim
