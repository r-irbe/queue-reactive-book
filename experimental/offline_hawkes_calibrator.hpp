// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// Offline dyadic Hawkes point-process intensity replay, moment-based
// branching-ratio calibrator, exact O(N) Ozaki + Queue-Reactive MLE
// (Hybrid EM + Projected Damped Newton with Armijo line search),
// and equal-volume VPIN flow-toxicity evaluator.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
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
    std::int64_t opp_queue_depth = 0;
    std::int64_t spread_ticks = 1;
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

inline void ks_exp1_from_sorted(const std::vector<double>& sorted_positive,
                                double& out_d,
                                double& out_pval) noexcept {
    const std::size_t n = sorted_positive.size();
    if (n == 0) {
        out_d = 1.0;
        out_pval = 0.0;
        return;
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    double d_max = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double cdf = 1.0 - std::exp(-sorted_positive[i]);
        const double d_plus = (static_cast<double>(i + 1) * inv_n) - cdf;
        const double d_minus = cdf - (static_cast<double>(i) * inv_n);
        if (d_plus > d_max) d_max = d_plus;
        if (d_minus > d_max) d_max = d_minus;
    }
    out_d = d_max;
    const double en = std::sqrt(static_cast<double>(n));
    const double lam = (en + 0.12 + (0.11 / en)) * d_max;
    double pval = 0.0;
    for (int k = 1; k <= 25; ++k) {
        const double term = 2.0 * std::exp(-2.0 * static_cast<double>(k * k) * lam * lam);
        pval += ((k & 1) != 0) ? term : -term;
    }
    out_pval = std::clamp(pval, 0.0, 1.0);
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

struct HawkesLikelihoodEval {
    double log_likelihood = 0.0;
    double grad_mu = 0.0;
    double grad_alpha = 0.0;
    double grad_beta = 0.0;
    double grad_gamma = 0.0;
    double fisher_mu = 0.0;
    double fisher_alpha = 0.0;
    double fisher_beta = 0.0;
    double fisher_gamma = 0.0;
    double em_sum_p_mu = 0.0;
    double em_sum_p_alpha = 0.0;
    double em_sum_phi_dt = 0.0;
    double em_sum_rplus_e = 0.0;
};

struct HawkesMleResult {
    std::size_t n_total = 0;
    std::size_t n_train = 0;
    std::size_t n_oos = 0;
    double mean_queue_depth = 0.0;
    double mean_spread_ticks = 1.0;

    // Fitted MLE parameters (rates in Hz = events/s)
    double mu_hz = 0.0;
    double alpha_hz = 0.0;
    double beta_hz = 0.0;
    double branching_ratio = 0.0;
    double gamma_qr = 0.0;

    // Analytical score vector at MLE
    double score_mu = 0.0;
    double score_alpha = 0.0;
    double score_beta = 0.0;
    double score_gamma = 0.0;

    // Observed Fisher Information diagonal & asymptotic standard errors
    double fisher_mu = 0.0;
    double fisher_alpha = 0.0;
    double fisher_beta = 0.0;
    double fisher_gamma = 0.0;
    double se_mu = 0.0;
    double se_alpha = 0.0;
    double se_beta = 0.0;
    double se_gamma = 0.0;

    // In-sample log-likelihoods & information criteria
    double log_likelihood_init = 0.0;
    double log_likelihood_train = 0.0;
    double log_likelihood_poisson_train = 0.0;
    double aic_hawkes = 0.0;
    double bic_hawkes = 0.0;
    double aic_poisson = 0.0;

    // Out-of-sample (OOS) evaluation & Ogata time-rescaling goodness-of-fit
    double log_likelihood_oos = 0.0;
    double log_likelihood_poisson_oos = 0.0;
    double oos_llr_per_event = 0.0;
    double ogata_residual_mean = 0.0;
    double ogata_residual_std = 0.0;
    double ks_statistic = 0.0;
    double ks_pvalue = 0.0;
    std::uint32_t iterations = 0;
    bool converged = false;
    std::vector<double> oos_ogata_residuals;
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

    // Exact O(N) Ozaki recursive log-likelihood, analytical score vector
    // (dL/dmu, dL/dalpha, dL/dbeta, dL/dgamma_qr), and observed Fisher diagonal.
    static HawkesLikelihoodEval evaluate_log_likelihood(
        std::span<const ArrivalEvent> events,
        double mu,
        double alpha,
        double beta,
        double gamma_qr,
        double q_bar_override = 0.0) noexcept {
        HawkesLikelihoodEval out{};
        const std::size_t n = events.size();
        if (n == 0 || beta <= 0.0) return out;

        double q_bar = q_bar_override;
        if (q_bar <= 0.0) {
            double sum_q = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum_q += static_cast<double>(std::max<std::int64_t>(0, events[k].queue_depth));
            }
            q_bar = std::max(1.0, sum_q / static_cast<double>(n));
        } else {
            q_bar = std::max(1.0, q_bar);
        }

        double r = 0.0;
        double rp = 0.0;
        double rpp = 0.0;
        const double inv_beta = 1.0 / beta;
        const double inv_beta2 = inv_beta * inv_beta;
        const double inv_beta3 = inv_beta2 * inv_beta;

        for (std::size_t k = 0; k < n; ++k) {
            if (k > 0) {
                const std::int64_t raw_dt_us = events[k].time - events[k - 1].time;
                const double dt = (raw_dt_us > 0)
                    ? (static_cast<double>(raw_dt_us) * 1e-6)
                    : 1e-7;
                const double bdt = beta * dt;
                const double e_bdt = std::exp(-bdt);
                const double one_m_e = 1.0 - e_bdt;
                const double r_plus = r + 1.0;
                const double rp_prev = rp;
                const double rpp_prev = rpp;

                const double q_prev = static_cast<double>(
                    std::max<std::int64_t>(0, events[k - 1].queue_depth)) / q_bar;
                const double phi_prev = 1.0 / std::max(1e-9, 1.0 + gamma_qr * q_prev);
                const double phi_prev2 = phi_prev * phi_prev;
                const double phi_prev3 = phi_prev2 * phi_prev;

                const double comp_k =
                    mu * phi_prev * dt + (alpha * inv_beta) * r_plus * one_m_e;
                out.log_likelihood -= comp_k;
                out.em_sum_phi_dt += phi_prev * dt;
                out.em_sum_rplus_e += (r_plus * inv_beta) * one_m_e;

                out.grad_mu -= phi_prev * dt;
                out.grad_alpha -= (r_plus * inv_beta) * one_m_e;

                const double d_comp_dbeta =
                    (alpha * inv_beta) * rp_prev * one_m_e +
                    alpha * r_plus * (bdt * e_bdt - one_m_e) * inv_beta2;
                const double d2_comp_dbeta2 =
                    (alpha * inv_beta) * rpp_prev * one_m_e +
                    2.0 * alpha * rp_prev * (bdt * e_bdt - one_m_e) * inv_beta2 +
                    alpha * r_plus * (-(bdt * bdt) * e_bdt - 2.0 * bdt * e_bdt + 2.0 * one_m_e) * inv_beta3;

                out.grad_beta -= d_comp_dbeta;
                out.grad_gamma += mu * dt * q_prev * phi_prev2;

                out.fisher_beta += d2_comp_dbeta2;
                out.fisher_gamma += 2.0 * mu * dt * (q_prev * q_prev) * phi_prev3;

                r = e_bdt * r_plus;
                rp = e_bdt * (rp_prev - dt * r_plus);
                rpp = e_bdt * (rpp_prev - 2.0 * dt * rp_prev + (dt * dt) * r_plus);
            }

            const double q_k = static_cast<double>(
                std::max<std::int64_t>(0, events[k].queue_depth)) / q_bar;
            const double phi_k = 1.0 / std::max(1e-9, 1.0 + gamma_qr * q_k);
            const double phi_k2 = phi_k * phi_k;
            const double phi_k3 = phi_k2 * phi_k;

            const double lam = std::max(1e-12, mu * phi_k + alpha * r);
            out.log_likelihood += std::log(lam);

            const double inv_lam = 1.0 / lam;
            const double inv_lam2 = inv_lam * inv_lam;
            out.em_sum_p_mu += (mu * phi_k) * inv_lam;
            out.em_sum_p_alpha += (alpha * r) * inv_lam;

            const double d_lam_dmu = phi_k;
            const double d_lam_dalpha = r;
            const double d_lam_dbeta = alpha * rp;
            const double d2_lam_dbeta2 = alpha * rpp;
            const double d_lam_dgamma = -mu * q_k * phi_k2;
            const double d2_lam_dgamma2 = 2.0 * mu * (q_k * q_k) * phi_k3;

            out.grad_mu += d_lam_dmu * inv_lam;
            out.grad_alpha += d_lam_dalpha * inv_lam;
            out.grad_beta += d_lam_dbeta * inv_lam;
            out.grad_gamma += d_lam_dgamma * inv_lam;

            out.fisher_mu += (d_lam_dmu * d_lam_dmu) * inv_lam2;
            out.fisher_alpha += (d_lam_dalpha * d_lam_dalpha) * inv_lam2;
            out.fisher_beta += (d_lam_dbeta * d_lam_dbeta) * inv_lam2 - d2_lam_dbeta2 * inv_lam;
            out.fisher_gamma += (d_lam_dgamma * d_lam_dgamma) * inv_lam2 - d2_lam_dgamma2 * inv_lam;
        }
        return out;
    }

    // Hybrid EM + Projected Damped Newton with Armijo backtracking line search
    static HawkesMleResult calibrate_mle(
        std::span<const ArrivalEvent> events,
        double train_fraction = 0.70) {
        HawkesMleResult res{};
        res.n_total = events.size();
        if (res.n_total < 4) {
            res.calibration_hash = detail::kFnvOffsetBasis;
            return res;
        }

        const double clamped_frac = std::clamp(train_fraction, 0.20, 0.95);
        res.n_train = std::clamp<std::size_t>(
            static_cast<std::size_t>(static_cast<double>(res.n_total) * clamped_frac),
            2, res.n_total - 1);
        res.n_oos = res.n_total - res.n_train;

        const std::span<const ArrivalEvent> train_ev = events.subspan(0, res.n_train);

        double sum_q = 0.0;
        double sum_spread = 0.0;
        for (std::size_t k = 0; k < res.n_train; ++k) {
            sum_q += static_cast<double>(std::max<std::int64_t>(0, train_ev[k].queue_depth));
            sum_spread += static_cast<double>(std::max<std::int64_t>(1, train_ev[k].spread_ticks));
        }
        res.mean_queue_depth = std::max(1.0, sum_q / static_cast<double>(res.n_train));
        res.mean_spread_ticks = sum_spread / static_cast<double>(res.n_train);
        const double q_bar = res.mean_queue_depth;

        double span_train = 0.0;
        for (std::size_t k = 1; k < res.n_train; ++k) {
            const std::int64_t raw_dt = train_ev[k].time - train_ev[k - 1].time;
            span_train += (raw_dt > 0) ? (static_cast<double>(raw_dt) * 1e-6) : 1e-7;
        }
        span_train = std::max(1e-9, span_train);

        const double lam_pois_train =
            static_cast<double>(res.n_train) / span_train;
        res.log_likelihood_poisson_train =
            static_cast<double>(res.n_train) * std::log(std::max(1e-12, lam_pois_train)) -
            lam_pois_train * span_train;

        double mu = std::max(1e-3, 0.50 * lam_pois_train);
        double beta = std::max(5.0, 2.50 * lam_pois_train);
        double alpha = 0.45 * beta;
        double gamma_qr = 0.15;

        HawkesLikelihoodEval cur_eval =
            evaluate_log_likelihood(train_ev, mu, alpha, beta, gamma_qr, q_bar);
        res.log_likelihood_init = cur_eval.log_likelihood;

        constexpr std::uint32_t kMaxIter = 60;
        for (std::uint32_t it = 0; it < kMaxIter; ++it) {
            res.iterations = it + 1;
            const double prev_ll = cur_eval.log_likelihood;

            // 1. Exact EM step on (mu, alpha)
            double mu_em = std::max(1e-4, cur_eval.em_sum_p_mu / std::max(1e-9, cur_eval.em_sum_phi_dt));
            double alpha_em = std::max(1e-4, cur_eval.em_sum_p_alpha / std::max(1e-9, cur_eval.em_sum_rplus_e));
            if (alpha_em / beta > 0.98) {
                alpha_em = 0.98 * beta;
            }
            const HawkesLikelihoodEval em_eval =
                evaluate_log_likelihood(train_ev, mu_em, alpha_em, beta, gamma_qr, q_bar);
            if (em_eval.log_likelihood >= cur_eval.log_likelihood) {
                mu = mu_em;
                alpha = alpha_em;
                cur_eval = em_eval;
            }

            // 2. Projected Damped Newton with Armijo backtracking line search on (beta, gamma_qr)
            const double h_beta = std::max(1e-4, std::abs(cur_eval.fisher_beta)) + 1e-3;
            const double h_gamma = std::max(1e-4, std::abs(cur_eval.fisher_gamma)) + 1e-3;
            const double d_beta = std::clamp(cur_eval.grad_beta / h_beta, -0.50 * beta, 0.50 * beta);
            const double d_gamma = std::clamp(cur_eval.grad_gamma / h_gamma, -0.50, 0.50);

            double step = 1.0;
            for (int ls = 0; ls < 12; ++ls) {
                const double beta_min = std::max(1e-3, alpha / 0.98);
                const double beta_try = std::max(beta_min, beta + step * d_beta);
                const double gamma_try = std::clamp(gamma_qr + step * d_gamma, 0.0, 10.0);
                const HawkesLikelihoodEval try_eval =
                    evaluate_log_likelihood(train_ev, mu, alpha, beta_try, gamma_try, q_bar);
                const double dir_deriv =
                    cur_eval.grad_beta * (beta_try - beta) +
                    cur_eval.grad_gamma * (gamma_try - gamma_qr);
                if (try_eval.log_likelihood >= cur_eval.log_likelihood + 1e-4 * std::max(0.0, dir_deriv)) {
                    beta = beta_try;
                    gamma_qr = gamma_try;
                    cur_eval = try_eval;
                    break;
                }
                step *= 0.5;
            }

            if (it >= 5 && std::abs(cur_eval.log_likelihood - prev_ll) < 1e-6) {
                res.converged = true;
                break;
            }
        }

        // Final score & Fisher evaluation at MLE
        cur_eval = evaluate_log_likelihood(train_ev, mu, alpha, beta, gamma_qr, q_bar);
        res.mu_hz = mu;
        res.alpha_hz = alpha;
        res.beta_hz = beta;
        res.branching_ratio = alpha / std::max(1e-9, beta);
        res.gamma_qr = gamma_qr;

        res.score_mu = cur_eval.grad_mu;
        res.score_alpha = cur_eval.grad_alpha;
        res.score_beta = cur_eval.grad_beta;
        res.score_gamma = cur_eval.grad_gamma;

        res.fisher_mu = cur_eval.fisher_mu;
        res.fisher_alpha = cur_eval.fisher_alpha;
        res.fisher_beta = cur_eval.fisher_beta;
        res.fisher_gamma = cur_eval.fisher_gamma;

        res.se_mu = 1.0 / std::sqrt(std::max(1e-12, res.fisher_mu));
        res.se_alpha = 1.0 / std::sqrt(std::max(1e-12, res.fisher_alpha));
        res.se_beta = 1.0 / std::sqrt(std::max(1e-12, res.fisher_beta));
        res.se_gamma = 1.0 / std::sqrt(std::max(1e-12, res.fisher_gamma));

        res.log_likelihood_train = cur_eval.log_likelihood;
        res.aic_hawkes = 8.0 - 2.0 * res.log_likelihood_train;
        res.bic_hawkes = 4.0 * std::log(static_cast<double>(res.n_train)) - 2.0 * res.log_likelihood_train;
        res.aic_poisson = 2.0 - 2.0 * res.log_likelihood_poisson_train;

        // Out-of-sample (OOS) evaluation & exact Ogata time-rescaling residuals tau_k = Lambda_k
        double r = 0.0;
        double ll_oos = 0.0;
        double span_oos = 0.0;
        res.oos_ogata_residuals.reserve(res.n_oos);

        for (std::size_t k = 0; k < res.n_total; ++k) {
            if (k > 0) {
                const std::int64_t raw_dt = events[k].time - events[k - 1].time;
                const double dt = (raw_dt > 0) ? (static_cast<double>(raw_dt) * 1e-6) : 1e-7;
                const double e_bdt = std::exp(-beta * dt);
                const double r_plus = r + 1.0;
                const double q_prev = static_cast<double>(
                    std::max<std::int64_t>(0, events[k - 1].queue_depth)) / q_bar;
                const double phi_prev = 1.0 / std::max(1e-9, 1.0 + gamma_qr * q_prev);
                const double comp_k =
                    mu * phi_prev * dt + (alpha / beta) * r_plus * (1.0 - e_bdt);
                r = e_bdt * r_plus;
                if (k >= res.n_train) {
                    ll_oos -= comp_k;
                    span_oos += dt;
                    if (comp_k > 0.0 && std::isfinite(comp_k)) {
                        res.oos_ogata_residuals.push_back(comp_k);
                    }
                }
            }
            const double q_k = static_cast<double>(
                std::max<std::int64_t>(0, events[k].queue_depth)) / q_bar;
            const double phi_k = 1.0 / std::max(1e-9, 1.0 + gamma_qr * q_k);
            const double lam = std::max(1e-12, mu * phi_k + alpha * r);
            if (k >= res.n_train) {
                ll_oos += std::log(lam);
            }
        }

        res.log_likelihood_oos = ll_oos;
        res.log_likelihood_poisson_oos =
            static_cast<double>(res.n_oos) * std::log(std::max(1e-12, lam_pois_train)) -
            lam_pois_train * span_oos;
        res.oos_llr_per_event =
            (res.log_likelihood_oos - res.log_likelihood_poisson_oos) /
            static_cast<double>(std::max<std::size_t>(1, res.n_oos));

        if (!res.oos_ogata_residuals.empty()) {
            double sum_tau = 0.0;
            double sum_tau2 = 0.0;
            for (double tau : res.oos_ogata_residuals) {
                sum_tau += tau;
                sum_tau2 += tau * tau;
            }
            const double m = sum_tau / static_cast<double>(res.oos_ogata_residuals.size());
            const double v = std::max(0.0, (sum_tau2 / static_cast<double>(res.oos_ogata_residuals.size())) - (m * m));
            res.ogata_residual_mean = m;
            res.ogata_residual_std = std::sqrt(v);

            std::vector<double> sorted_res = res.oos_ogata_residuals;
            std::sort(sorted_res.begin(), sorted_res.end());
            detail::ks_exp1_from_sorted(sorted_res, res.ks_statistic, res.ks_pvalue);
        }

        ReplayHash h = detail::kFnvOffsetBasis;
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.mu_hz));
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.alpha_hz));
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.beta_hz));
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.gamma_qr));
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.log_likelihood_train));
        h = detail::fnv1a_i64(h, detail::quantize_fp(res.oos_llr_per_event));
        res.calibration_hash = h;
        return res;
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

// Co-read LOBSTER _message_5.csv and companion _orderbook_5.csv (when present)
// so ArrivalEvent::queue_depth reflects the true pre-event resting L1 queue depth
// on the active side (bid_size_1 for dir > 0, ask_size_1 for dir < 0).
inline bool load_lobster_paired_csv(
    const std::string& message_csv_path,
    std::vector<ArrivalEvent>& out_arrivals,
    std::vector<SignedTrade>& out_trades) {
    out_arrivals.clear();
    out_trades.clear();
    std::string resolved_msg_path = message_csv_path;
    std::ifstream msg_ifs(resolved_msg_path);
    if (!msg_ifs.is_open() && !message_csv_path.empty() && message_csv_path[0] != '/') {
        resolved_msg_path = std::string("../") + message_csv_path;
        msg_ifs.open(resolved_msg_path);
    }
    if (!msg_ifs.is_open()) return false;

    std::string ob_path = resolved_msg_path;
    const std::size_t pos_msg5 = ob_path.find("_message_5.csv");
    if (pos_msg5 != std::string::npos) {
        ob_path.replace(pos_msg5, 14, "_orderbook_5.csv");
    } else {
        const std::size_t pos_msg = ob_path.find("_message_");
        if (pos_msg != std::string::npos) {
            ob_path.replace(pos_msg, 9, "_orderbook_");
        }
    }
    std::ifstream ob_ifs;
    bool has_ob = false;
    if (ob_path != resolved_msg_path) {
        ob_ifs.open(ob_path);
        has_ob = ob_ifs.is_open();
    }

    bool have_prev_ob = false;
    std::int64_t prev_ask_p1 = 0;
    std::int64_t prev_ask_q1 = 0;
    std::int64_t prev_bid_p1 = 0;
    std::int64_t prev_bid_q1 = 0;

    std::string msg_line;
    std::string ob_line;
    std::uint64_t seq = 0;
    while (std::getline(msg_ifs, msg_line)) {
        bool ob_row_ok = false;
        std::int64_t cur_ask_p1 = 0;
        std::int64_t cur_ask_q1 = 0;
        std::int64_t cur_bid_p1 = 0;
        std::int64_t cur_bid_q1 = 0;

        if (has_ob && std::getline(ob_ifs, ob_line) && !ob_line.empty()) {
            std::stringstream ob_ss(ob_line);
            std::string ob_tok;
            std::int64_t ob_vals[4]{};
            int ob_idx = 0;
            while (ob_idx < 4 && std::getline(ob_ss, ob_tok, ',')) {
                ob_vals[ob_idx++] = std::strtoll(ob_tok.c_str(), nullptr, 10);
            }
            if (ob_idx == 4) {
                cur_ask_p1 = ob_vals[0];
                cur_ask_q1 = std::max<std::int64_t>(0, ob_vals[1]);
                cur_bid_p1 = ob_vals[2];
                cur_bid_q1 = std::max<std::int64_t>(0, ob_vals[3]);
                ob_row_ok = true;
            }
        }

        if (msg_line.empty()) {
            if (ob_row_ok) {
                prev_ask_p1 = cur_ask_p1;
                prev_ask_q1 = cur_ask_q1;
                prev_bid_p1 = cur_bid_p1;
                prev_bid_q1 = cur_bid_q1;
                have_prev_ob = true;
            }
            continue;
        }

        std::stringstream ss(msg_line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 6) continue;

        const double t_sec = std::strtod(cols[0].c_str(), nullptr);
        const int ev_type = std::atoi(cols[1].c_str());
        const std::int64_t size = std::strtoll(cols[3].c_str(), nullptr, 10);
        const std::int64_t price = std::strtoll(cols[4].c_str(), nullptr, 10);
        const int dir = std::atoi(cols[5].c_str());
        if (size <= 0 || price <= 0) continue;

        std::int64_t q_active = size;
        std::int64_t q_opp = size;
        std::int64_t spread_ticks = 1;

        if (have_prev_ob || ob_row_ok) {
            const std::int64_t ask_p1 = have_prev_ob ? prev_ask_p1 : cur_ask_p1;
            const std::int64_t ask_q1 = have_prev_ob ? prev_ask_q1 : cur_ask_q1;
            const std::int64_t bid_p1 = have_prev_ob ? prev_bid_p1 : cur_bid_p1;
            const std::int64_t bid_q1 = have_prev_ob ? prev_bid_q1 : cur_bid_q1;
            q_active = (dir > 0) ? bid_q1 : ask_q1;
            q_opp = (dir > 0) ? ask_q1 : bid_q1;
            spread_ticks = std::max<std::int64_t>(1, (ask_p1 - bid_p1) / 100);
        }

        if (ob_row_ok) {
            prev_ask_p1 = cur_ask_p1;
            prev_ask_q1 = cur_ask_q1;
            prev_bid_p1 = cur_bid_p1;
            prev_bid_q1 = cur_bid_q1;
            have_prev_ob = true;
        }

        const std::int64_t t_us = static_cast<std::int64_t>(t_sec * 1e6);
        ++seq;
        out_arrivals.push_back(ArrivalEvent{
            .seq = seq,
            .time = t_us,
            .mark = ev_type,
            .queue_depth = q_active,
            .opp_queue_depth = q_opp,
            .spread_ticks = spread_ticks
        });
        if (ev_type == 4 || ev_type == 5) {
            out_trades.push_back(SignedTrade{
                .seq = seq,
                .ts_ns = static_cast<std::uint64_t>(t_us) * 1000ULL,
                .price = price,
                .volume = size,
                .side = static_cast<std::int8_t>(dir >= 0 ? 1 : -1)
            });
        }
    }
    return !out_arrivals.empty();
}

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
static_assert(!has_matching_feedback_v<HawkesMleResult>,
              "HawkesMleResult must not expose a matching-feedback member");
static_assert(!has_matching_feedback_v<VpinResult>,
              "VpinResult must not expose a matching-feedback member");
static_assert(!is_matching_kernel_input_v<HawkesMleResult>,
              "HawkesMleResult is an offline diagnostic, never a kernel input");
static_assert(!is_matching_kernel_input_v<VolumeBucket>,
              "VolumeBucket is an offline diagnostic, never a kernel input");

// ============================================================================
// 6-CHANNEL SIGN-AWARE QUEUE-REACTIVE HAWKES MLE (docs/39 blueprint, Area 3)
// ============================================================================

// Channel encoding for the six event streams.
//   0 = BuyLimit   1 = SellLimit    (baseline modulated by phi = 1/(1+g*q))
//   2 = BuyCancel  3 = SellCancel   (baseline modulated by phi = 1 + g*q)
//   4 = BuyMarket  5 = SellMarket   (baseline modulated by phi = 1 + g*q)
inline constexpr std::uint8_t kSixChannelBuyLimit = 0;
inline constexpr std::uint8_t kSixChannelSellLimit = 1;

// LOBSTER event type (ArrivalEvent::mark) + side -> six-channel id.
//   mark 1        -> limit submission (addition)
//   mark 2 or 3   -> cancel / delete
//   mark 4 or 5   -> execution (visible / hidden), aggressive flow
//   anything else -> execution bucket (conservative)
inline std::uint8_t six_channel_from_lobster(std::int32_t mark, bool is_buy) noexcept {
    const std::uint8_t side_bit = is_buy ? 0 : 1;
    // Group index: 0 = limit additions, 1 = cancels/deletes, 2 = executions.
    std::uint8_t group = 2;
    if (mark == 1) group = 0;
    else if (mark == 2 || mark == 3) group = 1;
    return static_cast<std::uint8_t>(2 * group + side_bit);
}

struct SixChannelEvent {
    EventTime time = 0;
    std::uint8_t channel = 0;  // 0..5, see constants above
    std::int64_t queue_depth = 0;
};

// Per-channel analytical quantities. SoA layout over the 6 channels: the
// per-channel loops below auto-vectorize to AVX2 under the project's
// -mavx2 -O3 (QRLOB_ARCH_FLAGS); no intrinsics required for [6]-wide doubles.
struct HawkesSixChannelEval {
    double log_likelihood = 0.0;
    double grad_mu[6] = {};
    double grad_gamma[6] = {};
    double grad_alpha = 0.0;
    double grad_beta = 0.0;
    double fisher_mu[6] = {};
    double fisher_gamma[6] = {};
    double fisher_alpha = 0.0;
    double fisher_beta = 0.0;
};

// Sign-aware per-channel baseline modulator:
//   additions (channel 0/1):  phi = 1 / (1 + gamma * q)      in (0, 1]
//   removals  (channel 2..5): phi = 1 + gamma * q            in [phi_min, ...)
// Both keep 1 + gamma*q bounded away from zero via phi_floor.
inline double six_channel_phi(std::uint8_t channel, double gamma, double q,
                              double phi_floor = 0.05) noexcept {
    const double gq = gamma * q;
    if (channel < 2) {
        return 1.0 / std::max(phi_floor, 1.0 + gq);
    }
    return std::max(phi_floor, 1.0 + gq);
}

// Exact O(N) six-channel Ozaki log-likelihood with shared single-index
// excitation (alpha, beta common to all channels) and per-channel
// (mu_m, gamma_m) baselines. Analytical score and observed Fisher diagonal
// follow the same compensator algebra as the scalar evaluator above.
static HawkesSixChannelEval evaluate_log_likelihood_6ch(
    std::span<const SixChannelEvent> events,
    const double* mu,
    double alpha,
    double beta,
    const double* gamma,
    double q_bar_override = 0.0) noexcept {
    HawkesSixChannelEval out{};
    const std::size_t n = events.size();
    if (n == 0 || beta <= 0.0) return out;

    double q_bar = q_bar_override;
    if (q_bar <= 0.0) {
        double sum_q = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            sum_q += static_cast<double>(std::max<std::int64_t>(0, events[k].queue_depth));
        }
        q_bar = std::max(1.0, sum_q / static_cast<double>(n));
    } else {
        q_bar = std::max(1.0, q_bar);
    }

    const double inv_beta = 1.0 / beta;
    const double inv_beta2 = inv_beta * inv_beta;
    const double inv_beta3 = inv_beta2 * inv_beta;

    double r = 0.0;
    double rp = 0.0;
    double rpp = 0.0;

    for (std::size_t k = 0; k < n; ++k) {
        const std::uint8_t ch = static_cast<std::uint8_t>(events[k].channel % 6U);
        const double q_k = static_cast<double>(
            std::max<std::int64_t>(0, events[k].queue_depth)) / q_bar;

        if (k > 0) {
            const std::int64_t raw_dt_us = events[k].time - events[k - 1].time;
            const double dt = (raw_dt_us > 0)
                ? (static_cast<double>(raw_dt_us) * 1e-6)
                : 1e-7;
            const std::uint8_t chp = static_cast<std::uint8_t>(events[k - 1].channel % 6U);
            const double q_prev = static_cast<double>(
                std::max<std::int64_t>(0, events[k - 1].queue_depth)) / q_bar;
            const double phi_prev = six_channel_phi(chp, gamma[chp], q_prev);
            const double bdt = beta * dt;
            const double e_bdt = std::exp(-bdt);
            const double one_m_e = 1.0 - e_bdt;
            const double r_plus = r + 1.0;
            const double rp_prev = rp;
            const double rpp_prev = rpp;

            // Compensator of the shared excitation + previous channel baseline.
            const double comp_k =
                mu[chp] * phi_prev * dt + (alpha * inv_beta) * r_plus * one_m_e;
            out.log_likelihood -= comp_k;

            out.grad_mu[chp] -= phi_prev * dt;
            out.grad_alpha -= (r_plus * inv_beta) * one_m_e;
            if (chp < 2) {
                const double denom = std::max(0.05, 1.0 + gamma[chp] * q_prev);
                out.grad_gamma[chp] += mu[chp] * dt * q_prev / (denom * denom);
            } else {
                out.grad_gamma[chp] -= mu[chp] * dt * q_prev;
            }

            const double d_comp_dbeta =
                (alpha * inv_beta) * rp_prev * one_m_e +
                alpha * r_plus * (bdt * e_bdt - one_m_e) * inv_beta2;
            const double d2_comp_dbeta2 =
                (alpha * inv_beta) * rpp_prev * one_m_e +
                2.0 * alpha * rp_prev * (bdt * e_bdt - one_m_e) * inv_beta2 +
                alpha * r_plus * (-(bdt * bdt) * e_bdt - 2.0 * bdt * e_bdt + 2.0 * one_m_e) * inv_beta3;
            out.grad_beta -= d_comp_dbeta;
            out.fisher_beta += d2_comp_dbeta2;

            // Baseline compensator second-order (mu and gamma of prev channel).
            out.fisher_mu[chp] += 0.0;  // comp linear in mu: no second-order term
            if (chp < 2) {
                const double denom = std::max(0.05, 1.0 + gamma[chp] * q_prev);
                const double d2 = 2.0 * mu[chp] * dt * q_prev * q_prev / (denom * denom * denom);
                out.fisher_gamma[chp] += d2;
            }

            r = e_bdt * r_plus;
            rp = e_bdt * (rp_prev - dt * r_plus);
            rpp = e_bdt * (rpp_prev - 2.0 * dt * rp_prev + (dt * dt) * r_plus);
        }

        const double phi_k = six_channel_phi(ch, gamma[ch], q_k);
        const double lam = std::max(1e-12, mu[ch] * phi_k + alpha * r);
        out.log_likelihood += std::log(lam);

        const double inv_lam = 1.0 / lam;
        const double inv_lam2 = inv_lam * inv_lam;

        const double d_lam_dmu = phi_k;
        const double d_lam_dalpha = r;
        const double d_lam_dbeta = alpha * rp;
        const double d2_lam_dbeta2 = alpha * rpp;
        double d_lam_dgamma;
        double d2_lam_dgamma2;
        if (ch < 2) {
            const double denom = std::max(0.05, 1.0 + gamma[ch] * q_k);
            d_lam_dgamma = -mu[ch] * q_k / (denom * denom);
            d2_lam_dgamma2 = 2.0 * mu[ch] * q_k * q_k / (denom * denom * denom);
        } else {
            d_lam_dgamma = mu[ch] * q_k;
            d2_lam_dgamma2 = 0.0;
        }

        out.grad_mu[ch] += d_lam_dmu * inv_lam;
        out.grad_alpha += d_lam_dalpha * inv_lam;
        out.grad_beta += d_lam_dbeta * inv_lam;
        out.grad_gamma[ch] += d_lam_dgamma * inv_lam;

        out.fisher_mu[ch] += (d_lam_dmu * d_lam_dmu) * inv_lam2;
        out.fisher_alpha += (d_lam_dalpha * d_lam_dalpha) * inv_lam2;
        out.fisher_beta += (d_lam_dbeta * d_lam_dbeta) * inv_lam2 - d2_lam_dbeta2 * inv_lam;
        out.fisher_gamma[ch] += (d_lam_dgamma * d_lam_dgamma) * inv_lam2
                                - d2_lam_dgamma2 * inv_lam;
    }
    return out;
}

// Six-channel calibration: stage 1 fits the shared (alpha, beta) excitation
// with the scalar single-channel model; stage 2 runs projected damped Newton
// with Armijo backtracking on the 12 baselines (mu[6], gamma[6]) with the
// excitation held fixed, using the analytical per-channel score and Fisher
// diagonal from evaluate_log_likelihood_6ch. Newton steps are clamped per
// channel and the Fisher diagonal is floored for conditioning.
struct HawkesSixChannelResult {
    double mu[6] = {};
    double gamma[6] = {};
    double alpha = 0.0;
    double beta = 0.0;
    double log_likelihood = 0.0;
    double score_max_abs = 0.0;
    double fisher_min = 0.0;
    std::uint32_t iterations = 0;
    bool converged = false;
};

static HawkesSixChannelResult calibrate_mle_6ch(
    std::span<const SixChannelEvent> events,
    double train_fraction = 0.70) {
    HawkesSixChannelResult res{};

    // Stage 1: shared excitation via the scalar model on mark-agnostic events.
    std::vector<ArrivalEvent> scalar_events;
    scalar_events.reserve(events.size());
    for (const SixChannelEvent& e : events) {
        scalar_events.push_back(ArrivalEvent{
            .seq = 0, .time = e.time, .mark = 1,
            .queue_depth = e.queue_depth, .opp_queue_depth = 0, .spread_ticks = 1});
    }
    const std::size_t n_total = events.size();
    if (n_total < 32) return res;
    const std::size_t n_train = std::max<std::size_t>(
        16, static_cast<std::size_t>(static_cast<double>(n_total) * train_fraction));
    const HawkesMleResult scalar =
        StateDependentHawkesCalibrator::calibrate_mle(
            std::span<const ArrivalEvent>(scalar_events.data(), n_train), train_fraction);
    res.alpha = scalar.alpha_hz;
    res.beta = scalar.beta_hz;
    if (!(res.beta > 0.0)) return res;

    // Stage 2 initialization: per-channel Poisson rates; gamma at neutral.
    std::size_t counts[6] = {};
    for (std::size_t k = 0; k < n_train; ++k) {
        ++counts[events[k].channel % 6U];
    }
    const double span_s = static_cast<double>(
        events[n_train - 1].time - events[0].time) * 1e-6;
    const double span = std::max(1e-6, span_s);
    for (int c = 0; c < 6; ++c) {
        res.mu[c] = std::max(1e-3, static_cast<double>(counts[c]) / span);
        res.gamma[c] = 0.0;
    }

    const auto train = std::span<const SixChannelEvent>(events.data(), n_train);
    HawkesSixChannelEval ev =
        evaluate_log_likelihood_6ch(train, res.mu, res.alpha, res.beta, res.gamma);
    double ll = ev.log_likelihood;
    const double phi_floor = 0.05;

    constexpr std::uint32_t kMaxIters = 60;
    for (std::uint32_t it = 0; it < kMaxIters; ++it) {
        res.iterations = it + 1;
        double step_norm = 0.0;
        bool improved = false;

        for (int c = 0; c < 6; ++c) {
            // Joint damped-Newton step on (mu_c, gamma_c) using the Fisher
            // diagonal as the damping metric (per-channel 2x2 system).
            const double g_mu = ev.grad_mu[c];
            const double g_ga = ev.grad_gamma[c];
            const double f_mu = std::max(1e-8, ev.fisher_mu[c]);
            const double f_ga = std::max(1e-8, ev.fisher_gamma[c]);
            const double d_mu = std::clamp(g_mu / f_mu, -0.50, 0.50);
            const double d_ga = std::clamp(g_ga / f_ga, -0.50, 0.50);

            double step = 1.0;
            bool took = false;
            for (int bt = 0; bt < 12; ++bt) {
                const double new_mu = std::max(1e-6, res.mu[c] + step * d_mu);
                const double new_ga = std::clamp(res.gamma[c] + step * d_ga,
                                                 -0.90, 10.0);
                double mu_t[6];
                double ga_t[6];
                for (int j = 0; j < 6; ++j) {
                    mu_t[j] = res.mu[j];
                    ga_t[j] = res.gamma[j];
                }
                mu_t[c] = new_mu;
                ga_t[c] = new_ga;
                const HawkesSixChannelEval trial =
                    evaluate_log_likelihood_6ch(train, mu_t, res.alpha, res.beta, ga_t);
                if (trial.log_likelihood >= ll - 1e-12) {
                    res.mu[c] = new_mu;
                    res.gamma[c] = new_ga;
                    ll = trial.log_likelihood;
                    ev = trial;
                    took = true;
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            step_norm = std::max(step_norm, std::fabs(d_mu) + std::fabs(d_ga));
            if (!took) break;  // no admissible step: stop on this sweep
        }

        if (!improved || step_norm < 1e-6) {
            if (step_norm < 1e-6) res.converged = true;
            break;
        }
    }

    (void)phi_floor;
    res.log_likelihood = ll;
    res.score_max_abs = 0.0;
    for (int c = 0; c < 6; ++c) {
        res.score_max_abs = std::max(res.score_max_abs,
                                     std::fabs(ev.grad_mu[c]));
        res.score_max_abs = std::max(res.score_max_abs,
                                     std::fabs(ev.grad_gamma[c]));
        res.fisher_min = (c == 0)
            ? std::min(ev.fisher_mu[c], ev.fisher_gamma[c])
            : std::min(res.fisher_min,
                       std::min(ev.fisher_mu[c], ev.fisher_gamma[c]));
    }
    return res;
}

static_assert(!is_matching_kernel_input_v<VpinResult>,
              "VpinResult is an offline diagnostic, never a kernel input");

} // namespace experimental
} // namespace sim
