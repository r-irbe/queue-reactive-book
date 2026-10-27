// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include <cmath>
#include <algorithm>

namespace Catch {

class Approx {
public:
    explicit Approx(double val) noexcept
        : val_(val), epsilon_(1e-5), margin_(1e-12) {}

    static Approx custom() noexcept { return Approx(0.0); }

    Approx& epsilon(double eps) noexcept {
        epsilon_ = eps;
        return *this;
    }

    Approx& margin(double m) noexcept {
        margin_ = m;
        return *this;
    }

    friend bool operator==(double lhs, const Approx& rhs) noexcept {
        return std::abs(lhs - rhs.val_) <= rhs.margin_ ||
               std::abs(lhs - rhs.val_) <= rhs.epsilon_ * (std::abs(lhs) + std::abs(rhs.val_));
    }

    friend bool operator==(const Approx& lhs, double rhs) noexcept {
        return rhs == lhs;
    }

    friend bool operator!=(double lhs, const Approx& rhs) noexcept {
        return !(lhs == rhs);
    }

    friend bool operator!=(const Approx& lhs, double rhs) noexcept {
        return !(lhs == rhs);
    }

    friend bool operator<=(double lhs, const Approx& rhs) noexcept {
        return lhs < rhs.val_ || lhs == rhs;
    }

    friend bool operator<=(const Approx& lhs, double rhs) noexcept {
        return lhs.val_ < rhs || lhs == rhs;
    }

    friend bool operator>=(double lhs, const Approx& rhs) noexcept {
        return lhs > rhs.val_ || lhs == rhs;
    }

    friend bool operator>=(const Approx& lhs, double rhs) noexcept {
        return lhs.val_ > rhs || lhs == rhs;
    }

    double value() const noexcept { return val_; }

private:
    double val_;
    double epsilon_;
    double margin_;
};

} // namespace Catch
