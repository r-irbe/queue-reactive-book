// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "export_schema.hpp"

#include <array>
#include <cstddef>

namespace sim {

template <std::size_t Rows, std::size_t Cols>
inline std::array<ExportColumn, Rows * Cols> moving_window_feature_schema() noexcept {
    std::array<ExportColumn, Rows * Cols> out{};
    std::size_t pos = 0;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t col = 0; col < Cols; ++col) {
            out[pos++] = ExportColumn{.name = "window_value", .row = row, .col = col};
        }
    }
    return out;
}

template <std::size_t Rows, std::size_t Cols>
class MovingFeatureWindow {
public:
    using Row = std::array<double, Cols>;
    using Tensor = std::array<double, Rows * Cols>;

    void push(const Row& row) noexcept {
        rows_[write_] = row;
        write_ = (write_ + 1U) % Rows;
        if (count_ < Rows) ++count_;
    }

    void clear() noexcept {
        rows_ = {};
        write_ = 0;
        count_ = 0;
    }

    std::size_t count() const noexcept { return count_; }
    bool full() const noexcept { return count_ == Rows; }
    bool empty() const noexcept { return count_ == 0; }

    Row row(std::size_t chronological_index) const noexcept {
        if (chronological_index >= count_) return {};
        return rows_[physical_index(chronological_index)];
    }

    Tensor export_tensor() const noexcept {
        Tensor out{};
        const std::size_t pad_rows = Rows - count_;
        for (std::size_t i = 0; i < count_; ++i) {
            const Row current = row(i);
            const std::size_t output_row = pad_rows + i;
            for (std::size_t col = 0; col < Cols; ++col) {
                out[output_row * Cols + col] = current[col];
            }
        }
        return out;
    }

private:
    std::array<Row, Rows> rows_{};
    std::size_t write_ = 0;
    std::size_t count_ = 0;

    std::size_t oldest_index() const noexcept {
        return full() ? write_ : 0;
    }

    std::size_t physical_index(std::size_t chronological_index) const noexcept {
        return (oldest_index() + chronological_index) % Rows;
    }
};

} // namespace sim
