// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// Lock-free POSIX shared-memory SPSC ring buffer (`ShmIpcEvent`, `ShmIpcHeader`,
// `ShmRingChannel`, `ShmSpscProducer`, `ShmSpscConsumer`, `ShmSpscEndpoint`,
// `QrlobShmMicrostructureExt`, `ShmSpscFeatureExportSink`) with 64-byte
// cache-aligned binary event frames and unified leangrph ABI support.

#pragma once

#include "sim/feature_export_contract.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sim {
namespace experimental {

// Legacy magic identifier: "SHM_IPC1"
inline constexpr std::uint64_t kShmMagic = 0x53484D5F49504331ULL;
// Unified leangrph / rust_mmap_engine magic identifier: "LEANRINC"
inline constexpr std::uint64_t kLeangrphShmMagic = 0x434E49524E41454CULL;
inline constexpr std::uint32_t kShmVersion = 1;
inline constexpr std::size_t kLeangrphRingCapacity = 1024;
inline constexpr std::size_t kRingCapacity = 65536;
inline constexpr std::size_t kRingMask = kRingCapacity - 1;
inline constexpr const char* kDefaultShmPath = "/dev/shm/leangrph_ipc";
inline constexpr std::uint16_t kShmFlagQrlobTelemetry = 0x4000U;

[[nodiscard]] inline constexpr bool is_valid_shm_magic(std::uint64_t magic) noexcept {
    return magic == kShmMagic || magic == kLeangrphShmMagic;
}

// 16-byte Microstructure Telemetry Extension overlaid on ShmIpcEvent::_reserved[16]
// when (ShmIpcEvent::flags & kShmFlagQrlobTelemetry) != 0.
struct QrlobShmMicrostructureExt {
    std::int32_t  ofi_q16;
    std::uint32_t hawkes_intensity_q16;
    std::uint16_t nr3d_vol_q8;
    std::uint16_t vpin_bps;
    std::uint16_t spread_ticks;
    std::uint8_t  level_and_regime;
    std::int8_t   reprice_shift;
};

static_assert(sizeof(QrlobShmMicrostructureExt) == 16,
              "QrlobShmMicrostructureExt must be exactly 16 bytes");
static_assert(std::is_trivially_copyable_v<QrlobShmMicrostructureExt>,
              "QrlobShmMicrostructureExt must be trivially copyable");

// 64-byte Contiguous Binary IPC Event (matches rust_mmap_engine::shm_spsc_ipc::ShmIpcEvent)
struct alignas(64) ShmIpcEvent {
    std::uint64_t seq;
    std::uint64_t timestamp_ns;
    std::uint64_t order_id;
    std::uint32_t price;
    std::uint32_t qty;
    std::uint8_t  side;        // 0 = Bid, 1 = Ask
    std::uint8_t  event_type;  // 1 = Add, 2 = Cancel, 3 = Executed/Ack, 4 = Telemetry/Snapshot
    std::uint16_t flags;
    std::uint32_t client_id;
    std::uint64_t latency_tsc;
    std::uint8_t  _reserved[16];
};

static_assert(sizeof(ShmIpcEvent) == 64, "ShmIpcEvent must be exactly 64 bytes");
static_assert(offsetof(ShmIpcEvent, _reserved) == 48,
              "ShmIpcEvent::_reserved must start at byte offset 48");

inline void pack_microstructure_ext(ShmIpcEvent& ev,
                                    const QrlobShmMicrostructureExt& ext) noexcept {
    ev.flags = static_cast<std::uint16_t>(ev.flags | kShmFlagQrlobTelemetry);
    std::memcpy(ev._reserved, &ext, sizeof(QrlobShmMicrostructureExt));
}

[[nodiscard]] inline bool unpack_microstructure_ext(
    const ShmIpcEvent& ev,
    QrlobShmMicrostructureExt& out_ext) noexcept {
    if ((ev.flags & kShmFlagQrlobTelemetry) == 0U) {
        return false;
    }
    std::memcpy(&out_ext, ev._reserved, sizeof(QrlobShmMicrostructureExt));
    return true;
}

// 128-byte aligned IPC Region Header
struct alignas(128) ShmIpcHeader {
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t status;
    std::uint32_t capacity;
    std::uint32_t req_offset;
    std::uint32_t resp_offset;
    std::uint8_t  _pad[100];
};

static_assert(sizeof(ShmIpcHeader) == 128, "ShmIpcHeader must be exactly 128 bytes");

[[nodiscard]] inline constexpr bool validate_shm_header(const ShmIpcHeader* hdr) noexcept {
    if (!hdr) return false;
    if (!is_valid_shm_magic(hdr->magic)) return false;
    if (hdr->version != kShmVersion) return false;
    if (hdr->capacity != kRingCapacity && hdr->capacity != kLeangrphRingCapacity) return false;
    return true;
}

// 128-byte cache-line isolated SPSC Ring Channel
struct alignas(128) ShmRingChannel {
    std::atomic<std::uint64_t> head;
    std::uint8_t _pad_head[120];
    std::atomic<std::uint64_t> tail;
    std::uint8_t _pad_tail[120];
    ShmIpcEvent slots[kRingCapacity];
};

static_assert(offsetof(ShmRingChannel, head) == 0, "head must be at offset 0");
static_assert(offsetof(ShmRingChannel, tail) == 128, "tail must be at offset 128");

// Compact 1024-slot SPSC Ring Channel matching rust_mmap_engine::shm_spsc_ipc
struct alignas(128) LeangrphShmRingChannel {
    std::atomic<std::uint64_t> head;
    std::uint8_t _pad_head[120];
    std::atomic<std::uint64_t> tail;
    std::uint8_t _pad_tail[120];
    ShmIpcEvent slots[kLeangrphRingCapacity];
};

static_assert(offsetof(LeangrphShmRingChannel, head) == 0, "head must be at offset 0");
static_assert(offsetof(LeangrphShmRingChannel, tail) == 128, "tail must be at offset 128");

inline constexpr std::size_t kTotalShmSize = sizeof(ShmIpcHeader) + 2 * sizeof(ShmRingChannel);
inline constexpr std::size_t kLeangrphTotalShmSize =
    sizeof(ShmIpcHeader) + 2 * sizeof(LeangrphShmRingChannel);

// Zero-allocation SPSC Producer for Matching Engine Telemetry / IPC
class ShmSpscProducer {
public:
    ShmSpscProducer() noexcept
        : channel_(nullptr), cached_head_(0), seq_(0), capacity_(kRingCapacity), mask_(kRingMask) {}

    explicit ShmSpscProducer(ShmRingChannel* channel,
                             std::size_t capacity = kRingCapacity) noexcept
        : channel_(channel),
          cached_head_(0),
          seq_(0),
          capacity_((capacity == kLeangrphRingCapacity) ? kLeangrphRingCapacity : kRingCapacity),
          mask_(capacity_ - 1) {}

    void bind_channel(ShmRingChannel* channel,
                      std::size_t capacity = kRingCapacity) noexcept {
        channel_ = channel;
        cached_head_ = 0;
        seq_ = 0;
        capacity_ = (capacity == kLeangrphRingCapacity) ? kLeangrphRingCapacity : kRingCapacity;
        mask_ = capacity_ - 1;
    }

    [[nodiscard]] bool is_bound() const noexcept {
        return channel_ != nullptr;
    }

    // Direct non-blocking enqueue into pre-allocated slot.
    // Zero dynamic memory allocation on the hot path.
    [[gnu::hot]] bool try_enqueue(const ShmIpcEvent& ev) noexcept {
        if (!channel_) [[unlikely]] return false;

        const std::uint64_t current_tail = channel_->tail.load(std::memory_order_relaxed);
        if (current_tail - cached_head_ >= capacity_) {
            cached_head_ = channel_->head.load(std::memory_order_acquire);
            if (current_tail - cached_head_ >= capacity_) {
                return false; // Ring buffer full
            }
        }

        const std::size_t idx = current_tail & mask_;
        ShmIpcEvent* slot = &channel_->slots[idx];

        std::memcpy(slot, &ev, sizeof(ShmIpcEvent));

#if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();
#else
        std::atomic_thread_fence(std::memory_order_release);
#endif

        channel_->tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    [[gnu::hot]] bool publish_add(std::uint64_t order_id, std::uint32_t price,
                                  std::uint32_t qty, std::uint8_t side,
                                  std::uint64_t ts_ns, std::uint32_t client_id = 0) noexcept {
        ShmIpcEvent ev{};
        ev.seq = ++seq_;
        ev.timestamp_ns = ts_ns;
        ev.order_id = order_id;
        ev.price = price;
        ev.qty = qty;
        ev.side = side;
        ev.event_type = 1; // Add
        ev.flags = 0;
        ev.client_id = client_id;
        return try_enqueue(ev);
    }

    [[gnu::hot]] bool publish_cancel(std::uint64_t order_id, std::uint32_t price,
                                     std::uint32_t qty, std::uint8_t side,
                                     std::uint64_t ts_ns) noexcept {
        ShmIpcEvent ev{};
        ev.seq = ++seq_;
        ev.timestamp_ns = ts_ns;
        ev.order_id = order_id;
        ev.price = price;
        ev.qty = qty;
        ev.side = side;
        ev.event_type = 2; // Cancel
        ev.flags = 0;
        return try_enqueue(ev);
    }

    [[gnu::hot]] bool publish_fill(std::uint64_t maker_id, std::uint64_t taker_id,
                                   std::uint32_t price, std::uint32_t fill_qty,
                                   std::uint8_t maker_side, std::uint64_t ts_ns) noexcept {
        ShmIpcEvent ev{};
        ev.seq = ++seq_;
        ev.timestamp_ns = ts_ns;
        ev.order_id = maker_id;
        ev.price = price;
        ev.qty = fill_qty;
        ev.side = maker_side;
        ev.event_type = 3; // Executed/Ack
        ev.flags = 0x8000;
        ev.client_id = static_cast<std::uint32_t>(taker_id & 0xFFFFFFFFULL);
        return try_enqueue(ev);
    }

    [[gnu::hot]] bool publish_microstructure(
        std::uint64_t order_id,
        std::uint32_t price,
        std::uint32_t qty,
        std::uint8_t side,
        std::uint8_t event_type,
        std::uint64_t ts_ns,
        const QrlobShmMicrostructureExt& ext,
        std::uint32_t client_id = 0,
        std::uint64_t latency_tsc = 0) noexcept {
        ShmIpcEvent ev{};
        ev.seq = ++seq_;
        ev.timestamp_ns = ts_ns;
        ev.order_id = order_id;
        ev.price = price;
        ev.qty = qty;
        ev.side = side;
        ev.event_type = event_type;
        ev.flags = kShmFlagQrlobTelemetry;
        ev.client_id = client_id;
        ev.latency_tsc = latency_tsc;
        pack_microstructure_ext(ev, ext);
        return try_enqueue(ev);
    }

    std::uint64_t current_seq() const noexcept { return seq_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    ShmRingChannel* channel_;
    std::uint64_t cached_head_;
    std::uint64_t seq_;
    std::size_t capacity_;
    std::size_t mask_;
};

// Zero-allocation SPSC Consumer for Replay / Bridge Verification
class ShmSpscConsumer {
public:
    ShmSpscConsumer() noexcept
        : channel_(nullptr), cached_tail_(0), capacity_(kRingCapacity), mask_(kRingMask) {}

    explicit ShmSpscConsumer(ShmRingChannel* channel,
                             std::size_t capacity = kRingCapacity) noexcept
        : channel_(channel),
          cached_tail_(0),
          capacity_((capacity == kLeangrphRingCapacity) ? kLeangrphRingCapacity : kRingCapacity),
          mask_(capacity_ - 1) {}

    void bind_channel(ShmRingChannel* channel,
                      std::size_t capacity = kRingCapacity) noexcept {
        channel_ = channel;
        cached_tail_ = 0;
        capacity_ = (capacity == kLeangrphRingCapacity) ? kLeangrphRingCapacity : kRingCapacity;
        mask_ = capacity_ - 1;
    }

    [[nodiscard]] bool is_bound() const noexcept {
        return channel_ != nullptr;
    }

    [[gnu::hot]] bool try_dequeue(ShmIpcEvent& out) noexcept {
        if (!channel_) [[unlikely]] return false;

        const std::uint64_t current_head = channel_->head.load(std::memory_order_relaxed);
        if (current_head == cached_tail_) {
            cached_tail_ = channel_->tail.load(std::memory_order_acquire);
            if (current_head == cached_tail_) {
                return false; // Ring buffer empty
            }
        }

        const std::size_t idx = current_head & mask_;
        std::memcpy(&out, &channel_->slots[idx], sizeof(ShmIpcEvent));
        channel_->head.store(current_head + 1, std::memory_order_release);
        return true;
    }

    std::size_t capacity() const noexcept { return capacity_; }

private:
    ShmRingChannel* channel_;
    std::uint64_t cached_tail_;
    std::size_t capacity_;
    std::size_t mask_;
};

// POSIX Shared Memory Endpoint Manager
class ShmSpscEndpoint {
public:
    ShmSpscEndpoint() noexcept
        : ptr_(nullptr),
          len_(0),
          fd_(-1),
          is_creator_(false),
          magic_(kShmMagic),
          ring_capacity_(kRingCapacity) {}

    ~ShmSpscEndpoint() {
        close();
    }

    bool open(const char* path = kDefaultShmPath,
              bool create = true,
              bool use_leangrph_abi = false) noexcept {
        close();

        magic_ = use_leangrph_abi ? kLeangrphShmMagic : kShmMagic;
        ring_capacity_ = use_leangrph_abi ? kLeangrphRingCapacity : kRingCapacity;
        const std::size_t shm_bytes =
            use_leangrph_abi ? kLeangrphTotalShmSize : kTotalShmSize;

        int flags = O_RDWR;
        if (create) {
            flags |= O_CREAT;
        }

        fd_ = ::open(path, flags, 0666);
        if (fd_ < 0) {
            return fallback_anonymous(shm_bytes);
        }

        if (create) {
            if (::ftruncate(fd_, static_cast<off_t>(shm_bytes)) != 0) {
                ::close(fd_);
                fd_ = -1;
                return fallback_anonymous(shm_bytes);
            }
            is_creator_ = true;
        } else {
            struct stat st{};
            if (::fstat(fd_, &st) == 0 && st.st_size >= static_cast<off_t>(sizeof(ShmIpcHeader))) {
                const std::size_t file_len = static_cast<std::size_t>(st.st_size);
                void* m = ::mmap(nullptr, file_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
                if (m == MAP_FAILED) {
                    ::close(fd_);
                    fd_ = -1;
                    return false;
                }
                ptr_ = static_cast<std::uint8_t*>(m);
                len_ = file_len;
                const auto* hdr = header();
                if (!validate_shm_header(hdr)) {
                    close();
                    return false;
                }
                magic_ = hdr->magic;
                ring_capacity_ = hdr->capacity;
                return true;
            }
        }

        void* m = ::mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (m == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            return fallback_anonymous(shm_bytes);
        }

        ptr_ = static_cast<std::uint8_t*>(m);
        len_ = shm_bytes;

        if (create) {
            init_header();
        } else if (!validate_shm_header(header())) {
            close();
            return false;
        }
        return true;
    }

    void close() noexcept {
        if (ptr_ && ptr_ != MAP_FAILED) {
            ::munmap(ptr_, len_);
            ptr_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    [[nodiscard]] ShmIpcHeader* header() noexcept {
        return reinterpret_cast<ShmIpcHeader*>(ptr_);
    }

    [[nodiscard]] const ShmIpcHeader* header() const noexcept {
        return reinterpret_cast<const ShmIpcHeader*>(ptr_);
    }

    [[nodiscard]] ShmRingChannel* request_channel() noexcept {
        return reinterpret_cast<ShmRingChannel*>(ptr_ + sizeof(ShmIpcHeader));
    }

    [[nodiscard]] ShmRingChannel* response_channel() noexcept {
        const std::size_t ch_bytes =
            (ring_capacity_ == kLeangrphRingCapacity)
                ? sizeof(LeangrphShmRingChannel)
                : sizeof(ShmRingChannel);
        return reinterpret_cast<ShmRingChannel*>(ptr_ + sizeof(ShmIpcHeader) + ch_bytes);
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return ptr_ != nullptr && ptr_ != MAP_FAILED && validate_shm_header(header());
    }

    [[nodiscard]] std::size_t ring_capacity() const noexcept {
        return ring_capacity_;
    }

private:
    bool fallback_anonymous(std::size_t shm_bytes) noexcept {
        void* m = ::mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            return false;
        }
        ptr_ = static_cast<std::uint8_t*>(m);
        len_ = shm_bytes;
        init_header();
        return true;
    }

    void init_header() noexcept {
        std::memset(ptr_, 0, len_);
        auto* hdr = header();
        const std::size_t ch_bytes =
            (ring_capacity_ == kLeangrphRingCapacity)
                ? sizeof(LeangrphShmRingChannel)
                : sizeof(ShmRingChannel);
        hdr->magic = magic_;
        hdr->version = kShmVersion;
        hdr->status = 1;
        hdr->capacity = static_cast<std::uint32_t>(ring_capacity_);
        hdr->req_offset = sizeof(ShmIpcHeader);
        hdr->resp_offset = static_cast<std::uint32_t>(sizeof(ShmIpcHeader) + ch_bytes);
    }

    std::uint8_t* ptr_;
    std::size_t len_;
    int fd_;
    bool is_creator_;
    std::uint64_t magic_;
    std::size_t ring_capacity_;
};

// Zero-allocation FeatureExportSink bridging ExportFrame snapshots +
// 16-byte QrlobShmMicrostructureExt into a lock-free SPSC ring channel.
class ShmSpscFeatureExportSink final : public FeatureExportSink {
public:
    ShmSpscFeatureExportSink() noexcept = default;

    explicit ShmSpscFeatureExportSink(ShmRingChannel* channel,
                                      std::size_t capacity = kRingCapacity) noexcept
        : producer_(channel, capacity) {}

    void bind_channel(ShmRingChannel* channel,
                      std::size_t capacity = kRingCapacity) noexcept {
        producer_.bind_channel(channel, capacity);
    }

    void set_telemetry_hints(std::uint32_t hawkes_intensity_q16,
                             std::uint16_t nr3d_vol_q8,
                             std::uint16_t vpin_bps,
                             std::uint8_t level_and_regime = 0,
                             std::int8_t reprice_shift = 0) noexcept {
        hawkes_intensity_q16_ = hawkes_intensity_q16;
        nr3d_vol_q8_ = nr3d_vol_q8;
        vpin_bps_ = vpin_bps;
        level_and_regime_ = level_and_regime;
        reprice_shift_ = reprice_shift;
    }

    void on_frame(const ExportFrame& frame) override {
        if (have_last_ && frame.seq <= last_seq_) {
            ++seq_violations_;
            return;
        }

        const auto& snap = frame.snapshot;
        std::int64_t ofi_delta = 0;
        if (have_last_) {
            const std::int64_t db = snap.best_bid_qty - prev_bid_qty_;
            const std::int64_t da = snap.best_ask_qty - prev_ask_qty_;
            ofi_delta = db - da;
        }
        prev_bid_qty_ = snap.best_bid_qty;
        prev_ask_qty_ = snap.best_ask_qty;
        have_last_ = true;
        last_seq_ = frame.seq;
        last_hash_ = frame.replay_hash;

        QrlobShmMicrostructureExt ext{};
        const std::int64_t clamped_ofi = std::clamp<std::int64_t>(ofi_delta, -32767, 32767);
        ext.ofi_q16 = static_cast<std::int32_t>(clamped_ofi * 65536LL);
        ext.hawkes_intensity_q16 = hawkes_intensity_q16_;
        ext.nr3d_vol_q8 = nr3d_vol_q8_;
        ext.vpin_bps = vpin_bps_;
        ext.spread_ticks = static_cast<std::uint16_t>(
            std::clamp<std::int64_t>(snap.spread > 0 ? snap.spread : 1, 1, 65535));
        ext.level_and_regime = level_and_regime_;
        ext.reprice_shift = reprice_shift_;

        ShmIpcEvent ev{};
        ev.seq = frame.seq;
        ev.timestamp_ns = static_cast<std::uint64_t>(std::max<TimeNs>(0, snap.ts));
        ev.order_id = frame.seq;
        ev.price = static_cast<std::uint32_t>(std::max<Tick>(0, snap.best_bid));
        ev.qty = static_cast<std::uint32_t>(std::max<Qty>(0, snap.best_bid_qty));
        ev.side = (snap.imbalance_l1 >= 0.0) ? 0U : 1U;
        ev.event_type = 4U; // Telemetry / ExportFrame
        ev.flags = kShmFlagQrlobTelemetry;
        ev.client_id = static_cast<std::uint32_t>(frame.replay_hash & 0xFFFFFFFFULL);
        ev.latency_tsc = frame.replay_hash;
        pack_microstructure_ext(ev, ext);

        if (producer_.try_enqueue(ev)) {
            ++enqueued_frames_;
        } else {
            ++dropped_frames_;
        }
    }

    [[nodiscard]] std::size_t enqueued_frames() const noexcept { return enqueued_frames_; }
    [[nodiscard]] std::size_t dropped_frames() const noexcept { return dropped_frames_; }
    [[nodiscard]] std::size_t seq_violations() const noexcept { return seq_violations_; }
    [[nodiscard]] std::uint64_t last_hash() const noexcept { return last_hash_; }

private:
    ShmSpscProducer producer_{};
    EventSeq last_seq_ = 0;
    std::uint64_t last_hash_ = 0;
    Qty prev_bid_qty_ = 0;
    Qty prev_ask_qty_ = 0;
    std::uint32_t hawkes_intensity_q16_ = 65536U;
    std::uint16_t nr3d_vol_q8_ = 256U;
    std::uint16_t vpin_bps_ = 2500U;
    std::uint8_t level_and_regime_ = 1U;
    std::int8_t reprice_shift_ = 0;
    std::size_t enqueued_frames_ = 0;
    std::size_t dropped_frames_ = 0;
    std::size_t seq_violations_ = 0;
    bool have_last_ = false;
};

} // namespace experimental

using experimental::is_valid_shm_magic;
using experimental::kDefaultShmPath;
using experimental::kLeangrphRingCapacity;
using experimental::kLeangrphShmMagic;
using experimental::kLeangrphTotalShmSize;
using experimental::kRingCapacity;
using experimental::kRingMask;
using experimental::kShmFlagQrlobTelemetry;
using experimental::kShmMagic;
using experimental::kShmVersion;
using experimental::kTotalShmSize;
using experimental::LeangrphShmRingChannel;
using experimental::pack_microstructure_ext;
using experimental::QrlobShmMicrostructureExt;
using experimental::ShmIpcEvent;
using experimental::ShmIpcHeader;
using experimental::ShmRingChannel;
using experimental::ShmSpscConsumer;
using experimental::ShmSpscEndpoint;
using experimental::ShmSpscFeatureExportSink;
using experimental::ShmSpscProducer;
using experimental::unpack_microstructure_ext;
using experimental::validate_shm_header;

} // namespace sim
