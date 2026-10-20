// Lock-free POSIX shared-memory SPSC ring buffer (`ShmIpcEvent`, `ShmIpcHeader`,
// `ShmRingChannel`, `ShmSpscProducer`, `ShmSpscConsumer`, `ShmSpscEndpoint`)
// with 64-byte cache-aligned binary event frames.

#pragma once

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

// Magic identifier: "SHM_IPC1"
inline constexpr std::uint64_t kShmMagic = 0x53484D5F49504331ULL;
inline constexpr std::uint32_t kShmVersion = 1;
inline constexpr std::size_t kRingCapacity = 65536;
inline constexpr std::size_t kRingMask = kRingCapacity - 1;
inline constexpr const char* kDefaultShmPath = "/dev/shm/leangrph_ipc";

// 64-byte Contiguous Binary IPC Event (matches rust_mmap_engine::shm_spsc_ipc::ShmIpcEvent)
struct alignas(64) ShmIpcEvent {
    std::uint64_t seq;
    std::uint64_t timestamp_ns;
    std::uint64_t order_id;
    std::uint32_t price;
    std::uint32_t qty;
    std::uint8_t  side;        // 0 = Bid, 1 = Ask
    std::uint8_t  event_type;  // 1 = Add, 2 = Cancel, 3 = Executed/Ack
    std::uint16_t flags;
    std::uint32_t client_id;
    std::uint64_t latency_tsc;
    std::uint8_t  _reserved[16];
};

static_assert(sizeof(ShmIpcEvent) == 64, "ShmIpcEvent must be exactly 64 bytes");

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

inline constexpr std::size_t kTotalShmSize = sizeof(ShmIpcHeader) + 2 * sizeof(ShmRingChannel);

// Zero-allocation SPSC Producer for Matching Engine Telemetry / IPC
class ShmSpscProducer {
public:
    ShmSpscProducer() noexcept : channel_(nullptr), cached_head_(0), seq_(0) {}

    explicit ShmSpscProducer(ShmRingChannel* channel) noexcept
        : channel_(channel), cached_head_(0), seq_(0) {}

    void bind_channel(ShmRingChannel* channel) noexcept {
        channel_ = channel;
        cached_head_ = 0;
        seq_ = 0;
    }

    [[nodiscard]] bool is_bound() const noexcept {
        return channel_ != nullptr;
    }

    // Direct non-blocking enqueue into pre-allocated slot.
    // Zero dynamic memory allocation on the hot path.
    [[gnu::hot]] bool try_enqueue(const ShmIpcEvent& ev) noexcept {
        if (!channel_) [[unlikely]] return false;

        const std::uint64_t current_tail = channel_->tail.load(std::memory_order_relaxed);
        if (current_tail - cached_head_ >= kRingCapacity) {
            cached_head_ = channel_->head.load(std::memory_order_acquire);
            if (current_tail - cached_head_ >= kRingCapacity) {
                return false; // Ring buffer full
            }
        }

        const std::size_t idx = current_tail & kRingMask;
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

    std::uint64_t current_seq() const noexcept { return seq_; }

private:
    ShmRingChannel* channel_;
    std::uint64_t cached_head_;
    std::uint64_t seq_;
};

// Zero-allocation SPSC Consumer for Replay / Bridge Verification
class ShmSpscConsumer {
public:
    ShmSpscConsumer() noexcept : channel_(nullptr), cached_tail_(0) {}

    explicit ShmSpscConsumer(ShmRingChannel* channel) noexcept
        : channel_(channel), cached_tail_(0) {}

    void bind_channel(ShmRingChannel* channel) noexcept {
        channel_ = channel;
        cached_tail_ = 0;
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

        const std::size_t idx = current_head & kRingMask;
        std::memcpy(&out, &channel_->slots[idx], sizeof(ShmIpcEvent));
        channel_->head.store(current_head + 1, std::memory_order_release);
        return true;
    }

private:
    ShmRingChannel* channel_;
    std::uint64_t cached_tail_;
};

// POSIX Shared Memory Endpoint Manager
class ShmSpscEndpoint {
public:
    ShmSpscEndpoint() noexcept : ptr_(nullptr), len_(0), fd_(-1), is_creator_(false) {}

    ~ShmSpscEndpoint() {
        close();
    }

    bool open(const char* path = kDefaultShmPath, bool create = true) noexcept {
        close();

        int flags = O_RDWR;
        if (create) {
            flags |= O_CREAT;
        }

        fd_ = ::open(path, flags, 0666);
        if (fd_ < 0) {
            return fallback_anonymous();
        }

        if (create) {
            if (::ftruncate(fd_, static_cast<off_t>(kTotalShmSize)) != 0) {
                ::close(fd_);
                fd_ = -1;
                return fallback_anonymous();
            }
            is_creator_ = true;
        }

        void* m = ::mmap(nullptr, kTotalShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (m == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            return fallback_anonymous();
        }

        ptr_ = static_cast<std::uint8_t*>(m);
        len_ = kTotalShmSize;

        if (create) {
            init_header();
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

    [[nodiscard]] ShmRingChannel* request_channel() noexcept {
        return reinterpret_cast<ShmRingChannel*>(ptr_ + sizeof(ShmIpcHeader));
    }

    [[nodiscard]] ShmRingChannel* response_channel() noexcept {
        return reinterpret_cast<ShmRingChannel*>(ptr_ + sizeof(ShmIpcHeader) + sizeof(ShmRingChannel));
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return ptr_ != nullptr && ptr_ != MAP_FAILED;
    }

private:
    bool fallback_anonymous() noexcept {
        void* m = ::mmap(nullptr, kTotalShmSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            return false;
        }
        ptr_ = static_cast<std::uint8_t*>(m);
        len_ = kTotalShmSize;
        init_header();
        return true;
    }

    void init_header() noexcept {
        std::memset(ptr_, 0, len_);
        auto* hdr = header();
        hdr->magic = kShmMagic;
        hdr->version = kShmVersion;
        hdr->status = 1;
        hdr->capacity = static_cast<std::uint32_t>(kRingCapacity);
        hdr->req_offset = sizeof(ShmIpcHeader);
        hdr->resp_offset = sizeof(ShmIpcHeader) + sizeof(ShmRingChannel);
    }

    std::uint8_t* ptr_;
    std::size_t len_;
    int fd_;
    bool is_creator_;
};

} // namespace experimental

using experimental::kDefaultShmPath;
using experimental::kRingCapacity;
using experimental::kRingMask;
using experimental::kShmMagic;
using experimental::kShmVersion;
using experimental::kTotalShmSize;
using experimental::ShmIpcEvent;
using experimental::ShmIpcHeader;
using experimental::ShmRingChannel;
using experimental::ShmSpscConsumer;
using experimental::ShmSpscEndpoint;
using experimental::ShmSpscProducer;

} // namespace sim
