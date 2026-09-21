CXX ?= g++
CXXSTD ?= -std=c++20
WARN       ?= -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
OPT      ?= -O2 -DNDEBUG
INCLUDES ?= -Iinclude
PYTHON     ?= python3

# --- Opt-in diagnostic / release build modes (default behavior unchanged) -----
# These NEVER affect the matching kernel semantics; they only change build flags.
#   make test MODE=debug        # -O0 + assertions + frame pointers
#   make test MODE=asan         # AddressSanitizer + UBSan, halt on first error
#   make test MODE=ubsan        # UBSan only, halt on first error
#   make test MODE=tsan         # ThreadSanitizer
#   make bench MODE=release-lto # -O3 + LTO + -march=native (benchmark evidence)
#   make test WERROR=1          # promote compile warnings to errors
# Convenience targets: `make asan|ubsan|tsan|debug` build+run tests in that mode.
SANITIZE_FLAGS ?=
ifeq ($(MODE),debug)
  OPT := -O0 -g3 -fno-omit-frame-pointer
else ifeq ($(MODE),asan)
  OPT := -O1 -g -fno-omit-frame-pointer
  SANITIZE_FLAGS := -fsanitize=address,undefined -fno-sanitize-recover=all
else ifeq ($(MODE),ubsan)
  OPT := -O1 -g -fno-omit-frame-pointer
  SANITIZE_FLAGS := -fsanitize=undefined -fno-sanitize-recover=all
else ifeq ($(MODE),tsan)
  OPT := -O1 -g
  SANITIZE_FLAGS := -fsanitize=thread
else ifeq ($(MODE),release-lto)
  OPT := -O3 -DNDEBUG -flto -march=native
endif

ifeq ($(WERROR),1)
  WARN += -Werror
endif
VCPKG      ?= $(CURDIR)/.vcpkg/vcpkg
VCPKG_ROOT ?= $(CURDIR)/.vcpkg

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    VCPKG_TRIPLET ?= arm64-osx
  else
    VCPKG_TRIPLET ?= x64-osx
  endif
else ifeq ($(UNAME_S),Linux)
  VCPKG_TRIPLET ?= x64-linux
else
  VCPKG_TRIPLET ?= x64-osx
endif

VCPKG_INSTALLED_DIR ?= $(CURDIR)/vcpkg_installed
VCPKG_INCLUDEDIR = $(VCPKG_INSTALLED_DIR)/$(VCPKG_TRIPLET)/include
VCPKG_LIBDIR = $(VCPKG_INSTALLED_DIR)/$(VCPKG_TRIPLET)/lib

SRC := $(wildcard src/*.cpp)
TEST_SRC := $(wildcard tests/*.cpp)
BENCH_SRC := $(wildcard bench/*.cpp)
EXAMPLE_SRC := $(wildcard examples/*.cpp)

BUILD_DIR := build
RESULTS_DIR := results

BIN_RUN     := $(BUILD_DIR)/sim
BIN_TEST     := $(BUILD_DIR)/sim_test
BIN_BENCH    := $(BUILD_DIR)/sim_bench
BIN_EXAMPLE  := $(BUILD_DIR)/sim_example

.PHONY: all help build build-bin test test-bin bench bench-bin bench-save bench-shards example example-bin lint lint-fix run clean dirs deps deps-clean asan ubsan tsan debug release-lto

CPPFLAGS += $(INCLUDES)
CXXFLAGS += $(CXXSTD) $(WARN) $(OPT) $(SANITIZE_FLAGS)
LDFLAGS  += $(SANITIZE_FLAGS)

ifneq ($(wildcard $(VCPKG_INCLUDEDIR)),)
CPPFLAGS += -isystem $(VCPKG_INCLUDEDIR)
LDFLAGS  += -L$(VCPKG_LIBDIR)
endif

TEST_LDLIBS := -L$(VCPKG_LIBDIR)/manual-link -lCatch2Main -lCatch2
BENCH_LDLIBS := -lbenchmark
EXAMPLE_LDLIBS :=

ifeq ($(UNAME_S),Linux)
BENCH_LDLIBS += -lpthread
endif

all: build

COMPILE_BASE := $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

define BUILD_IF_SOURCES
	@if [ -z "$($(1))" ]; then echo "$(2)"; \
	else $(3); fi
endef

help:
	@echo "Available targets:"
	@echo "  help       - Show this help"
	@echo "  build      - Build main binary (build/sim)"
	@echo "  test       - Build and run tests (build/sim_test)"
	@echo "  test-bin   - Build test binary only (build/sim_test)"
	@echo "  bench      - Build and run benchmarks (build/sim_bench)"
	@echo "  bench-bin  - Build benchmark binary only (build/sim_bench)"
	@echo "  bench-save - Run benchmarks and save timestamped JSON to results/"
	@echo "  example    - Build and run examples (build/sim_example)"
	@echo "  lint       - Run clang-tidy checks (HFT-focused profile)"
	@echo "  lint-fix   - Run clang-tidy with automatic fixes"
	@echo "  asan       - Build and run tests with Address+UB sanitizer"
	@echo "  ubsan      - Build and run tests with UB sanitizer"
	@echo "  tsan       - Build and run tests with Thread sanitizer"
	@echo "  debug      - Build and run tests with -O0 + debug info"
	@echo "  release-lto- Build benchmark binary with -O3 + LTO + native"
	@echo "  run        - Run main binary (build/sim)"
	@echo "  clean      - Remove build artifacts"
	@echo "  deps       - Install dependencies via vcpkg"
	@echo "  deps-clean - Remove vcpkg_installed artifacts"
	@echo ""
	@echo "Build-mode knobs (default build is unchanged):"
	@echo "  MODE=debug|asan|ubsan|tsan|release-lto   e.g. make test MODE=asan"
	@echo "  WERROR=1                                 promote compile warnings to errors"

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS_DIR)

build: build-bin

build-bin: dirs
	$(call BUILD_IF_SOURCES,SRC,No src/*.cpp yet — add sources when the implementation lands.,$(COMPILE_BASE) $(SRC) -o $(BIN_RUN))

test: test-bin
	@./$(BIN_TEST)

test-bin: dirs
	$(call BUILD_IF_SOURCES,TEST_SRC,No tests/*.cpp yet — add sources when the test suite lands.,$(COMPILE_BASE) $(TEST_SRC) $(SRC) $(TEST_LDLIBS) -o $(BIN_TEST))

bench: bench-bin
	@./$(BIN_BENCH)

bench-bin: dirs
	$(call BUILD_IF_SOURCES,BENCH_SRC,No bench/*.cpp yet — add sources when the benchmark suite lands.,$(COMPILE_BASE) $(BENCH_SRC) $(SRC) $(BENCH_LDLIBS) -o $(BIN_BENCH))

# Run the suite and persist a timestamped JSON + console log + env metadata to
# results/. Forward extra args via BENCH_ARGS="--benchmark_filter=BM_FlatLadder.*".
bench-save: dirs
	@chmod +x scripts/run_bench.sh
	@./scripts/run_bench.sh $(BENCH_ARGS)

# CC-004: run N independent benchmark processes (private books, no shared
# state) to characterise host resource contention. NOT a scaling study.
# Control shard count with BENCH_SHARDS=N; forward bench args via BENCH_ARGS.
bench-shards: dirs
	@chmod +x scripts/run_bench_shards.sh
	@./scripts/run_bench_shards.sh $(BENCH_ARGS)

example: example-bin
	@./$(BIN_EXAMPLE)

example-bin: dirs
	$(call BUILD_IF_SOURCES,EXAMPLE_SRC,No examples/*.cpp yet — add sources when examples land.,$(COMPILE_BASE) $(EXAMPLE_SRC) $(SRC) $(EXAMPLE_LDLIBS) -o $(BIN_EXAMPLE))

lint:
	@chmod +x scripts/run_clang_tidy.sh
	@./scripts/run_clang_tidy.sh check

lint-fix:
	@chmod +x scripts/run_clang_tidy.sh
	@./scripts/run_clang_tidy.sh fix

# Opt-in diagnostic build modes. Each rebuilds the test suite under the named
# sanitizer/mode and runs it. The default `make test` is unaffected.
asan ubsan tsan debug:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory test MODE=$@

# Release-with-LTO build of the benchmark binary for performance evidence runs.
release-lto:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory bench-bin MODE=release-lto
	@echo "Built $(BIN_BENCH) with MODE=release-lto (-O3 -flto -march=native)."

run: $(BIN_RUN)
	./$(BIN_RUN)

# NOTE: results/ is intentionally preserved across `make clean` so historical
# bench runs survive for trend analysis. Use `rm -rf results/bench-*` manually
# to prune old runs.
clean:
	rm -rf $(BUILD_DIR)

deps:
	@test -x $(VCPKG) || { echo "vcpkg binary not found at $(VCPKG); run: git clone --depth 1 https://github.com/microsoft/vcpkg .vcpkg && .vcpkg/bootstrap-vcpkg.sh -disableMetrics"; exit 1; }
	VCPKG_ROOT=$(VCPKG_ROOT) $(VCPKG) install --triplet $(VCPKG_TRIPLET)

deps-clean:
	rm -rf $(VCPKG_INSTALLED_DIR)
