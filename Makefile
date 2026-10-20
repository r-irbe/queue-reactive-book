CXX ?= g++
CXXSTD ?= -std=c++20
WARN       ?= -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
OPT      ?= -O2 -DNDEBUG
INCLUDES ?= -Iinclude -Iinclude/test_shim
QRLOB_BUILD_EXPERIMENTAL ?= ON
TEST_EXP_FLAGS :=
ifeq ($(QRLOB_BUILD_EXPERIMENTAL),ON)
  TEST_EXP_FLAGS := -Iexperimental -DQRLOB_BUILD_EXPERIMENTAL=1
else ifeq ($(QRLOB_BUILD_EXPERIMENTAL),1)
  TEST_EXP_FLAGS := -Iexperimental -DQRLOB_BUILD_EXPERIMENTAL=1
else
  TEST_EXP_FLAGS := -DQRLOB_BUILD_EXPERIMENTAL=0
endif

# Opt-in diagnostic and release build modes:
#   make test MODE=debug        # -O0 + assertions + frame pointers
#   make test MODE=asan         # AddressSanitizer + UBSan, halt on first error
#   make test MODE=ubsan        # UBSan only, halt on first error
#   make test MODE=tsan         # ThreadSanitizer
#   make bench MODE=release-lto # -O3 + LTO + -march=native
#   make test WERROR=1          # promote compile warnings to errors
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
  OPT := -O3 -DNDEBUG -flto=auto -march=native
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

LIB_SRC := src/flat_ladder_book.cpp src/lob.cpp
MAIN_SRC := src/main.cpp
SRC := $(LIB_SRC) $(MAIN_SRC)
TEST_SRC := $(wildcard tests/*.cpp)
BENCH_SRC := $(wildcard bench/*.cpp)
EXAMPLE_SRC := $(wildcard examples/*.cpp)

BUILD_DIR := build

BIN_RUN      := $(BUILD_DIR)/sim
BIN_TEST     := $(BUILD_DIR)/sim_test
BIN_BENCH    := $(BUILD_DIR)/sim_bench
BIN_EXAMPLE  := $(BUILD_DIR)/sim_example

.PHONY: all help build build-bin test test_core test-bin bench bench-bin example example-bin run clean dirs deps deps-clean asan ubsan tsan debug release-lto

CPPFLAGS += $(INCLUDES)
CXXFLAGS += $(CXXSTD) $(WARN) $(OPT) $(SANITIZE_FLAGS)
LDFLAGS  += $(SANITIZE_FLAGS)

ifneq ($(wildcard $(VCPKG_INCLUDEDIR)),)
CPPFLAGS += -isystem $(VCPKG_INCLUDEDIR)
LDFLAGS  += -L$(VCPKG_LIBDIR)
endif

TEST_LDLIBS :=
ifneq ($(wildcard $(VCPKG_LIBDIR)/manual-link/libCatch2Main.a),)
TEST_LDLIBS := -L$(VCPKG_LIBDIR)/manual-link -lCatch2Main -lCatch2
endif

BENCH_LDLIBS :=
ifneq ($(wildcard $(VCPKG_LIBDIR)/libbenchmark.a),)
BENCH_LDLIBS := -lbenchmark
endif
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
	@echo "  help        - Show this help"
	@echo "  build       - Build main simulation binary (build/sim)"
	@echo "  test        - Build and run full test suite (build/sim_test)"
	@echo "  test_core   - Build and run core-only tests (QRLOB_BUILD_EXPERIMENTAL=OFF)"
	@echo "  test-bin    - Build test binary only (build/sim_test)"
	@echo "  bench       - Build and run benchmarks (build/sim_bench)"
	@echo "  bench-bin   - Build benchmark binary only (build/sim_bench)"
	@echo "  example     - Build and run examples (build/sim_example)"
	@echo "  asan        - Build and run tests with Address+UB sanitizer"
	@echo "  ubsan       - Build and run tests with UB sanitizer"
	@echo "  tsan        - Build and run tests with Thread sanitizer"
	@echo "  debug       - Build and run tests with -O0 + debug info"
	@echo "  release-lto - Build benchmark binary with -O3 + LTO + native"
	@echo "  run         - Run main binary (build/sim)"
	@echo "  clean       - Remove build artifacts"
	@echo "  deps        - Install optional vcpkg packages"
	@echo "  deps-clean  - Remove vcpkg_installed artifacts"
	@echo ""
	@echo "Build-mode knobs:"
	@echo "  MODE=debug|asan|ubsan|tsan|release-lto   e.g. make test MODE=asan"
	@echo "  QRLOB_BUILD_EXPERIMENTAL=ON|OFF          toggle experimental test surface"
	@echo "  WERROR=1                                 promote compile warnings to errors"

dirs:
	@mkdir -p $(BUILD_DIR)

build: build-bin

build-bin: dirs
	$(call BUILD_IF_SOURCES,LIB_SRC,No src/*.cpp yet.,$(COMPILE_BASE) $(MAIN_SRC) $(LIB_SRC) -o $(BIN_RUN))

test: test-bin
	@./$(BIN_TEST)

test_core:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory test QRLOB_BUILD_EXPERIMENTAL=OFF

test-bin: dirs
	$(call BUILD_IF_SOURCES,TEST_SRC,No tests/*.cpp yet.,$(COMPILE_BASE) $(TEST_EXP_FLAGS) $(TEST_SRC) $(LIB_SRC) $(TEST_LDLIBS) -o $(BIN_TEST))

bench: bench-bin
	@./$(BIN_BENCH)

bench-bin: dirs
	$(call BUILD_IF_SOURCES,BENCH_SRC,No bench/*.cpp yet.,$(COMPILE_BASE) $(BENCH_SRC) $(LIB_SRC) $(BENCH_LDLIBS) -o $(BIN_BENCH))

example: example-bin
	@./$(BIN_EXAMPLE)

example-bin: dirs
	$(call BUILD_IF_SOURCES,EXAMPLE_SRC,No examples/*.cpp yet.,$(COMPILE_BASE) $(EXAMPLE_SRC) $(LIB_SRC) $(EXAMPLE_LDLIBS) -o $(BIN_EXAMPLE))

asan ubsan tsan debug:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory test MODE=$@

release-lto:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory bench-bin MODE=release-lto
	@echo "Built $(BIN_BENCH) with MODE=release-lto (-O3 -flto=auto -march=native)."

run: $(BIN_RUN)
	./$(BIN_RUN)

clean:
	rm -rf $(BUILD_DIR)

deps:
	@test -x $(VCPKG) || { echo "vcpkg binary not found at $(VCPKG); run: git clone --depth 1 https://github.com/microsoft/vcpkg .vcpkg && .vcpkg/bootstrap-vcpkg.sh -disableMetrics"; exit 1; }
	VCPKG_ROOT=$(VCPKG_ROOT) $(VCPKG) install --triplet $(VCPKG_TRIPLET)

deps-clean:
	rm -rf $(VCPKG_INSTALLED_DIR)
