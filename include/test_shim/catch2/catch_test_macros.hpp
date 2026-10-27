// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "catch_approx.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace test_shim {

struct TestFailureException : public std::exception {
    const char* what() const noexcept override {
        return "Test assertion failure";
    }
};

struct SectionTracker {
    std::vector<std::string> active_path;
    std::vector<std::string> target_path;
    bool in_run = false;
    bool section_executed = false;
    std::vector<std::vector<std::string>> discovered_paths;

    void reset_for_test() {
        active_path.clear();
        target_path.clear();
        discovered_paths.clear();
        in_run = true;
    }

    bool enter_section(const std::string& name) {
        active_path.push_back(name);
        if (target_path.empty()) {
            return true;
        }
        if (active_path.size() <= target_path.size()) {
            for (std::size_t i = 0; i < active_path.size(); ++i) {
                if (active_path[i] != target_path[i]) {
                    active_path.pop_back();
                    return false;
                }
            }
            return true;
        }
        discovered_paths.push_back(active_path);
        active_path.pop_back();
        return false;
    }

    void leave_section() {
        if (!active_path.empty()) {
            active_path.pop_back();
        }
    }
};

inline SectionTracker& get_section_tracker() {
    static SectionTracker tracker;
    return tracker;
}

class SectionGuard {
public:
    explicit SectionGuard(const std::string& name) : entered_(false) {
        entered_ = get_section_tracker().enter_section(name);
    }
    ~SectionGuard() {
        if (entered_) {
            get_section_tracker().leave_section();
        }
    }
    explicit operator bool() const noexcept { return entered_; }
private:
    bool entered_;
};

struct TestStats {
    std::size_t total_assertions = 0;
    std::size_t failed_assertions = 0;
    std::size_t total_tests = 0;
    std::size_t passed_tests = 0;
    std::size_t failed_tests = 0;
    std::string current_test_name;
};

inline TestStats& get_stats() {
    static TestStats stats;
    return stats;
}

struct TestCase {
    std::string name;
    std::string tags;
    std::function<void()> fn;
};

inline std::vector<TestCase>& get_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

struct AutoReg {
    AutoReg(const char* name, const char* tags, void(*func)()) {
        get_registry().push_back(TestCase{name, tags, func});
    }
};

inline void log_failure(const char* file, int line, const char* expr, const std::string& msg = "") {
    auto& s = get_stats();
    ++s.failed_assertions;
    std::cerr << "FAILED: " << file << ":" << line << "\n"
              << "  Test: " << s.current_test_name << "\n"
              << "  Expression: " << expr;
    if (!msg.empty()) {
        std::cerr << " (" << msg << ")";
    }
    std::cerr << "\n";
}

inline void log_success() {
    auto& s = get_stats();
    ++s.total_assertions;
}

inline int run_all_tests() {
    auto& registry = get_registry();
    auto& stats = get_stats();
    stats.total_tests = registry.size();
    stats.passed_tests = 0;
    stats.failed_tests = 0;
    stats.total_assertions = 0;
    stats.failed_assertions = 0;

    std::cout << "===================================================\n";
    std::cout << "Running " << registry.size() << " test cases...\n";
    std::cout << "===================================================\n";

    for (const auto& tc : registry) {
        stats.current_test_name = tc.name;
        bool test_passed = true;
        auto& tracker = get_section_tracker();
        tracker.reset_for_test();

        try {
            tc.fn();
            while (!tracker.discovered_paths.empty()) {
                tracker.target_path = tracker.discovered_paths.back();
                tracker.discovered_paths.pop_back();
                tracker.active_path.clear();
                tc.fn();
            }
        } catch (const TestFailureException&) {
            test_passed = false;
        } catch (const std::exception& e) {
            std::cerr << "UNCAUGHT EXCEPTION in " << tc.name << ": " << e.what() << "\n";
            test_passed = false;
            ++stats.failed_assertions;
        } catch (...) {
            std::cerr << "UNKNOWN UNCAUGHT EXCEPTION in " << tc.name << "\n";
            test_passed = false;
            ++stats.failed_assertions;
        }

        if (test_passed) {
            ++stats.passed_tests;
        } else {
            ++stats.failed_tests;
        }
    }

    std::cout << "===================================================\n";
    std::cout << "Test Summary:\n";
    std::cout << "  Passed: " << stats.passed_tests << "/" << stats.total_tests << "\n";
    std::cout << "  Failed: " << stats.failed_tests << "/" << stats.total_tests << "\n";
    std::cout << "  Assertions: " << stats.total_assertions << " total, "
              << stats.failed_assertions << " failed\n";
    std::cout << "===================================================\n";

    return (stats.failed_tests == 0 && stats.failed_assertions == 0) ? 0 : 1;
}

} // namespace test_shim

#define TEST_SHIM_CONCAT_IMPL(a, b) a##b
#define TEST_SHIM_CONCAT(a, b) TEST_SHIM_CONCAT_IMPL(a, b)

#define TEST_CASE(name, tags) \
    static void TEST_SHIM_CONCAT(test_func_, __LINE__)(); \
    static const ::test_shim::AutoReg TEST_SHIM_CONCAT(autoreg_, __LINE__)(name, tags, &TEST_SHIM_CONCAT(test_func_, __LINE__)); \
    static void TEST_SHIM_CONCAT(test_func_, __LINE__)()

#define SECTION(name) \
    if (::test_shim::SectionGuard TEST_SHIM_CONCAT(section_guard_, __LINE__){name})

#define REQUIRE(expr) \
    do { \
        if (expr) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr); \
            throw ::test_shim::TestFailureException(); \
        } \
    } while (false)

#define CHECK(expr) \
    do { \
        if (expr) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr); \
        } \
    } while (false)

#define REQUIRE_FALSE(expr) \
    do { \
        if (!(expr)) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, "!(" #expr ")"); \
            throw ::test_shim::TestFailureException(); \
        } \
    } while (false)

#define CHECK_FALSE(expr) \
    do { \
        if (!(expr)) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, "!(" #expr ")"); \
        } \
    } while (false)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { \
            (void)(expr); \
        } catch (...) { \
            threw = true; \
        } \
        if (threw) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "expected exception was not thrown"); \
            throw ::test_shim::TestFailureException(); \
        } \
    } while (false)

#define CHECK_THROWS(expr) \
    do { \
        bool threw = false; \
        try { \
            (void)(expr); \
        } catch (...) { \
            threw = true; \
        } \
        if (threw) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "expected exception was not thrown"); \
        } \
    } while (false)

#define REQUIRE_THROWS_AS(expr, exc_type) \
    do { \
        bool threw_as = false; \
        try { \
            (void)(expr); \
        } catch (const exc_type&) { \
            threw_as = true; \
        } catch (...) { \
            threw_as = false; \
        } \
        if (threw_as) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "expected exception " #exc_type " was not thrown"); \
            throw ::test_shim::TestFailureException(); \
        } \
    } while (false)

#define CHECK_THROWS_AS(expr, exc_type) \
    do { \
        bool threw_as = false; \
        try { \
            (void)(expr); \
        } catch (const exc_type&) { \
            threw_as = true; \
        } catch (...) { \
            threw_as = false; \
        } \
        if (threw_as) { \
            ::test_shim::log_success(); \
        } else { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "expected exception " #exc_type " was not thrown"); \
        } \
    } while (false)

#define REQUIRE_NOTHROW(expr) \
    do { \
        try { \
            (void)(expr); \
            ::test_shim::log_success(); \
        } catch (const std::exception& e) { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, std::string("unexpected exception: ") + e.what()); \
            throw ::test_shim::TestFailureException(); \
        } catch (...) { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "unexpected unknown exception"); \
            throw ::test_shim::TestFailureException(); \
        } \
    } while (false)

#define CHECK_NOTHROW(expr) \
    do { \
        try { \
            (void)(expr); \
            ::test_shim::log_success(); \
        } catch (const std::exception& e) { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, std::string("unexpected exception: ") + e.what()); \
        } catch (...) { \
            ::test_shim::log_failure(__FILE__, __LINE__, #expr, "unexpected unknown exception"); \
        } \
    } while (false)

#define INFO(msg) do {} while(false)
#define CAPTURE(var) do {} while(false)

#ifndef CATCH_CONFIG_RUNNER
__attribute__((weak)) int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return ::test_shim::run_all_tests();
}
#endif
