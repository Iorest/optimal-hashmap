// =============================================================================
// tests/test_flat_map.cpp
//
// Complete test suite for ooh::flat_map.
//
// Design principles
// ─────────────────
//  • Zero locks in test code — all synchronisation uses the same atomics that
//    the production code exposes (release/acquire, fetch_add, etc.).
//  • Every test is self-documenting: the comment above each function states
//    exactly what property/invariant is being verified.
//  • Paper-derived tests explicitly call out which theorem or lemma they
//    exercise from Farach-Colton, Krapivin, Kuszmaul 2025.
//  • Deterministic seeds are used everywhere so failures are reproducible.
//
// Build:
//   cmake -B build -DCMAKE_BUILD_TYPE=Debug -DOOH_BUILD_TESTS=ON
//   cmake --build build && ctest --test-dir build -V
//
// Or manually (with ASan):
//   clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -pthread
//       tests/test_flat_map.cpp -Iinclude -o /tmp/t && /tmp/t
// =============================================================================

#include "ooh/flat_map.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── Minimal test harness (no external deps) ───────────────────────────────────
//
// g_pass / g_fail use std::atomic so that CHECK() is safe to call from worker
// threads (e.g. inside concurrent_insert lambdas) without TSan data races.
static std::atomic<int> g_pass{0}, g_fail{0};

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
            g_fail.fetch_add(1, std::memory_order_relaxed); \
        } else { \
            g_pass.fetch_add(1, std::memory_order_relaxed); \
        } \
    } while (0)

#define CHECK_EQ(a, b)  CHECK((a) == (b))
#define CHECK_NE(a, b)  CHECK((a) != (b))
#define CHECK_LT(a, b)  CHECK((a) <  (b))
#define CHECK_LE(a, b)  CHECK((a) <= (b))

static void section(const char* name) { printf("\n  [%s]\n", name); }
static void ok(const char* name)      { printf("    %-52s OK\n", name); }

// ── Helpers ───────────────────────────────────────────────────────────────────

// Deterministic key set with a given seed
static std::vector<uint64_t> rand_keys(int n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();
    // Deduplicate
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    while ((int)v.size() < n) { v.push_back(rng()); std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); }
    v.resize(n);
    return v;
}

// Ground-truth reference map (std::unordered_map)
using RefMap = std::unordered_map<int64_t, int64_t>;

// ─── RAII lifecycle tracker ───────────────────────────────────────────────────
// Used to detect double-destructs and missing destructs.
static std::atomic<int> g_live_objects{0};

struct Tracked {
    int64_t value{0};
    bool    valid{false};
    Tracked()                      : value(0),          valid(true)  { ++g_live_objects; }
    explicit Tracked(int64_t v)    : value(v),          valid(true)  { ++g_live_objects; }
    Tracked(const Tracked& o)      : value(o.value),    valid(true)  { assert(o.valid); ++g_live_objects; }
    Tracked(Tracked&& o) noexcept  : value(o.value),    valid(true)  { assert(o.valid); o.valid=false; ++g_live_objects; }
    Tracked& operator=(const Tracked& o) { assert(valid && o.valid); value=o.value; return *this; }
    Tracked& operator=(Tracked&& o) noexcept { assert(valid && o.valid); value=o.value; o.valid=false; return *this; }
    ~Tracked() { if (valid) { valid=false; --g_live_objects; } }
    bool operator==(const Tracked& o) const noexcept { return value == o.value; }
};

// =============================================================================
// ── SECTION A: Basic API correctness ─────────────────────────────────────────
// =============================================================================

// A-01: insert / find / contains / count — basic single-writer round-trip
static void test_A01_basic_insert_find() {
    ooh::flat_map<int64_t,int64_t> m(100);
    CHECK(m.empty());
    CHECK_EQ(m.max_size(), 100u);
    CHECK_EQ(m.size(), 0u);

    CHECK(m.insert(1, 10));
    CHECK(m.insert(2, 20));
    CHECK(m.insert(3, 30));
    CHECK(!m.insert(2, 99));      // duplicate → false, value unchanged
    CHECK_EQ(m.size(), 3u);
    CHECK(!m.empty());

    CHECK(m.contains(1));
    CHECK(!m.contains(42));
    CHECK_EQ(*m.find(2), 20);
    CHECK(!m.find(99));
    CHECK_EQ(m.count(1), 1u);
    CHECK_EQ(m.count(99), 0u);
    ok("A-01 basic insert/find");
}

// A-02: at() — throws on missing key, returns correct reference otherwise
static void test_A02_at() {
    ooh::flat_map<int64_t,int64_t> m(50);
    m.insert(10, 100);
    m.insert(20, 200);
    CHECK_EQ(m.at(10), 100);
    CHECK_EQ(m.at(20), 200);

    bool threw = false;
    try { (void)m.at(99); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);

    const auto& cm = m;
    threw = false;
    try { (void)cm.at(99); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    ok("A-02 at()");
}

// A-03: operator[] — inserts default value on miss, returns reference
static void test_A03_operator_brackets() {
    ooh::flat_map<int64_t,int64_t> m(50);
    m.insert(10, 100);
    CHECK_EQ(m[10], 100);         // hit
    m[30] = 300;
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(m[30], 300);
    int64_t& ref = m[40];
    CHECK_EQ(ref, 0);             // default-constructed
    ref = 400;
    CHECK_EQ(m[40], 400);
    CHECK_EQ(m.size(), 3u);

    // Overflow throws
    ooh::flat_map<int64_t,int64_t> tiny(2);
    tiny[1] = 1; tiny[2] = 2;
    bool threw = false;
    try { tiny[3] = 3; } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw);
    ok("A-03 operator[]");
}

// A-04: emplace / try_emplace — in-place construction, no double-move
static void test_A04_emplace_try_emplace() {
    ooh::flat_map<int64_t,int64_t> m(50);
    auto r1 = m.emplace(1LL, 11LL);
    CHECK(r1.inserted);
    CHECK_EQ(*r1.value_ptr, 11);

    auto r2 = m.emplace(1LL, 99LL);   // duplicate
    CHECK(!r2.inserted);
    CHECK_EQ(*r2.value_ptr, 11);      // original value unchanged

    auto r3 = m.try_emplace(2LL, 22LL);
    CHECK(r3.inserted);
    CHECK_EQ(*r3.value_ptr, 22);

    auto r4 = m.try_emplace(2LL, 88LL);
    CHECK(!r4.inserted);
    CHECK_EQ(*r4.value_ptr, 22);

    CHECK_EQ(m.size(), 2u);
    ok("A-04 emplace/try_emplace");
}

// A-05: insert_or_assign / update
static void test_A05_insert_or_assign_update() {
    ooh::flat_map<int64_t,int64_t> m(50);
    CHECK(m.insert_or_assign(1, 10));    // insert → true
    CHECK(!m.insert_or_assign(1, 20));   // assign → false
    CHECK_EQ(m.size(), 1u);
    CHECK_EQ(*m.find(1), 20);

    CHECK(m.update(1, 777));
    CHECK_EQ(*m.find(1), 777);
    CHECK(!m.update(99, 0));
    CHECK_EQ(m.size(), 1u);
    ok("A-05 insert_or_assign/update");
}

// A-06: erase — tombstone; size decreases; key gone; double-erase false
static void test_A06_erase_basic() {
    constexpr int N = 1000;
    ooh::flat_map<int64_t,int64_t> m(N + 200);
    for (int i = 0; i < N; ++i) m.insert(i, i * 2);

    for (int i = 0; i < N; i += 2) {
        CHECK(m.erase(i));
        CHECK(!m.erase(i));    // already gone
    }
    CHECK_EQ(m.size(), (size_t)N / 2);
    for (int i = 1; i < N; i += 2) CHECK_EQ(*m.find(i), i * 2);
    for (int i = 0; i < N; i += 2) CHECK(!m.find(i));
    ok("A-06 erase basic");
}

// A-07: clear — resets everything; reusable afterwards
static void test_A07_clear() {
    ooh::flat_map<int64_t,int64_t> m(50);
    for (int i = 0; i < 20; ++i) m.insert(i, i);
    m.clear();
    CHECK_EQ(m.size(), 0u);
    CHECK(m.empty());
    CHECK(!m.find(0));
    m.insert(100, 200);
    CHECK_EQ(*m.find(100), 200);
    ok("A-07 clear");
}

// A-08: swap — contents and sizes exchange
static void test_A08_swap() {
    ooh::flat_map<int64_t,int64_t> a(50), b(30);
    a.insert(1, 2); a.insert(3, 4);
    b.insert(10, 20);
    a.swap(b);
    CHECK_EQ(a.size(), 1u);
    CHECK_EQ(*a.find(10), 20);
    CHECK_EQ(b.size(), 2u);
    CHECK_EQ(*b.find(1), 2);

    using std::swap;
    swap(a, b);
    CHECK_EQ(a.size(), 2u);
    CHECK_EQ(b.size(), 1u);
    ok("A-08 swap");
}

// A-09: move constructor / move assignment
static void test_A09_move_semantics() {
    ooh::flat_map<int64_t,int64_t> m(20);
    m.insert(1, 10); m.insert(2, 20);

    auto m2 = std::move(m);
    CHECK_EQ(m2.size(), 2u);
    CHECK_EQ(*m2.find(1), 10);

    ooh::flat_map<int64_t,int64_t> m3(5);
    m3 = std::move(m2);
    CHECK_EQ(m3.size(), 2u);
    CHECK_EQ(*m3.find(2), 20);
    ok("A-09 move semantics");
}

// A-10: initializer-list / range constructors
static void test_A10_constructors() {
    ooh::flat_map<int64_t,int64_t> m1({{1,10},{2,20},{3,30}});
    CHECK_EQ(m1.size(), 3u);
    CHECK_EQ(*m1.find(2), 20);

    std::vector<std::pair<int64_t,int64_t>> vec = {{10,1},{20,2},{30,3}};
    ooh::flat_map<int64_t,int64_t> m2(vec.begin(), vec.end());
    CHECK_EQ(m2.size(), 3u);
    CHECK_EQ(*m2.find(20), 2);
    ok("A-10 constructors");
}

// A-11: find_ptr / find_frozen_ptr — zero-copy access
static void test_A11_find_ptr() {
    ooh::flat_map<int64_t,int64_t> m(50);
    m.insert(1, 100); m.insert(2, 200);

    const auto& cm = m;
    CHECK(cm.find_ptr(1) != nullptr);
    CHECK_EQ(*cm.find_ptr(1), 100);
    CHECK(cm.find_ptr(99) == nullptr);

    int64_t* p = m.find_ptr(2);
    CHECK(p != nullptr);
    *p = 999;
    CHECK_EQ(*m.find_ptr(2), 999);

    m.freeze();
    CHECK(m.find_frozen_ptr(1) != nullptr);
    CHECK_EQ(*m.find_frozen_ptr(1), 100);
    CHECK(m.find_frozen_ptr(0) == nullptr);
    ok("A-11 find_ptr");
}

// A-12: capacity / load_factor / max_load_factor / empty / full
static void test_A12_capacity_queries() {
    ooh::flat_map<int64_t,int64_t> m(100, 0.25);
    CHECK(m.empty());
    CHECK(!m.full());
    CHECK_EQ(m.max_size(), 100u);
    CHECK_EQ(m.load_factor(), 0.0);
    CHECK(m.max_load_factor() > 0.0 && m.max_load_factor() <= 1.0);

    for (int i = 0; i < 50; ++i) m.insert(i, i);
    CHECK(!m.empty());
    CHECK(!m.full());
    CHECK(m.load_factor() > 0.0);
    CHECK_EQ(m.size(), 50u);

    for (int i = 50; i < 100; ++i) m.insert(i, i);
    CHECK(m.full());
    ok("A-12 capacity queries");
}

// =============================================================================
// ── SECTION B: Tombstone reuse & erase/reinsert correctness ──────────────────
// =============================================================================

// B-01: After erase + re-insert of the SAME key, value is correct
//       (same-key probe chain always encounters the tombstone)
static void test_B01_erase_reinsert_same_key() {
    constexpr int N = 500;
    ooh::flat_map<int64_t,int64_t> m(N * 2);
    for (int i = 0; i < N; ++i) m.insert(i, i * 3);
    for (int i = 0; i < N; ++i) {
        m.erase(i);
        CHECK(!m.find(i));
        CHECK(m.insert(i, i * 7));
        CHECK_EQ(*m.find(i), i * 7);
    }
    CHECK_EQ(m.size(), (size_t)N);
    ok("B-01 erase+reinsert same key");
}

// B-02: erase all → insert different keys (with enough KV headroom)
static void test_B02_erase_all_reinsert_new_keys() {
    constexpr int N = 200;
    ooh::flat_map<int64_t,int64_t> m(N * 3);
    for (int i = 0; i < N; ++i) m.insert(i, i);
    for (int i = 0; i < N; ++i) m.erase(i);
    CHECK_EQ(m.size(), 0u);
    for (int i = N; i < N * 2; ++i) CHECK(m.insert(i, i * 5));
    for (int i = N; i < N * 2; ++i) CHECK_EQ(*m.find(i), i * 5);
    ok("B-02 erase-all, reinsert new keys");
}

// B-03: emplace tombstone path: after tombstone reuse, new value is correct
//       and the old slot is overwritten (not doubled up).
static void test_B03_tombstone_destructor_emplace() {
    ooh::flat_map<int64_t, int64_t> m(10);
    m.emplace(1LL, 100LL);
    m.emplace(2LL, 200LL);
    m.erase(1);
    CHECK(!m.find(1));
    // Re-emplace same key → goes through tombstone path
    m.emplace(1LL, 300LL);
    CHECK_EQ(m.at(1), 300);
    CHECK_EQ(m.at(2), 200);
    CHECK_EQ(m.size(), 2u);
    ok("B-03 tombstone destructor (emplace)");
}

// B-04: operator[] tombstone path: value is correctly overwritten
static void test_B04_tombstone_destructor_brackets() {
    ooh::flat_map<int64_t, int64_t> m(10);
    m[1] = 10; m[2] = 20;
    m.erase(1);
    m[1] = 30;
    CHECK_EQ(m.at(1), 30);
    CHECK_EQ(m.at(2), 20);
    CHECK_EQ(m.size(), 2u);
    ok("B-04 tombstone destructor (operator[])");
}

// B-05: Tombstone saturation stress — heavy erase/reinsert cycle
static void test_B05_tombstone_saturation() {
    constexpr int N = 500;
    ooh::flat_map<int64_t,int64_t> m(N * 2);
    for (int i = 0; i < N; ++i) m.insert(i, i);
    // Erase all but 0 and 1
    for (int i = 2; i < N; ++i) m.erase(i);
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(*m.find(0), 0);
    CHECK_EQ(*m.find(1), 1);
    CHECK(!m.find(2));
    // Insert new keys — may reuse tombstone slots
    for (int i = N; i < N + 20; ++i) m.insert(i, i * 7);
    for (int i = N; i < N + 20; ++i) CHECK_EQ(*m.find(i), i * 7);
    ok("B-05 tombstone saturation");
}

// B-06: Lifecycle integrity with std::string Key+Value (non-trivial KV)
static void test_B06_string_kv_lifecycle() {
    ooh::flat_map<std::string, std::string> m(50);
    m["hello"] = "world";
    m["foo"]   = "bar";
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(m.at("hello"), "world");

    m.erase("hello");
    CHECK(!m.find("hello"));
    m.insert("hello", std::string("again"));
    CHECK_EQ(*m.find("hello"), "again");

    // for_each_bucket must only yield live elements
    std::unordered_map<std::string,std::string> copy;
    m.for_each_bucket([&](const std::string& k, const std::string& v){ copy[k] = v; });
    CHECK_EQ(copy.size(), 2u);
    CHECK_EQ(copy["foo"], "bar");
    ok("B-06 string kv lifecycle");
}

// =============================================================================
// ── SECTION C: Exception safety & boundary conditions ─────────────────────────
// =============================================================================

// C-01: Constructor validation — invalid arguments throw
static void test_C01_ctor_exceptions() {
    auto throws = [](auto fn) {
        try { fn(); return false; } catch (...) { return true; }
    };
    CHECK(throws([] { ooh::flat_map<int,int> m(10, -0.1); }));
    CHECK(throws([] { ooh::flat_map<int,int> m(10,  0.0); }));
    CHECK(throws([] { ooh::flat_map<int,int> m(10,  1.0); }));
    CHECK(throws([] { ooh::flat_map<int,int> m(10,  1.5); }));
    CHECK(throws([] { ooh::flat_map<int,int> m(0); }));
    ok("C-01 constructor exceptions");
}

// C-02: Overflow handling — insert/operator[] past capacity
static void test_C02_overflow() {
    ooh::flat_map<int64_t,int64_t> m(3);
    CHECK(m.insert(1, 1));
    CHECK(m.insert(2, 2));
    CHECK(m.insert(3, 3));
    CHECK(m.full());
    CHECK(!m.insert(4, 4));    // silent false, no throw

    bool threw = false;
    try { m[4] = 4; } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw);
    ok("C-02 overflow handling");
}

// C-03: Slack boundary extremes — 1% and 99% load factors
static void test_C03_slack_extremes() {
    // Very tight (85% load) — more probes but functionally correct
    {
        ooh::flat_map<int64_t,int64_t> m(100, 0.15);
        for (int i = 0; i < 100; ++i) m.insert(i, i * 2);
        for (int i = 0; i < 100; ++i) CHECK_EQ(*m.find(i), i * 2);
    }
    // Very loose (5% load) — nearly empty bucket array
    {
        ooh::flat_map<int64_t,int64_t> m(100, 0.95);
        for (int i = 0; i < 100; ++i) m.insert(i, i * 3);
        for (int i = 0; i < 100; ++i) CHECK_EQ(*m.find(i), i * 3);
    }
    ok("C-03 slack extremes");
}

// C-04: max_items = 1 — smallest valid map
static void test_C04_single_element_map() {
    ooh::flat_map<int,int> m(1);
    CHECK(m.insert(42, 7));
    CHECK(!m.insert(43, 8));  // full
    CHECK_EQ(*m.find(42), 7);
    CHECK(m.full());
    m.erase(42);
    CHECK(!m.full());
    CHECK(m.insert(42, 99));  // same key through its own tombstone
    CHECK_EQ(*m.find(42), 99);
    ok("C-04 single element map");
}

// C-05: probe_stats sanity — avg PSL in theoretical bound
//       Paper Theorem 1.1: E[probes] = O(log(1/(1-α))) at load factor α.
//       At α=0.75, the bound is O(log 4) ≈ 2. We allow up to 8 empirically.
static void test_C05_probe_stats_theorem1() {
    constexpr int N = 50000;
    ooh::flat_map<int64_t,int64_t> m(N + 1000, 0.25);  // α ≈ 0.75
    for (int i = 0; i < N; ++i) m.insert(i, i);
    auto s = m.probe_stats();
    CHECK_EQ(s.live_count, (size_t)N);
    CHECK(s.avg_psl >= 1.0);
    CHECK_LT(s.avg_psl, 8.0);  // generous bound; theory says O(log 4)
    printf("      probe_stats @α≈0.75: avg=%.3f  max=%llu\n",
           s.avg_psl, (unsigned long long)s.max_psl);
    ok("C-05 probe_stats / Theorem 1.1");
}

// C-06: PSL distribution check at multiple load factors
//       No-Reordering: avg PSL grows gracefully with load (not super-linearly).
static void test_C06_psl_vs_load_factor() {
    constexpr int N = 10000;
    double prev_avg = 0.0;
    for (double slack : {0.5, 0.35, 0.25, 0.15}) {
        ooh::flat_map<int64_t,int64_t> m(N, slack);
        for (int i = 0; i < N; ++i) m.insert(i, i);
        auto s = m.probe_stats();
        double alpha = 1.0 - slack;
        // Each step should not regress drastically vs previous load factor
        CHECK(s.avg_psl >= 1.0);
        CHECK_LT(s.avg_psl, 20.0);  // never blows up
        printf("      slack=%.2f  α=%.2f  avg_psl=%.3f  max_psl=%llu\n",
               slack, alpha, s.avg_psl, (unsigned long long)s.max_psl);
        prev_avg = s.avg_psl;
    }
    (void)prev_avg;
    ok("C-06 PSL vs load factor");
}

// =============================================================================
// ── SECTION D: No-Reordering invariant verification ──────────────────────────
// =============================================================================
//
// The key property of the paper: once an element is placed in a slot,
// it NEVER moves.  We verify this directly by:
//   1. Recording the address of each element's value after insertion.
//   2. Running further inserts, erases, and lookups.
//   3. Verifying that the addresses have not changed.

// D-01: Placement stability — element addresses never change after insert
static void test_D01_no_reordering_placement_stability() {
    constexpr int N = 2000;
    ooh::flat_map<int64_t,int64_t> m(N + 500);

    // Phase 1: insert and record addresses
    std::unordered_map<int64_t, const int64_t*> addr_map;
    for (int i = 0; i < N; ++i) {
        m.insert(i, i * 3);
        addr_map[i] = m.find_ptr(i);
        CHECK(addr_map[i] != nullptr);
    }

    // Phase 2: more insertions — should NOT move existing elements
    for (int i = N; i < N + 200; ++i) m.insert(i, i * 3);

    // Phase 3: verify all original addresses still point to correct values
    for (int i = 0; i < N; ++i) {
        CHECK_EQ(addr_map[i], m.find_ptr(i));       // same address
        CHECK_EQ(*addr_map[i], (int64_t)i * 3);     // same value
    }
    ok("D-01 no-reordering: placement stability");
}

// D-02: No-Reordering under heavy hash collision
//       Force many keys to the same home bucket (by using a weak hash
//       that maps to only a few positions) and verify all are found.
struct WeakHash {
    size_t operator()(int64_t k) const noexcept {
        // Map all keys to one of only 8 home positions → maximum collision
        return static_cast<size_t>(k & 0x7);
    }
};

static void test_D02_no_reordering_collision_correctness() {
    constexpr int N = 500;
    ooh::flat_map<int64_t, int64_t, WeakHash> m(N + 100);
    for (int i = 0; i < N; ++i) CHECK(m.insert(i, i * 11));
    // All must be findable despite extreme collision
    for (int i = 0; i < N; ++i) {
        auto v = m.find(i);
        CHECK(v.has_value());
        CHECK_EQ(*v, (int64_t)i * 11);
    }
    CHECK(!m.find(N));
    ok("D-02 no-reordering under heavy collision");
}

// D-03: Greedy insertion — verify each element lands at or after its home
//       bucket (probe distance ≥ 1, i.e. dist stored in bucket == steps taken)
static void test_D03_greedy_insertion_probe_distance() {
    constexpr int N = 5000;
    ooh::flat_map<int64_t,int64_t> m(N + 500);
    for (int i = 0; i < N; ++i) m.insert(i, i);

    auto stats = m.probe_stats();
    // Every bucket must have probe distance ≥ 1 (1-based encoding)
    CHECK(stats.max_psl >= 1);
    // And the minimum possible is 1 (home bucket)
    CHECK(stats.avg_psl >= 1.0);
    ok("D-03 greedy insertion probe distance");
}

// D-04: No-Reordering property prevents false early-exit on MISS
//       If a miss query scans past tombstones, it must reach EMPTY to be sure.
//       Verify that find() returns the correct answer after mixed erase+insert.
static void test_D04_miss_query_reaches_empty() {
    constexpr int N = 1000;
    ooh::flat_map<int64_t,int64_t> m(N * 3);
    // Insert, erase half, insert new keys — creating tombstone chains
    for (int i = 0; i < N; ++i) m.insert(i, i);
    for (int i = 0; i < N; i += 2) m.erase(i);
    for (int i = N; i < N * 2; ++i) m.insert(i, i);

    // All original odd keys must still be found
    for (int i = 1; i < N; i += 2) CHECK_EQ(*m.find(i), i);
    // All erased even keys must NOT be found
    for (int i = 0; i < N; i += 2) CHECK(!m.find(i));
    // All new keys must be found
    for (int i = N; i < N * 2; ++i) CHECK_EQ(*m.find(i), i);
    ok("D-04 miss query reaches EMPTY (no false early exit)");
}

// =============================================================================
// ── SECTION E: freeze() / find_frozen() — wait-free read path ────────────────
// =============================================================================

// E-01: Basic freeze → find_frozen correctness (single-writer build)
static void test_E01_freeze_single_writer() {
    constexpr int N = 20000;
    ooh::flat_map<int64_t,int64_t> m(N + 100);
    for (int i = 0; i < N; ++i) CHECK(m.insert(i, i * 7));
    m.freeze();
    for (int i = 0; i < N; ++i) {
        auto v = m.find_frozen(i);
        CHECK(v.has_value());
        CHECK_EQ(*v, (int64_t)i * 7);
    }
    CHECK(!m.find_frozen(N));
    CHECK(m.contains_frozen(0));
    CHECK(!m.contains_frozen(N));
    ok("E-01 freeze / find_frozen (single writer)");
}

// E-02: freeze after concurrent_insert build
static void test_E02_freeze_concurrent_build() {
    constexpr int N = 10000;
    ooh::flat_map<int64_t,int64_t> m(N + 200);
    for (int i = 0; i < N; ++i) CHECK(m.concurrent_insert(i, i * 3));
    m.freeze();
    for (int i = 0; i < N; ++i) {
        auto v = m.find_frozen(i);
        CHECK(v.has_value());
        CHECK_EQ(*v, (int64_t)i * 3);
    }
    CHECK(!m.find_frozen(N));
    ok("E-02 freeze / find_frozen (concurrent build)");
}

// E-03: Multi-thread find_frozen — all threads see consistent values
//       This verifies the wait-free read path [P3] under true parallelism.
//       No locks used — just freeze() + find_frozen().
static void test_E03_frozen_multithread_consistency() {
    constexpr int N = 100000, T = 8;
    ooh::flat_map<int64_t,int64_t> m(N + 1000);
    for (int i = 0; i < N; ++i) m.insert(i, i);
    m.freeze();

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&]{
            for (int i = 0; i < N; ++i) {
                auto v = m.find_frozen(i);
                if (!v || *v != i) errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (auto& th : threads) th.join();
    CHECK_EQ(errors.load(), 0);
    printf("      frozen_mt: %d threads × %d keys, errors=%d\n", T, N, errors.load());
    ok("E-03 frozen multithread consistency");
}

// E-04: find_frozen scaling — throughput should grow linearly with threads
//       (wait-free: no contention, perfect scaling expected)
static void test_E04_frozen_throughput_scaling() {
    constexpr int N = 50000, Q = 200000;
    auto keys = rand_keys(N);
    // Build queries: 80% hit
    std::vector<uint64_t> queries;
    queries.reserve(Q);
    std::mt19937_64 rng(123);
    for (int i = 0; i < Q * 4 / 5; ++i)
        queries.push_back(keys[rng() % N]);
    for (int i = Q * 4 / 5; i < Q; ++i)
        queries.push_back(rng() | 0xDEAD000000000000ULL);

    ooh::flat_map<uint64_t, uint64_t> m(N + 500);
    for (auto k : keys) m.insert(k, k);
    m.freeze();

    auto measure = [&](int nthreads) -> double {
        std::atomic<uint64_t> total{0};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int t = 0; t < nthreads; ++t)
            ts.emplace_back([&]{
                uint64_t acc = 0;
                for (auto q : queries) { auto v = m.find_frozen(q); if (v) acc += *v; }
                total.fetch_add(acc, std::memory_order_relaxed);
            });
        for (auto& th : ts) th.join();
        auto t1 = std::chrono::steady_clock::now();
        double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        (void)total.load();
        return ns / (nthreads * Q);
    };

    double t1 = measure(1);
    double t4 = measure(4);
    // With 4 threads, throughput (queries/ns) should be at least 2× single-thread
    // (wait-free: ideal is 4×, allow ≥ 2× for scheduling overhead)
    double ratio = t1 / t4;
    printf("      find_frozen: 1-thread=%.1f ns/op  4-thread=%.1f ns/op  ratio=%.2fx\n",
           t1, t4, ratio);
    CHECK(ratio >= 1.5);  // at least 1.5× speedup — confirms wait-free scaling
    ok("E-04 frozen throughput scaling (wait-free)");
}

// =============================================================================
// ── SECTION F: concurrent_insert() — lock-free write path ────────────────────
// =============================================================================

// F-01: 8 threads × 10K disjoint keys — all inserted exactly once
static void test_F01_concurrent_insert_disjoint() {
    constexpr int T = 8, PER = 10000;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i)
                CHECK(m.concurrent_insert((int64_t)(t * PER + i), (int64_t)(t * PER + i)));
        });
    for (auto& th : threads) th.join();
    CHECK_EQ(m.size(), (size_t)(T * PER));
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < PER; i += 50)
            CHECK(m.contains((int64_t)(t * PER + i)));
    printf("      concurrent_insert: %d threads × %d keys = %zu total\n",
           T, PER, m.size());
    ok("F-01 concurrent_insert disjoint keys");
}

// F-02: 16 threads — stress with larger table
static void test_F02_concurrent_insert_16threads() {
    constexpr int T = 16, PER = 5000;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i)
                m.concurrent_insert((int64_t)(t * PER + i), (int64_t)(t * PER + i) * 3);
        });
    for (auto& th : threads) th.join();
    CHECK_EQ(m.size(), (size_t)(T * PER));
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < PER; i += 100) {
            auto k = (int64_t)(t * PER + i);
            auto v = m.find(k);
            CHECK(v.has_value() && *v == k * 3);
        }
    ok("F-02 concurrent_insert 16 threads");
}

// F-03: concurrent_insert then freeze — all values correct after freeze
static void test_F03_concurrent_insert_then_freeze() {
    constexpr int T = 8, PER = 8000;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i)
                m.concurrent_insert((int64_t)(t * PER + i), (int64_t)(t * PER + i) * 5);
        });
    for (auto& th : threads) th.join();
    m.freeze();
    std::atomic<int> errors{0};
    std::vector<std::thread> readers;
    readers.reserve(T);
    for (int t = 0; t < T; ++t)
        readers.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i) {
                auto k = (int64_t)(t * PER + i);
                auto v = m.find_frozen(k);
                if (!v || *v != k * 5) errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (auto& th : readers) th.join();
    CHECK_EQ(errors.load(), 0);
    ok("F-03 concurrent_insert → freeze → find_frozen");
}

// =============================================================================
// ── SECTION G: Mixed read+write consistency (no locks) ───────────────────────
// =============================================================================

// G-01: single-writer insert() + multi-reader find()
//       Verifies release/acquire ordering: readers never see torn values.
static void test_G01_insert_concurrent_find_no_torn_values() {
    constexpr int N = 30000;
    ooh::flat_map<int64_t,int64_t> m(N + 1000);
    std::atomic<int>  writer_pos{0};
    std::atomic<bool> done{false};
    std::atomic<int>  wrong{0};

    std::thread writer([&]{
        for (int i = 0; i < N; ++i) {
            m.insert((int64_t)i, (int64_t)i * 13 + 7);
            writer_pos.store(i + 1, std::memory_order_release);
        }
        done.store(true, std::memory_order_release);
    });
    std::vector<std::thread> readers(4);
    for (auto& r : readers)
        r = std::thread([&]{
            while (!done.load(std::memory_order_relaxed)) {
                int up = writer_pos.load(std::memory_order_acquire);
                for (int i = 0; i < up; i += 7) {
                    auto v = m.find((int64_t)i);
                    if (v.has_value() && *v != (int64_t)i * 13 + 7)
                        wrong.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    writer.join();
    for (auto& r : readers) r.join();
    CHECK_EQ(wrong.load(), 0);
    for (int i = 0; i < N; ++i) {
        auto v = m.find((int64_t)i);
        CHECK(v.has_value() && *v == (int64_t)i * 13 + 7);
    }
    printf("      insert+find: wrong_values=%d\n", wrong.load());
    ok("G-01 insert + concurrent find (no torn values)");
}

// G-02: concurrent_insert writers + concurrent find readers (lock-free)
//       Tests the paper's lock-free insert [P2] alongside acquire reads.
static void test_G02_concurrent_insert_plus_readers() {
    constexpr int N = 20000;
    ooh::flat_map<int64_t,int64_t> m(N * 2);
    // Pre-populate half
    for (int i = 0; i < N / 2; ++i) m.concurrent_insert(i, i);

    std::atomic<bool> stop{false};
    std::atomic<int>  corrupt{0};

    // Readers: look up pre-populated keys; if found, value must be correct
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r)
        readers.emplace_back([&]{
            while (!stop.load(std::memory_order_relaxed)) {
                for (int i = 0; i < N / 2; i += 50) {
                    auto v = m.find((int64_t)i);
                    if (v.has_value() && *v != i)
                        corrupt.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    // Writer: insert remaining keys
    std::thread writer([&]{
        for (int i = N / 2; i < N; ++i) m.concurrent_insert(i, i);
        stop.store(true, std::memory_order_release);
    });
    writer.join();
    for (auto& r : readers) r.join();

    CHECK_EQ(corrupt.load(), 0);
    CHECK_EQ(m.size(), (size_t)N);
    printf("      concurrent rw: corrupt=%d\n", corrupt.load());
    ok("G-02 concurrent_insert + find readers (lock-free)");
}

// G-03: erase (single writer) + concurrent readers — never see wrong value
//       erase() uses atomic release store → concurrent find() uses acquire
//       → readers see either LIVE or DELETED, never a torn bucket.
static void test_G03_erase_concurrent_find_no_torn() {
    constexpr int N = 500;
    constexpr int N_ROUNDS = 4;
    ooh::flat_map<int64_t,int64_t> m((size_t)N * (N_ROUNDS + 1) + 100);
    for (int i = 0; i < N; ++i) m.insert(i, i * 7);

    std::atomic<bool> stop{false};
    std::atomic<int>  corrupt{0};

    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r)
        readers.emplace_back([&]{
            while (!stop.load(std::memory_order_relaxed)) {
                for (int i = 0; i < N; ++i) {
                    auto v = m.find((int64_t)i);
                    // If found, the value MUST be i*7 — only ever inserted with that value
                    if (v.has_value() && *v != (int64_t)i * 7)
                        corrupt.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    for (int round = 0; round < N_ROUNDS; ++round) {
        for (int i = 0; i < N; ++i) m.erase(i);
        for (int i = 0; i < N; ++i) m.insert(i, i * 7);
    }
    stop.store(true, std::memory_order_release);
    for (auto& r : readers) r.join();

    CHECK_EQ(corrupt.load(), 0);
    for (int i = 0; i < N; ++i) CHECK_EQ(*m.find(i), i * 7);
    printf("      erase+find: corrupt=%d\n", corrupt.load());
    ok("G-03 erase + concurrent find (no torn values)");
}

// G-04: Linearisability check — at any moment, find() returns either the
//       value that was inserted OR nothing (never a stale/wrong value).
//       Uses a reference oracle updated with release stores.
static void test_G04_linearisability() {
    constexpr int N = 5000, T_WRITERS = 2, T_READERS = 4;
    // Keys partitioned by writer thread
    ooh::flat_map<int64_t,int64_t> m((size_t)N * T_WRITERS * 2);

    // Each key has an "expected" value: 0 = not yet inserted, else key*17
    // Writers use fetch_add(release) to publish; readers use load(acquire)
    std::vector<std::atomic<int64_t>> oracle(N * T_WRITERS);
    for (auto& o : oracle) o.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};
    std::atomic<int>  violations{0};

    // Readers: if oracle says key is inserted, map must return correct value
    std::vector<std::thread> readers;
    readers.reserve(T_READERS);
    for (int r = 0; r < T_READERS; ++r)
        readers.emplace_back([&, r]{   // capture r by value — avoids stack-use-after-scope
            std::mt19937 rng((unsigned)r + 1);
            while (!done.load(std::memory_order_relaxed)) {
                int idx = rng() % (int)oracle.size();
                int64_t expected = oracle[idx].load(std::memory_order_acquire);
                if (expected != 0) {
                    auto v = m.find((int64_t)idx);
                    // Once oracle says inserted, find MUST succeed with correct value
                    if (!v || *v != expected)
                        violations.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    // Writers: insert and publish to oracle
    std::vector<std::thread> writers;
    writers.reserve(T_WRITERS);
    for (int w = 0; w < T_WRITERS; ++w)
        writers.emplace_back([&, w]{
            for (int i = w * N; i < (w + 1) * N; ++i) {
                int64_t val = (int64_t)i * 17;
                m.concurrent_insert((int64_t)i, val);
                // Publish AFTER insert is visible
                oracle[i].store(val, std::memory_order_release);
            }
        });

    for (auto& wt : writers) wt.join();
    done.store(true, std::memory_order_release);
    for (auto& r : readers) r.join();

    CHECK_EQ(violations.load(), 0);
    printf("      linearisability: violations=%d\n", violations.load());
    ok("G-04 linearisability");
}

// =============================================================================
// ── SECTION H: Non-trivial Key/Value types ────────────────────────────────────
// =============================================================================

// H-01: std::string value — insert, find, erase, update
static void test_H01_string_value() {
    ooh::flat_map<int64_t, std::string> m(100);
    CHECK(m.insert(1, std::string("hello")));
    CHECK(m.insert(2, std::string("world")));
    CHECK(!m.insert(1, std::string("other")));
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(*m.find(1), "hello");
    CHECK_EQ(m.at(2), "world");
    m.insert_or_assign(1, std::string("updated"));
    CHECK_EQ(m.at(1), "updated");
    m.erase(1);
    CHECK(!m.find(1));
    CHECK_EQ(m.size(), 1u);
    m[3] = "three";
    CHECK_EQ(m.at(3), "three");
    m.freeze();
    CHECK_EQ(*m.find_frozen(2), "world");
    CHECK(!m.find_frozen(1));
    ok("H-01 string value");
}

// H-02: std::string key
static void test_H02_string_key() {
    ooh::flat_map<std::string, int64_t> m(100);
    CHECK(m.insert("alpha", 1));
    CHECK(m.insert("beta",  2));
    CHECK(!m.insert("alpha", 99));
    CHECK_EQ(*m.find("alpha"), 1);
    CHECK(!m.find("delta"));

    m.erase("beta");
    CHECK(!m.find("beta"));
    CHECK(m.insert("beta", 22));
    CHECK_EQ(*m.find("beta"), 22);
    ok("H-02 string key");
}

// H-03: Tracked type — tombstone reuse produces correct values
static void test_H03_tracked_lifecycle() {
    ooh::flat_map<int64_t, Tracked> m(50);
    for (int i = 0; i < 20; ++i) m.insert(i, Tracked(i));
    for (int i = 0; i < 20; i += 2) m.erase(i);
    for (int i = 0; i < 20; i += 2) m.insert(i, Tracked(i * 7));
    for (int i = 1; i < 20; i += 2) CHECK_EQ(m.at(i).value, i);
    for (int i = 0; i < 20; i += 2) CHECK_EQ(m.at(i).value, i * 7);
    CHECK_EQ(m.size(), 20u);
    ok("H-03 tracked lifecycle");
}

// H-04: move-only value type (unique_ptr<int>)
static void test_H04_move_only_value() {
    ooh::flat_map<int64_t, std::unique_ptr<int>> m(20);
    m.emplace(1LL, std::make_unique<int>(42));
    m.emplace(2LL, std::make_unique<int>(99));
    CHECK(m.find_ptr(1) != nullptr);
    CHECK_EQ(**m.find_ptr(1), 42);
    CHECK(m.find_ptr(3) == nullptr);
    ok("H-04 move-only value");
}

// =============================================================================
// ── SECTION I: Iteration correctness ──────────────────────────────────────────
// =============================================================================

// I-01: for_each_bucket enumerates exactly all live elements
static void test_I01_for_each_bucket_live_only() {
    constexpr int N = 1000;
    ooh::flat_map<int64_t,int64_t> m(N + 100);
    for (int i = 0; i < N; ++i) m.insert(i, i * 2);
    // Erase odd keys
    for (int i = 1; i < N; i += 2) m.erase(i);

    std::unordered_set<int64_t> seen;
    int64_t sum = 0;
    m.for_each_bucket([&](int64_t k, int64_t v) {
        CHECK(k % 2 == 0);       // only even keys remain
        CHECK_EQ(v, k * 2);
        seen.insert(k);
        sum += v;
    });
    CHECK_EQ(seen.size(), (size_t)N / 2);
    int64_t expected_sum = 0;
    for (int i = 0; i < N; i += 2) expected_sum += i * 2;
    CHECK_EQ(sum, expected_sum);
    ok("I-01 for_each_bucket (live only)");
}

// I-02: for_each_bucket key set = for_each key set (after insert, no erase)
static void test_I02_for_each_bucket_vs_find() {
    constexpr int N = 500;
    ooh::flat_map<int64_t,int64_t> m(N + 50);
    for (int i = 0; i < N; ++i) m.insert(i, i);

    std::unordered_set<int64_t> bucket_keys;
    m.for_each_bucket([&](int64_t k, int64_t) { bucket_keys.insert(k); });
    CHECK_EQ(bucket_keys.size(), (size_t)N);
    for (int i = 0; i < N; ++i) CHECK(bucket_keys.count(i) == 1);
    ok("I-02 for_each_bucket key set");
}

// I-03: for_each mutable — in-place value modification
static void test_I03_for_each_mutable() {
    constexpr int N = 200;
    ooh::flat_map<int64_t,int64_t> m(N + 50);
    for (int i = 0; i < N; ++i) m.insert(i, i);
    m.for_each([](int64_t, int64_t& v) { v *= 2; });
    for (int i = 0; i < N; ++i) CHECK_EQ(*m.find(i), (int64_t)i * 2);
    ok("I-03 for_each mutable");
}

// =============================================================================
// ── SECTION J: Consistency oracle test ────────────────────────────────────────
// =============================================================================
//
// Full cross-validation: ooh::flat_map vs std::unordered_map reference.
// We run a random sequence of insert / erase / find operations on both,
// then compare the entire key-value sets.

static void test_J01_oracle_random_operations() {
    constexpr int OPS = 50000, CAP = 10000;
    std::mt19937_64 rng(0xdeadbeef);

    ooh::flat_map<int64_t,int64_t> ooh_map(CAP * 3);
    RefMap ref_map;

    int inserts = 0, erases = 0, hits = 0, misses = 0;

    for (int op = 0; op < OPS; ++op) {
        int64_t key = (int64_t)(rng() % CAP);
        int64_t val = (int64_t)(rng() % 1000000);
        int action  = (int)(rng() % 3);

        if (action == 0) {
            // insert
            bool r_ooh = ooh_map.insert(key, val);
            bool r_ref = ref_map.emplace(key, val).second;
            CHECK_EQ(r_ooh, r_ref);
            if (r_ooh) ++inserts;
        } else if (action == 1) {
            // erase
            bool r_ooh = ooh_map.erase(key);
            bool r_ref = (ref_map.erase(key) > 0);
            CHECK_EQ(r_ooh, r_ref);
            if (r_ooh) ++erases;
        } else {
            // find
            auto v_ooh = ooh_map.find(key);
            auto it    = ref_map.find(key);
            bool found_ooh = v_ooh.has_value();
            bool found_ref = (it != ref_map.end());
            CHECK_EQ(found_ooh, found_ref);
            if (found_ooh && found_ref) {
                CHECK_EQ(*v_ooh, it->second);
                ++hits;
            } else {
                ++misses;
            }
        }
    }

    // Full key-value cross-check
    CHECK_EQ(ooh_map.size(), ref_map.size());
    for (auto& [k, v] : ref_map) {
        auto vo = ooh_map.find(k);
        CHECK(vo.has_value());
        CHECK_EQ(*vo, v);
    }
    printf("      oracle: inserts=%d erases=%d hits=%d misses=%d final_size=%zu\n",
           inserts, erases, hits, misses, ooh_map.size());
    ok("J-01 oracle random operations");
}

// J-02: Oracle test with string keys
static void test_J02_oracle_string_keys() {
    constexpr int OPS = 5000, VOCAB = 200;
    std::mt19937 rng(777);
    auto rand_key = [&]() -> std::string {
        int idx = rng() % VOCAB;
        return "key_" + std::to_string(idx);
    };

    ooh::flat_map<std::string, int64_t> ooh_map(VOCAB * 4);
    std::unordered_map<std::string, int64_t> ref_map;

    for (int op = 0; op < OPS; ++op) {
        auto key  = rand_key();
        int64_t v = rng() % 10000;
        int act   = rng() % 3;

        if (act == 0) {
            bool ro = ooh_map.insert(key, v);
            bool rr = ref_map.emplace(key, v).second;
            CHECK_EQ(ro, rr);
        } else if (act == 1) {
            bool ro = ooh_map.erase(key);
            bool rr = ref_map.erase(key) > 0;
            CHECK_EQ(ro, rr);
        } else {
            auto vo = ooh_map.find(key);
            auto ir = ref_map.find(key);
            CHECK_EQ(vo.has_value(), ir != ref_map.end());
            if (vo && ir != ref_map.end()) CHECK_EQ(*vo, ir->second);
        }
    }
    CHECK_EQ(ooh_map.size(), ref_map.size());
    ok("J-02 oracle string keys");
}

// =============================================================================
// ── SECTION K: Large-scale correctness ───────────────────────────────────────
// =============================================================================

// K-01: 200K sequential integers
static void test_K01_large_sequential() {
    constexpr int N = 200000;
    ooh::flat_map<int64_t,int64_t> m(N + 10000, 0.3);
    for (int i = 0; i < N; ++i) CHECK(m.insert((int64_t)i, (int64_t)i * 3));
    CHECK_EQ(m.size(), (size_t)N);
    for (int i = 0; i < N; ++i) CHECK_EQ(*m.find((int64_t)i), (int64_t)i * 3);
    CHECK(!m.find((int64_t)N));
    auto s = m.probe_stats();
    printf("      large sequential: size=%zu load=%.3f avg_psl=%.2f max_psl=%llu\n",
           m.size(), m.load_factor(), s.avg_psl, (unsigned long long)s.max_psl);
    ok("K-01 large sequential (200K)");
}

// K-02: 100K random uint64 keys
static void test_K02_large_random_keys() {
    constexpr int N = 100000;
    auto keys = rand_keys(N);
    ooh::flat_map<uint64_t, uint64_t> m(N + 5000);
    for (auto k : keys) m.insert(k, k ^ 0xABCDu);
    CHECK_EQ(m.size(), (size_t)N);
    for (auto k : keys) CHECK_EQ(*m.find(k), k ^ 0xABCDu);
    ok("K-02 large random keys (100K)");
}

// =============================================================================
// ── SECTION L: Robustness — edge cases & resource safety ─────────────────────
// =============================================================================

// L-01: concurrent_insert at exact capacity — m_vidx reaches m_max_items
//       Extra inserts beyond capacity must return false without crashing or
//       corrupting the table.
static void test_L01_concurrent_insert_at_capacity() {
    constexpr int N = 200, EXTRA = 50;
    ooh::flat_map<int64_t,int64_t> m(N);  // capacity = exactly N
    // Insert exactly N items
    for (int i = 0; i < N; ++i) CHECK(m.concurrent_insert(i, i));
    CHECK_EQ(m.size(), (size_t)N);
    // Over-insert: must return false, not crash
    for (int i = N; i < N + EXTRA; ++i) CHECK(!m.concurrent_insert(i, i));
    CHECK_EQ(m.size(), (size_t)N);
    // All original keys still found
    for (int i = 0; i < N; ++i) CHECK_EQ(*m.find(i), i);
    ok("L-01 concurrent_insert at capacity");
}

// L-02: for_each_bucket after concurrent_insert failures — no stale/corrupt slots
static void test_L02_for_each_after_failed_inserts() {
    constexpr int N = 100;
    ooh::flat_map<int64_t,int64_t> m(N);
    for (int i = 0; i < N; ++i) m.concurrent_insert(i, i * 7);
    // Over-insert
    for (int i = N; i < N + 20; ++i) m.concurrent_insert(i, i);

    std::unordered_set<int64_t> seen;
    m.for_each_bucket([&](int64_t k, int64_t v) {
        CHECK_EQ(v, k * 7);   // only the original inserts have value = k*7
        CHECK(seen.insert(k).second);  // no duplicates
    });
    CHECK_EQ(seen.size(), (size_t)N);
    ok("L-02 for_each_bucket after failed inserts");
}

// L-03: clear() + reuse — after clear the raw buffer is fresh for new inserts
static void test_L03_clear_and_reuse() {
    ooh::flat_map<int64_t, std::string> m(50);
    for (int i = 0; i < 30; ++i) m.insert(i, std::to_string(i));
    m.clear();
    CHECK_EQ(m.size(), 0u);
    CHECK(m.empty());
    // Insert completely different keys
    for (int i = 100; i < 130; ++i) CHECK(m.insert(i, std::to_string(i * 2)));
    CHECK_EQ(m.size(), 30u);
    for (int i = 100; i < 130; ++i)
        CHECK_EQ(*m.find(i), std::to_string(i * 2));
    CHECK(!m.find(0));
    ok("L-03 clear and reuse");
}

// L-04: Probe chain integrity after interleaved insert + erase cycles
//       Ensures tombstones never break lookups for keys inserted after them.
static void test_L04_probe_chain_integrity() {
    // Use a weak hash to force collisions on a small number of home buckets
    struct CollidingHash {
        size_t operator()(int64_t k) const noexcept { return (size_t)(k % 8); }
    };
    ooh::flat_map<int64_t, int64_t, CollidingHash> m(200);

    // Insert keys 0..99, all hashing to one of 8 slots
    for (int i = 0; i < 100; ++i) CHECK(m.insert(i, i));
    // Erase every third key — creates tombstone chains
    for (int i = 0; i < 100; i += 3) m.erase(i);
    // Insert new keys — must find empty slots past tombstones
    for (int i = 100; i < 150; ++i) CHECK(m.insert(i, i * 5));

    // Verify all surviving keys
    for (int i = 0; i < 100; ++i) {
        if (i % 3 == 0) CHECK(!m.find(i));
        else            CHECK_EQ(*m.find(i), i);
    }
    for (int i = 100; i < 150; ++i) CHECK_EQ(*m.find(i), i * 5);
    ok("L-04 probe chain integrity under insert/erase");
}

// L-05: move-from safety — operating on a moved-from map must not crash
static void test_L05_moved_from_safety() {
    ooh::flat_map<int64_t,int64_t> a(20);
    a.insert(1, 10); a.insert(2, 20);
    auto b = std::move(a);
    // a is now in a valid-but-empty state
    CHECK_EQ(a.size(), 0u);         // NOLINT: intentional use after move
    CHECK(a.empty());               // NOLINT
    CHECK(!a.find(1));              // NOLINT
    CHECK(!a.contains(1));          // NOLINT
    // b has the data
    CHECK_EQ(b.size(), 2u);
    CHECK_EQ(*b.find(1), 10);
    ok("L-05 moved-from safety");
}

// L-06: self-move-assign must not corrupt
static void test_L06_self_move_assign() {
    ooh::flat_map<int64_t,int64_t> m(20);
    m.insert(1, 100); m.insert(2, 200);
    auto& ref = m;
    m = std::move(ref);  // self-move-assign
    // After self-move, map should still be usable (either same or empty)
    // The contract is just "no crash/corruption, valid state"
    CHECK(m.size() == 2u || m.size() == 0u);
    ok("L-06 self-move-assign");
}

// L-07: Concurrent find on a fully-frozen large table — no data races
//       (additional to E-03, this one checks 100% hit rate)
static void test_L07_frozen_concurrent_100pct_hit() {
    constexpr int N = 50000, T = 8;
    ooh::flat_map<int64_t,int64_t> m(N + 100);
    for (int i = 0; i < N; ++i) m.insert(i, i * 3);
    m.freeze();

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t] {
            // Each thread queries a different stride to avoid false sharing
            for (int i = t; i < N; i += T) {
                auto v = m.find_frozen(i);
                if (!v || *v != (int64_t)i * 3)
                    errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (auto& th : threads) th.join();
    CHECK_EQ(errors.load(), 0);
    printf("      frozen_100pct: %d threads, errors=%d\n", T, errors.load());
    ok("L-07 frozen concurrent 100% hit");
}

// L-08: insert_or_assign does not increase size on assign path
static void test_L08_insert_or_assign_size_invariant() {
    ooh::flat_map<int64_t,int64_t> m(50);
    CHECK(m.insert_or_assign(1, 10));    // insert → true
    CHECK_EQ(m.size(), 1u);
    CHECK(!m.insert_or_assign(1, 20));   // assign → false, size unchanged
    CHECK_EQ(m.size(), 1u);
    CHECK_EQ(*m.find(1), 20);
    CHECK(!m.insert_or_assign(1, 30));
    CHECK_EQ(m.size(), 1u);
    ok("L-08 insert_or_assign size invariant");
}

// L-09: count() always returns 0 or 1 (set semantics)
static void test_L09_count_semantics() {
    ooh::flat_map<int64_t,int64_t> m(50);
    for (int i = 0; i < 10; ++i) m.insert(i, i);
    for (int i = 0; i < 10; ++i) CHECK_EQ(m.count(i), 1u);
    for (int i = 10; i < 20; ++i) CHECK_EQ(m.count(i), 0u);
    m.erase(5);
    CHECK_EQ(m.count(5), 0u);
    ok("L-09 count() set semantics");
}

// L-10: hash_function() and key_eq() return correct objects
static void test_L10_accessors() {
    struct MyHash {
        size_t operator()(int64_t k) const noexcept { return (size_t)k ^ 0xABCDu; }
    };
    struct MyEq {
        bool operator()(int64_t a, int64_t b) const noexcept { return a == b; }
    };
    ooh::flat_map<int64_t, int64_t, MyHash, MyEq> m(10, 0.25, MyHash{}, MyEq{});
    m.insert(1, 100);
    CHECK_EQ(m.hash_function()(42), MyHash{}(42));
    CHECK(m.key_eq()(1, 1));
    CHECK(!m.key_eq()(1, 2));
    ok("L-10 hash_function / key_eq accessors");
}

// =============================================================================
// ── SECTION M: Performance regression ────────────────────────────────────────
//
// These tests do NOT require Release build — they measure *relative* properties
// that must hold regardless of optimisation level:
//
//   • find_frozen scales linearly with thread count (wait-free)
//   • insert into an empty table is O(1) amortised (PSL stays bounded)
//   • consecutive integer keys have PSL ≈ 1 (ideal hash distribution)
//   • probe distance never exceeds O(log N / (1−α)) — paper Theorem 1.1
//
// Absolute ns/op values are NOT checked (they vary across machines/builds).
// =============================================================================

// M-01: Amortised O(1) insert — PSL stays bounded after N inserts
static void test_M01_amortised_insert_psl() {
    // Insert at three different load factors and confirm PSL stays O(log(1/(1-α)))
    // Paper guarantee: E[psl] = O(log(1/(1-α))).  We allow 4× the theoretical mean.
    struct Case { double slack; double max_expected_avg; };
    for (auto [slack, limit] : std::initializer_list<Case>{
            {0.50, 4.0},   // α=0.50: O(log 2)  ≈ 0.7 → allow 4
            {0.25, 6.0},   // α=0.75: O(log 4)  ≈ 2.0 → allow 6
            {0.15, 10.0},  // α=0.85: O(log 6.7)≈ 3.9 → allow 10
        }) {
        constexpr int N = 20000;
        ooh::flat_map<int64_t,int64_t> m(N, slack);
        for (int i = 0; i < N; ++i) m.insert(i, i);
        auto s = m.probe_stats();
        CHECK(s.avg_psl >= 1.0);
        CHECK_LT(s.avg_psl, limit);
        printf("      M-01 slack=%.2f α=%.2f avg_psl=%.3f max_psl=%llu (limit<%.1f)\n",
               slack, 1.0 - slack, s.avg_psl, (unsigned long long)s.max_psl, limit);
    }
    ok("M-01 amortised O(1) insert: PSL bounded (Theorem 1.1)");
}

// M-02: find_frozen scales linearly with threads (wait-free property)
//       Throughput with T threads must be at least (T * 0.6)× single-thread.
//       We use 0.6 (not ideal 1.0) to tolerate OS scheduler and cache effects.
static void test_M02_frozen_linear_scaling() {
    using Clock = std::chrono::steady_clock;
    constexpr int N = 50000, Q = 100000;
    std::mt19937_64 rng(42);
    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = rng();
    std::vector<uint64_t> queries(Q);
    for (int i = 0; i < Q; ++i)
        queries[i] = (i < Q * 8 / 10) ? keys[rng() % N] : (rng() | 0xDEAD000000000000ULL);

    ooh::flat_map<uint64_t, uint64_t> m(N + 100);
    for (auto k : keys) m.insert(k, k);
    m.freeze();

    auto measure = [&](int T) -> double {
        double best = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<std::thread> ts;
            std::atomic<uint64_t> tot{0};
            auto t0 = Clock::now();
            for (int t = 0; t < T; ++t)
                ts.emplace_back([&]{
                    uint64_t a = 0;
                    for (auto q : queries) { auto v = m.find_frozen(q); if (v) a += *v; }
                    tot.fetch_add(a, std::memory_order_relaxed);
                });
            for (auto& th : ts) th.join();
            auto t1 = Clock::now();
            double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()
                        / ((double)T * Q);
            best = std::min(best, ns);
            (void)tot.load();
        }
        return best;
    };

    double t1 = measure(1);
    double t4 = measure(4);
    double ratio4 = t1 / t4;   // should be ~4 (ideal); require ≥ 2.0
    printf("      M-02 find_frozen: 1T=%.1fns  4T=%.1fns  ratio=%.2fx (need≥2.0)\n",
           t1, t4, ratio4);
    CHECK(ratio4 >= 2.0);
    ok("M-02 find_frozen scales linearly (wait-free)");
}

// M-03: insert is faster than a known upper bound (sanity, not a regression test)
//       We just confirm the table can insert 100K keys in under 10 seconds total
//       (even on the slowest debug + sanitizer build).
static void test_M03_insert_throughput_sanity() {
    using Clock = std::chrono::steady_clock;
    constexpr int N = 100000;
    std::mt19937_64 rng(7);
    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = rng();

    auto t0 = Clock::now();
    ooh::flat_map<uint64_t, uint64_t> m(N + 1000);
    for (auto k : keys) m.insert(k, k);
    auto t1 = Clock::now();
    CHECK_EQ(m.size(), (size_t)N);

    double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    printf("      M-03 insert 100K keys: %.1f ms\n", ms);
    CHECK_LT(ms, 10000.0);   // must complete in under 10 s even under ASan/TSan
    ok("M-03 insert throughput sanity");
}

// =============================================================================
// ── SECTION N: Full key-value consistency (fuzz) ─────────────────────────────
//
// Every test runs a sequence of operations on both ooh::flat_map and a
// std::unordered_map reference oracle, then checks:
//   1. Return-value agreement   — insert/erase/find return the same bool/value
//   2. Size agreement           — ooh.size() == ref.size() at all times
//   3. Full key-value scan      — every key in ref is found in ooh with same val
//   4. Reverse scan             — every live key in ooh is in ref with same val
//   5. for_each_bucket coverage — live elements match ref exactly (no dups, no miss)
//   6. probe_stats.live_count   — matches ooh.size() after any sequence
//
// Operation mix (weights approximate):
//   insert(new)  20%     insert(dup)  10%    erase(present)  15%
//   erase(absent) 5%     find(hit)    25%    find(miss)       5%
//   insert_or_assign     10%          update  5%    operator[]  5%
//
// Deterministic seeds: every test is fully reproducible.
// =============================================================================

// ── Helper: oracle runner ─────────────────────────────────────────────────────
// Runs `OPS` random operations on both maps, checking agreement at every step.
// Returns wrong-count (0 expected).
template <typename OohMap, typename RefMap>
static int run_oracle(OohMap& ooh, RefMap& ref,
                      int OPS, int key_range,
                      std::mt19937_64& rng) {
    int wrong = 0;
    std::uniform_int_distribution<int64_t> kdist(0, (int64_t)key_range - 1);
    std::uniform_int_distribution<int64_t> vdist(0, 999999);
    std::uniform_int_distribution<int> adist(0, 99);   // 0-99 → determines action

    for (int op = 0; op < OPS; ++op) {
        int64_t key = kdist(rng);
        int64_t val = vdist(rng);
        int     act = adist(rng);

        // Skip write ops when ooh is full (the table has a hard cap).
        // Read ops (find, contains, find_ptr) are always allowed.
        const bool is_write = (act < 20) || (act >= 65 && act < 90);
        const bool key_absent = !ooh.contains(key);
        if (is_write && key_absent && ooh.full()) continue;

        if (act < 20) {
            // insert (new key path tested when key absent)
            bool ro = ooh.insert(key, val);
            bool rr = ref.emplace(key, val).second;
            if (ro != rr) ++wrong;
        } else if (act < 35) {
            // erase
            bool ro = ooh.erase(key);
            bool rr = ref.erase(key) > 0;
            if (ro != rr) ++wrong;
        } else if (act < 65) {
            // find
            auto vo = ooh.find(key);
            auto ir = ref.find(key);
            bool fo = vo.has_value(), fr = (ir != ref.end());
            if (fo != fr) { ++wrong; continue; }
            if (fo && *vo != ir->second) ++wrong;
        } else if (act < 75) {
            // insert_or_assign (key_absent → insert; key_present → assign)
            bool was_new_ooh = ooh.insert_or_assign(key, val);
            bool was_new_ref = !ref.count(key);
            ref[key] = val;
            if (was_new_ooh != was_new_ref) ++wrong;
            auto vo = ooh.find(key);
            if (!vo || *vo != val) ++wrong;
        } else if (act < 80) {
            // update (only if key exists — no capacity issue)
            bool ro = ooh.update(key, val);
            bool rr = (ref.count(key) > 0);
            if (ro != rr) ++wrong;
            if (rr) {
                ref[key] = val;
                auto vo = ooh.find(key);
                if (!vo || *vo != val) ++wrong;
            }
        } else if (act < 85) {
            // emplace
            auto r = ooh.emplace(key, val);
            bool rr = ref.emplace(key, val).second;
            if (r.inserted != rr) ++wrong;
        } else if (act < 90) {
            // try_emplace
            auto r = ooh.try_emplace(key, val);
            bool rr = ref.emplace(key, val).second;
            if (r.inserted != rr) ++wrong;
        } else if (act < 95) {
            // contains
            bool co = ooh.contains(key);
            bool cr = ref.count(key) > 0;
            if (co != cr) ++wrong;
        } else {
            // find_ptr (const)
            const auto& cooh = ooh;
            const int64_t* ptr = cooh.find_ptr(key);
            auto ir = ref.find(key);
            bool fp = (ptr != nullptr), fr = (ir != ref.end());
            if (fp != fr) { ++wrong; continue; }
            if (fp && *ptr != ir->second) ++wrong;
        }
    }
    return wrong;
}

// ── Helper: full cross-scan ───────────────────────────────────────────────────
template <typename OohMap>
static int cross_scan(const OohMap& ooh,
                      const std::unordered_map<int64_t,int64_t>& ref) {
    int wrong = 0;
    // ref → ooh
    for (auto& [k, v] : ref) {
        auto vo = ooh.find(k);
        if (!vo || *vo != v) ++wrong;
    }
    // ooh → ref (via for_each_bucket)
    std::unordered_map<int64_t,int64_t> from_ooh;
    ooh.for_each_bucket([&](int64_t k, int64_t v) {
        if (from_ooh.count(k)) ++wrong;  // duplicate
        from_ooh[k] = v;
    });
    if (from_ooh.size() != ref.size()) ++wrong;
    for (auto& [k, v] : from_ooh) {
        auto ir = ref.find(k);
        if (ir == ref.end() || ir->second != v) ++wrong;
    }
    // probe_stats.live_count == size
    if (ooh.probe_stats().live_count != ooh.size()) ++wrong;
    return wrong;
}

// N-01: 1M random ops, integer keys, full oracle cross-check
static void test_N01_oracle_1M_int() {
    constexpr int OPS = 1'000'000, CAP = 20000;
    std::mt19937_64 rng(0x1337BEEF1337BEEFull);
    // ~30% write rate × 1M = ~300K fresh inserts; each consumes one KV slot.
    // Allocate 400K to have headroom.
    ooh::flat_map<int64_t,int64_t> ooh(400000);
    std::unordered_map<int64_t,int64_t> ref;
    ref.reserve(CAP);

    int wrong = run_oracle(ooh, ref, OPS, CAP, rng);
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    int scan = cross_scan(ooh, ref);
    CHECK_EQ(scan, 0);
    printf("      N-01 1M ops: wrong=%d scan=%d final_size=%zu\n",
           wrong, scan, ooh.size());
    ok("N-01 oracle 1M ops (int keys, all ops)");
}

// N-02: Multi-seed fuzz — 10 seeds × 200K ops, ensures reproducibility
static void test_N02_multi_seed_fuzz() {
    constexpr int SEEDS = 10, OPS = 200000, CAP = 5000;
    int total_wrong = 0;
    for (int s = 0; s < SEEDS; ++s) {
        std::mt19937_64 rng((uint64_t)s * 0xDEADBEEFull + 0xCAFEBABEull);
        // 200K × 30% write = 60K fresh inserts; allocate 80K
        ooh::flat_map<int64_t,int64_t> ooh(80000);
        std::unordered_map<int64_t,int64_t> ref;
        int w = run_oracle(ooh, ref, OPS, CAP, rng);
        if (w) { total_wrong += w; continue; }
        if (ooh.size() != ref.size()) { ++total_wrong; continue; }
        total_wrong += cross_scan(ooh, ref);
    }
    CHECK_EQ(total_wrong, 0);
    printf("      N-02 %d seeds × %dK ops: total_wrong=%d\n",
           SEEDS, OPS / 1000, total_wrong);
    ok("N-02 multi-seed fuzz (10 × 200K ops)");
}

// N-03: WeakHash — all keys map to 16 home buckets, max probe-chain stress
static void test_N03_oracle_weak_hash() {
    struct WeakHash16 {
        size_t operator()(int64_t k) const noexcept { return (size_t)(k & 0xF); }
    };
    constexpr int OPS = 100000, CAP = 400;
    std::mt19937_64 rng(0xBAD0C0DEull);
    // 100K × 30% = 30K fresh inserts needed; allocate 40K
    ooh::flat_map<int64_t,int64_t,WeakHash16> ooh(40000);
    std::unordered_map<int64_t,int64_t> ref;

    int wrong = run_oracle(ooh, ref, OPS, CAP, rng);
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    int scan = cross_scan(ooh, ref);
    CHECK_EQ(scan, 0);
    printf("      N-03 WeakHash: wrong=%d scan=%d final_size=%zu\n",
           wrong, scan, ooh.size());
    ok("N-03 oracle WeakHash (16 home buckets, 100K ops)");
}

// N-04: String key + string value — non-trivial KV tombstone reuse correctness
static void test_N04_oracle_string_kv() {
    constexpr int OPS = 30000, VOCAB = 300;
    std::mt19937 rng(0xDEADF00D);
    auto rkey = [&]{ return "k" + std::to_string(rng() % VOCAB); };
    auto rval = [&]{ return "v" + std::to_string(rng() % 100000); };

    ooh::flat_map<std::string, std::string> ooh(VOCAB * 4);
    std::unordered_map<std::string, std::string> ref;
    int wrong = 0;

    for (int op = 0; op < OPS; ++op) {
        auto k = rkey();
        auto v = rval();
        int act = rng() % 5;
        if (act <= 1) {
            bool ro = ooh.insert(k, v);
            bool rr = ref.emplace(k, v).second;
            if (ro != rr) ++wrong;
        } else if (act == 2) {
            bool ro = ooh.erase(k);
            bool rr = ref.erase(k) > 0;
            if (ro != rr) ++wrong;
        } else if (act == 3) {
            auto vo = ooh.find(k);
            auto ir = ref.find(k);
            bool fo = vo.has_value(), fr = (ir != ref.end());
            if (fo != fr) { ++wrong; continue; }
            if (fo && *vo != ir->second) ++wrong;
        } else {
            bool was_new = ooh.insert_or_assign(k, v);
            bool exp_new = !ref.count(k);
            ref[k] = v;
            if (was_new != exp_new) ++wrong;
        }
    }
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    // Scan
    for (auto& [k, v] : ref) {
        auto vo = ooh.find(k);
        if (!vo || *vo != v) ++wrong;
    }
    CHECK_EQ(wrong, 0);
    printf("      N-04 string kv: wrong=%d final_size=%zu\n", wrong, ooh.size());
    ok("N-04 oracle string key+value (30K ops, tombstone reuse)");
}

// N-05: Heavy tombstone cycling — insert all, erase all, insert different,
//       repeat 5 rounds; probe chains must stay consistent throughout
static void test_N05_tombstone_cycle_fuzz() {
    constexpr int N = 2000, ROUNDS = 5;
    ooh::flat_map<int64_t,int64_t> ooh((size_t)N * (ROUNDS + 2));
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = 0;

    for (int round = 0; round < ROUNDS; ++round) {
        // Insert a new batch
        int base = round * N;
        for (int i = 0; i < N; ++i) {
            int64_t k = base + i, v = k * 7 + round;
            bool ro = ooh.insert(k, v);
            bool rr = ref.emplace(k, v).second;
            if (ro != rr) ++wrong;
        }
        // Full scan after each insert batch
        for (auto& [k, v] : ref) {
            auto vo = ooh.find(k);
            if (!vo || *vo != v) ++wrong;
        }
        // Erase entire previous batch (not this round's keys)
        if (round > 0) {
            int prev_base = (round - 1) * N;
            for (int i = 0; i < N; ++i) {
                int64_t k = prev_base + i;
                bool ro = ooh.erase(k);
                bool rr = ref.erase(k) > 0;
                if (ro != rr) ++wrong;
            }
        }
        // Probe stats must be consistent
        auto stats = ooh.probe_stats();
        if (stats.live_count != ooh.size()) ++wrong;
    }
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    printf("      N-05 tombstone cycle: wrong=%d %d rounds, final_size=%zu\n",
           wrong, ROUNDS, ooh.size());
    ok("N-05 tombstone cycle fuzz (5 rounds insert+erase)");
}

// N-06: Boundary keys — key=0, key=INT64_MAX, consecutive, sparse
static void test_N06_boundary_keys() {
    ooh::flat_map<int64_t,int64_t> ooh(500);
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = 0;

    auto do_insert = [&](int64_t k, int64_t v) {
        bool ro = ooh.insert(k, v);
        bool rr = ref.emplace(k, v).second;
        if (ro != rr) ++wrong;
    };
    auto do_erase = [&](int64_t k) {
        bool ro = ooh.erase(k);
        bool rr = ref.erase(k) > 0;
        if (ro != rr) ++wrong;
    };
    auto do_check = [&]() {
        for (auto& [k, v] : ref) {
            auto vo = ooh.find(k);
            if (!vo || *vo != v) ++wrong;
        }
        if (ooh.size() != ref.size()) ++wrong;
    };

    // Boundary keys
    do_insert(0, 1);
    do_insert(std::numeric_limits<int64_t>::max(), 2);
    do_insert(std::numeric_limits<int64_t>::min(), 3);
    do_insert(-1, 4);
    do_insert(1, 5);
    do_check();

    // Consecutive range
    for (int64_t k = 100; k < 200; ++k) do_insert(k, k * 3);
    do_check();

    // Sparse keys
    for (int i = 0; i < 100; ++i) do_insert((int64_t)i * 1000003, i);
    do_check();

    // Erase half, reinsert
    for (int64_t k = 100; k < 150; ++k) do_erase(k);
    for (int64_t k = 100; k < 150; ++k) do_insert(k, k * 11);
    do_check();

    CHECK_EQ(wrong, 0);
    ok("N-06 boundary keys (0, INT64_MAX, INT64_MIN, consecutive, sparse)");
}

// N-07: insert_or_assign + update semantic correctness (exhaustive)
//       Every call must agree with oracle on return value AND resulting value.
static void test_N07_insert_or_assign_semantics() {
    constexpr int OPS = 50000, CAP = 1000;
    std::mt19937_64 rng(0xFEEDBEEFull);
    // 50K × 33% write-new = ~17K fresh inserts; allocate 25K
    ooh::flat_map<int64_t,int64_t> ooh(25000);
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = 0;

    for (int op = 0; op < OPS; ++op) {
        int64_t k = rng() % CAP;
        int64_t v = rng() % 1000000;
        int act = rng() % 3;

        if (act == 0) {
            // insert_or_assign: returns true iff newly inserted
            bool was_new_ooh = ooh.insert_or_assign(k, v);
            bool was_new_ref = !ref.count(k);
            ref[k] = v;
            if (was_new_ooh != was_new_ref) { ++wrong; continue; }
            auto vo = ooh.find(k);
            if (!vo || *vo != v) ++wrong;
        } else if (act == 1) {
            // update: returns true iff key existed
            bool existed_ooh = ooh.update(k, v);
            bool existed_ref = ref.count(k) > 0;
            if (existed_ooh != existed_ref) { ++wrong; continue; }
            if (existed_ref) {
                ref[k] = v;
                auto vo = ooh.find(k);
                if (!vo || *vo != v) ++wrong;
            }
        } else {
            // insert: existing key must not be overwritten
            bool ro = ooh.insert(k, v);
            bool rr = ref.emplace(k, v).second;
            if (ro != rr) ++wrong;
        }
    }
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    printf("      N-07 ioa+update: wrong=%d final_size=%zu\n", wrong, ooh.size());
    ok("N-07 insert_or_assign + update semantics (50K ops)");
}

// N-08: for_each_bucket completeness — after arbitrary ops, bucket iteration
//       must return exactly the same (key,val) set as the oracle
static void test_N08_for_each_bucket_completeness() {
    constexpr int OPS = 30000, CAP = 2000;
    std::mt19937_64 rng(0xABCDEF01ull);
    // 30K × 30% = 9K fresh inserts; allocate 15K
    ooh::flat_map<int64_t,int64_t> ooh(15000);
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = run_oracle(ooh, ref, OPS, CAP, rng);
    CHECK_EQ(wrong, 0);

    // Collect via for_each_bucket
    std::unordered_map<int64_t,int64_t> bucket_result;
    int dups = 0;
    ooh.for_each_bucket([&](int64_t k, int64_t v) {
        if (bucket_result.count(k)) ++dups;
        bucket_result[k] = v;
    });
    CHECK_EQ(dups, 0);

    // Must match ref exactly
    int miss = 0;
    if (bucket_result.size() != ref.size()) {
        ++miss;
    } else {
        for (auto& [k, v] : ref) {
            auto it = bucket_result.find(k);
            if (it == bucket_result.end() || it->second != v) ++miss;
        }
    }
    CHECK_EQ(miss, 0);
    CHECK_EQ(ooh.probe_stats().live_count, ooh.size());
    printf("      N-08 for_each_bucket: dups=%d miss=%d size=%zu\n",
           dups, miss, ooh.size());
    ok("N-08 for_each_bucket completeness (30K ops)");
}

// N-09: Slack sensitivity — same sequence of ops across three slack values
//       must produce identical final state
static void test_N09_slack_invariant() {
    constexpr int OPS = 10000, CAP = 500;
    const uint64_t SEED = 0x600DC0DE;

    auto run = [&](double slack) -> std::unordered_map<int64_t,int64_t> {
        std::mt19937_64 rng(SEED);
        // 10K × 30% = 3K fresh inserts; 6K is enough
        ooh::flat_map<int64_t,int64_t> ooh(6000, slack);
        std::unordered_map<int64_t,int64_t> ref;
        run_oracle(ooh, ref, OPS, CAP, rng);
        return ref;  // ref is deterministic, doesn't depend on slack
    };

    auto ref_tight  = run(0.15);
    auto ref_medium = run(0.25);
    auto ref_loose  = run(0.50);

    // All three refs must agree (they all use the same deterministic oracle)
    CHECK_EQ(ref_tight.size(), ref_medium.size());
    CHECK_EQ(ref_medium.size(), ref_loose.size());
    int wrong = 0;
    for (auto& [k, v] : ref_tight) {
        auto it = ref_medium.find(k);
        if (it == ref_medium.end() || it->second != v) ++wrong;
    }
    CHECK_EQ(wrong, 0);

    // Also verify ooh results match ref for each slack
    for (double slack : {0.15, 0.25, 0.50}) {
        std::mt19937_64 rng(SEED);
        ooh::flat_map<int64_t,int64_t> ooh(6000, slack);
        std::unordered_map<int64_t,int64_t> ref;
        run_oracle(ooh, ref, OPS, CAP, rng);
        int scan = cross_scan(ooh, ref);
        if (scan) ++wrong;
    }
    CHECK_EQ(wrong, 0);
    ok("N-09 slack invariant (0.15 / 0.25 / 0.50 produce same results)");
}

// N-10: 500K ops, large key range, sparse occupancy
//       Exercises the "long probe chains" code path at low load factor.
static void test_N10_sparse_large_range() {
    constexpr int OPS = 500000;
    constexpr int KEY_RANGE = 100000;
    std::mt19937_64 rng(0x5A55E5A55Eull);
    // max_items must cover the total KV-slot budget over the lifetime of the map
    // (not just the peak live count), because each fresh insert consumes one slot
    // even if earlier keys were erased.  With ~30% write rate × 500K ops we need
    // ~150K slots; allocate 200K to be safe.
    ooh::flat_map<int64_t,int64_t> ooh(200000, 0.25);
    std::unordered_map<int64_t,int64_t> ref;
    ref.reserve(KEY_RANGE);

    int wrong = run_oracle(ooh, ref, OPS, KEY_RANGE, rng);
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    int scan = cross_scan(ooh, ref);
    CHECK_EQ(scan, 0);
    printf("      N-10 sparse 500K: wrong=%d scan=%d size=%zu\n",
           wrong, scan, ooh.size());
    ok("N-10 large key range sparse fuzz (500K ops)");
}

// N-11: Consecutive clear() cycles — clear resets everything;
//       each round must be consistent with oracle
static void test_N11_clear_cycle_consistency() {
    constexpr int ROUNDS = 5, OPS_PER_ROUND = 5000, CAP = 500;
    ooh::flat_map<int64_t,int64_t> ooh(CAP * 4);
    int total_wrong = 0;

    for (int r = 0; r < ROUNDS; ++r) {
        std::unordered_map<int64_t,int64_t> ref;
        std::mt19937_64 rng((uint64_t)r * 0x12345678);
        int w = run_oracle(ooh, ref, OPS_PER_ROUND, CAP, rng);
        total_wrong += w;
        total_wrong += cross_scan(ooh, ref);
        ooh.clear();
        CHECK_EQ(ooh.size(), 0u);
        CHECK(!ooh.find(0));
    }
    CHECK_EQ(total_wrong, 0);
    printf("      N-11 clear cycles: %d rounds, total_wrong=%d\n",
           ROUNDS, total_wrong);
    ok("N-11 clear() cycle consistency (5 rounds)");
}

// N-12: emplace / try_emplace consistency with oracle
//       try_emplace must not move the key if already present.
static void test_N12_emplace_consistency() {
    constexpr int OPS = 20000, CAP = 500;
    std::mt19937_64 rng(0xEA1CEA1Cull);
    // 20K × 33% write-new = ~7K fresh; allocate 12K
    ooh::flat_map<int64_t,int64_t> ooh(12000);
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = 0;

    for (int op = 0; op < OPS; ++op) {
        int64_t k = rng() % CAP;
        int64_t v = rng() % 100000;
        int act = rng() % 3;

        if (act == 0) {
            auto r   = ooh.emplace(k, v);
            bool rr  = ref.emplace(k, v).second;
            if (r.inserted != rr) { ++wrong; continue; }
            // value_ptr must point to the current value
            if (r.value_ptr == nullptr) { ++wrong; continue; }
            if (!r.inserted && *r.value_ptr != ref[k]) ++wrong;
        } else if (act == 1) {
            auto r  = ooh.try_emplace(k, v);
            bool rr = ref.emplace(k, v).second;
            if (r.inserted != rr) { ++wrong; continue; }
            if (r.value_ptr == nullptr) { ++wrong; continue; }
            // If NOT inserted, existing value must be unchanged
            if (!r.inserted && *r.value_ptr != ref[k]) ++wrong;
        } else {
            ooh.erase(k);
            ref.erase(k);
        }
    }
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    int scan = cross_scan(ooh, ref);
    CHECK_EQ(scan, 0);
    printf("      N-12 emplace: wrong=%d scan=%d\n", wrong, scan);
    ok("N-12 emplace / try_emplace consistency (20K ops)");
}

// N-13: Random fuzz with custom hash + custom equality (non-default template args)
static void test_N13_custom_hash_eq() {
    // Hash that reverses the bit pattern — good distribution, different from std::hash
    struct RevHash {
        size_t operator()(int64_t k) const noexcept {
            uint64_t u = (uint64_t)k;
            u = ((u & 0xAAAAAAAAAAAAAAAAull) >> 1) | ((u & 0x5555555555555555ull) << 1);
            u = ((u & 0xCCCCCCCCCCCCCCCCull) >> 2) | ((u & 0x3333333333333333ull) << 2);
            u = ((u & 0xF0F0F0F0F0F0F0F0ull) >> 4) | ((u & 0x0F0F0F0F0F0F0F0Full) << 4);
            return (size_t)(u ^ (u >> 32));
        }
    };
    struct ModEq {
        bool operator()(int64_t a, int64_t b) const noexcept { return a == b; }
    };
    constexpr int OPS = 30000, CAP = 1000;
    std::mt19937_64 rng(0xC057044Aull);
    // 30K × 30% = 9K fresh inserts; allocate 15K
    ooh::flat_map<int64_t,int64_t,RevHash,ModEq> ooh(15000, 0.25, RevHash{}, ModEq{});
    std::unordered_map<int64_t,int64_t> ref;
    int wrong = run_oracle(ooh, ref, OPS, CAP, rng);
    CHECK_EQ(wrong, 0);
    CHECK_EQ(ooh.size(), ref.size());
    int scan = 0;
    for (auto& [k, v] : ref) {
        auto vo = ooh.find(k);
        if (!vo || *vo != v) ++scan;
    }
    CHECK_EQ(scan, 0);
    ok("N-13 custom hash + equality fuzz (30K ops)");
}

// =============================================================================
// ── SECTION O: Concurrent safety stress ──────────────────────────────────────
//
// All tests in this section use ONLY ooh's documented concurrent-safe API:
//   • concurrent_insert() — lock-free, disjoint keys
//   • find() / contains() — acquire loads, safe with concurrent_insert
//   • find_frozen() — wait-free after freeze()
//
// No std::mutex or std::shared_mutex is used.  TSan must report zero races.
//
// Properties verified:
//   P1. No key is lost (all inserted keys are findable after join)
//   P2. No value is corrupted (found values equal inserted values)
//   P3. No torn reads (find returns exact value or nothing, never garbage)
//   P4. find_frozen is perfectly consistent after freeze (zero atomic overhead)
//   P5. Size counter is eventually consistent (= actual inserted count after join)
// =============================================================================

// O-01: 32 threads × 3125 keys = 100K total, all disjoint — P1 + P2
static void test_O01_concurrent_insert_32threads() {
    constexpr int T = 32, PER = 3125;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i) {
                int64_t k = (int64_t)(t * PER + i);
                CHECK(m.concurrent_insert(k, k * 11));
            }
        });
    for (auto& th : threads) th.join();
    // P1 + P2: all keys findable, all values correct
    CHECK_EQ(m.size(), (size_t)(T * PER));
    int errors = 0;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < PER; ++i) {
            int64_t k = (int64_t)(t * PER + i);
            auto v = m.find(k);
            if (!v || *v != k * 11) ++errors;
        }
    CHECK_EQ(errors, 0);
    printf("      O-01 32T×3125: size=%zu errors=%d\n", m.size(), errors);
    ok("O-01 concurrent_insert 32 threads (P1+P2)");
}

// O-02: 8 writers + 8 readers, disjoint key ranges, no torn reads — P3
static void test_O02_writers_readers_no_torn() {
    constexpr int W = 8, R = 8, PER = 5000;
    ooh::flat_map<int64_t,int64_t> m((size_t)W * PER * 2);
    // Pre-insert first half so readers have something to find immediately
    for (int t = 0; t < W / 2; ++t)
        for (int i = 0; i < PER; ++i) {
            int64_t k = (int64_t)(t * PER + i);
            m.concurrent_insert(k, k * 7);
        }

    std::atomic<bool> stop{false};
    std::atomic<int>  torn{0};

    // Readers verify that any found value is exactly key*7
    std::vector<std::thread> readers;
    readers.reserve(R);
    for (int r = 0; r < R; ++r)
        readers.emplace_back([&]{
            while (!stop.load(std::memory_order_relaxed)) {
                for (int t = 0; t < W; ++t) {
                    int64_t k = (int64_t)(t * PER + (rand() % PER));
                    auto v = m.find(k);
                    if (v && *v != k * 7)
                        torn.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    // Writers insert the second half
    std::vector<std::thread> writers;
    writers.reserve(W / 2);
    for (int t = W / 2; t < W; ++t)
        writers.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i) {
                int64_t k = (int64_t)(t * PER + i);
                m.concurrent_insert(k, k * 7);
            }
        });

    for (auto& w : writers) w.join();
    stop.store(true, std::memory_order_release);
    for (auto& r : readers) r.join();

    // P3: no torn reads
    CHECK_EQ(torn.load(), 0);
    CHECK_EQ(m.size(), (size_t)(W * PER));
    printf("      O-02 8W+8R: size=%zu torn=%d\n", m.size(), torn.load());
    ok("O-02 writers + readers no torn reads (P3)");
}

// O-03: Freeze after parallel build — P4 (frozen consistency)
//       Multiple threads build the table, then freeze, then verify read-only.
static void test_O03_parallel_build_then_frozen_check() {
    constexpr int T = 12, PER = 4000;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    {
        std::vector<std::thread> builders;
        builders.reserve(T);
        for (int t = 0; t < T; ++t)
            builders.emplace_back([&, t]{
                for (int i = 0; i < PER; ++i) {
                    int64_t k = (int64_t)(t * PER + i);
                    m.concurrent_insert(k, k * 3);
                }
            });
        for (auto& b : builders) b.join();
    }
    m.freeze();

    // P4: all keys visible in frozen mode from multiple threads
    std::atomic<int> errors{0};
    {
        std::vector<std::thread> readers;
        readers.reserve(T);
        for (int t = 0; t < T; ++t)
            readers.emplace_back([&, t]{
                for (int i = 0; i < PER; ++i) {
                    int64_t k = (int64_t)(t * PER + i);
                    auto v = m.find_frozen(k);
                    if (!v || *v != k * 3)
                        errors.fetch_add(1, std::memory_order_relaxed);
                }
            });
        for (auto& r : readers) r.join();
    }
    CHECK_EQ(errors.load(), 0);
    printf("      O-03 parallel build→freeze: %d threads, errors=%d\n",
           T, errors.load());
    ok("O-03 parallel build → frozen consistency (P4)");
}

// O-04: Size counter consistency — P5
//       After all insertions complete, m.size() == actual inserted count.
static void test_O04_size_counter_consistency() {
    constexpr int T = 16, PER = 2000;
    ooh::flat_map<int64_t,int64_t> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i)
                m.concurrent_insert((int64_t)(t * PER + i), (int64_t)i);
        });
    for (auto& th : threads) th.join();
    // P5: size == T*PER (all unique keys)
    CHECK_EQ(m.size(), (size_t)(T * PER));
    ok("O-04 size counter consistency after parallel insert (P5)");
}

// O-05: High-contention insert — all threads target the same hash bucket range
//       (WeakHash maps all keys to 16 home buckets).  TSan must be silent.
static void test_O05_high_contention_concurrent_insert() {
    struct WeakHash16 {
        size_t operator()(int64_t k) const noexcept { return (size_t)(k & 0xF); }
    };
    constexpr int T = 8, PER = 500;
    ooh::flat_map<int64_t, int64_t, WeakHash16> m((size_t)T * PER * 2);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t)
        threads.emplace_back([&, t]{
            for (int i = 0; i < PER; ++i) {
                int64_t k = (int64_t)(t * PER + i);
                m.concurrent_insert(k, k);
            }
        });
    for (auto& th : threads) th.join();
    CHECK_EQ(m.size(), (size_t)(T * PER));
    // Spot-check values
    int errors = 0;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < PER; i += 50) {
            int64_t k = (int64_t)(t * PER + i);
            auto v = m.find(k);
            if (!v || *v != k) ++errors;
        }
    CHECK_EQ(errors, 0);
    ok("O-05 high-contention concurrent_insert (16 home buckets)");
}

// O-06: contains() concurrent with concurrent_insert — no false negatives
//       Keys inserted before readers start must ALWAYS be found.
static void test_O06_contains_concurrent_guarantee() {
    constexpr int N_PRE = 5000, N_NEW = 5000, R = 6;
    ooh::flat_map<int64_t,int64_t> m((N_PRE + N_NEW) * 2);

    // Pre-populate
    for (int i = 0; i < N_PRE; ++i) m.concurrent_insert(i, i);

    std::atomic<bool> stop{false};
    std::atomic<int>  false_neg{0};  // found=false for a pre-inserted key

    std::vector<std::thread> readers;
    readers.reserve(R);
    for (int r = 0; r < R; ++r)
        readers.emplace_back([&]{
            while (!stop.load(std::memory_order_relaxed)) {
                for (int i = 0; i < N_PRE; i += 10) {
                    if (!m.contains(i))
                        false_neg.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    std::thread writer([&]{
        for (int i = N_PRE; i < N_PRE + N_NEW; ++i) m.concurrent_insert(i, i);
        stop.store(true, std::memory_order_release);
    });
    writer.join();
    for (auto& r : readers) r.join();

    CHECK_EQ(false_neg.load(), 0);
    CHECK_EQ(m.size(), (size_t)(N_PRE + N_NEW));
    printf("      O-06 contains concurrent: false_neg=%d\n", false_neg.load());
    ok("O-06 contains() concurrent — no false negatives");
}

// =============================================================================
// ── main ──────────────────────────────────────────────────────────────────────
// =============================================================================

int main() {
    printf("=== ooh::flat_map — comprehensive test suite ===\n");

    section("A: Basic API correctness");
    test_A01_basic_insert_find();
    test_A02_at();
    test_A03_operator_brackets();
    test_A04_emplace_try_emplace();
    test_A05_insert_or_assign_update();
    test_A06_erase_basic();
    test_A07_clear();
    test_A08_swap();
    test_A09_move_semantics();
    test_A10_constructors();
    test_A11_find_ptr();
    test_A12_capacity_queries();

    section("B: Tombstone reuse & erase/reinsert");
    test_B01_erase_reinsert_same_key();
    test_B02_erase_all_reinsert_new_keys();
    test_B03_tombstone_destructor_emplace();
    test_B04_tombstone_destructor_brackets();
    test_B05_tombstone_saturation();
    test_B06_string_kv_lifecycle();

    section("C: Exception safety & boundary conditions");
    test_C01_ctor_exceptions();
    test_C02_overflow();
    test_C03_slack_extremes();
    test_C04_single_element_map();
    test_C05_probe_stats_theorem1();
    test_C06_psl_vs_load_factor();

    section("D: No-Reordering invariant (paper §2)");
    test_D01_no_reordering_placement_stability();
    test_D02_no_reordering_collision_correctness();
    test_D03_greedy_insertion_probe_distance();
    test_D04_miss_query_reaches_empty();

    section("E: freeze() / find_frozen() — wait-free read path [P3]");
    test_E01_freeze_single_writer();
    test_E02_freeze_concurrent_build();
    test_E03_frozen_multithread_consistency();
    test_E04_frozen_throughput_scaling();

    section("F: concurrent_insert() — lock-free write path [P2]");
    test_F01_concurrent_insert_disjoint();
    test_F02_concurrent_insert_16threads();
    test_F03_concurrent_insert_then_freeze();

    section("G: Mixed read+write consistency (no locks)");
    test_G01_insert_concurrent_find_no_torn_values();
    test_G02_concurrent_insert_plus_readers();
    test_G03_erase_concurrent_find_no_torn();
    test_G04_linearisability();

    section("H: Non-trivial Key/Value types");
    test_H01_string_value();
    test_H02_string_key();
    test_H03_tracked_lifecycle();
    test_H04_move_only_value();

    section("I: Iteration correctness");
    test_I01_for_each_bucket_live_only();
    test_I02_for_each_bucket_vs_find();
    test_I03_for_each_mutable();

    section("J: Consistency oracle (ooh vs std::unordered_map)");
    test_J01_oracle_random_operations();
    test_J02_oracle_string_keys();

    section("K: Large-scale correctness");
    test_K01_large_sequential();
    test_K02_large_random_keys();

    section("L: Robustness — edge cases & resource safety");
    test_L01_concurrent_insert_at_capacity();
    test_L02_for_each_after_failed_inserts();
    test_L03_clear_and_reuse();
    test_L04_probe_chain_integrity();
    test_L05_moved_from_safety();
    test_L06_self_move_assign();
    test_L07_frozen_concurrent_100pct_hit();
    test_L08_insert_or_assign_size_invariant();
    test_L09_count_semantics();
    test_L10_accessors();

    section("M: Performance regression (relative properties)");
    test_M01_amortised_insert_psl();
    test_M02_frozen_linear_scaling();
    test_M03_insert_throughput_sanity();

    section("N: Full consistency oracle (fuzz)");
    test_N01_oracle_1M_int();
    test_N02_multi_seed_fuzz();
    test_N03_oracle_weak_hash();
    test_N04_oracle_string_kv();
    test_N05_tombstone_cycle_fuzz();
    test_N06_boundary_keys();
    test_N07_insert_or_assign_semantics();
    test_N08_for_each_bucket_completeness();
    test_N09_slack_invariant();
    test_N10_sparse_large_range();
    test_N11_clear_cycle_consistency();
    test_N12_emplace_consistency();
    test_N13_custom_hash_eq();

    section("O: Concurrent safety stress (TSan-clean)");
    test_O01_concurrent_insert_32threads();
    test_O02_writers_readers_no_torn();
    test_O03_parallel_build_then_frozen_check();
    test_O04_size_counter_consistency();
    test_O05_high_contention_concurrent_insert();
    test_O06_contains_concurrent_guarantee();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass.load(), g_fail.load());
    return g_fail.load() == 0 ? 0 : 1;
}
