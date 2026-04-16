// bench/bench_gbench.cpp  —  Google Benchmark suite for ooh::flat_map
//
// Build:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release -DOOH_BUILD_GBENCH=ON
//   cmake --build build
//   ./build/bench_ooh_gbench [--benchmark_filter=<regex>]
//   ./build/bench_ooh_gbench --benchmark_format=json --benchmark_out=results.json

#include "ooh/flat_map.hpp"

#if __has_include(<ankerl/unordered_dense.h>)
#  include <ankerl/unordered_dense.h>
#  define HAS_ANKERL 1
#else
#  define HAS_ANKERL 0
#endif

#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

// ── Shared key/query generators ───────────────────────────────────────────────

static std::vector<uint64_t> make_keys(int n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();
    return v;
}

// hit_rate ∈ [0,1]: fraction of queries that are guaranteed hits.
// Miss keys come from a separate independent RNG stream to avoid clustering
// under identity hash (using a high-bit mask would bias slot distribution).
static std::vector<uint64_t> make_queries(
        const std::vector<uint64_t>& keys, int nq,
        double hit_rate, uint64_t seed = 99) {
    std::mt19937_64 rng_hit(seed);
    std::mt19937_64 rng_miss(seed ^ UINT64_C(0xdeadbeefcafe1234));
    std::uniform_int_distribution<int> pick(0, (int)keys.size() - 1);
    std::vector<uint64_t> q;
    q.reserve(nq);
    int hits = (int)(nq * hit_rate);
    for (int i = 0; i < hits; ++i)   q.push_back(keys[pick(rng_hit)]);
    for (int i = hits; i < nq; ++i)  q.push_back(rng_miss());
    std::mt19937 sh(42);
    std::shuffle(q.begin(), q.end(), sh);
    return q;
}

// ── A. Insert ─────────────────────────────────────────────────────────────────

static void BM_ooh_Insert(benchmark::State& state) {
    const int N = (int)state.range(0);
    auto keys = make_keys(N);
    for (auto _ : state) {
        ooh::flat_map<uint64_t, uint64_t> m(N, 0.25);
        for (auto k : keys) m.insert(k, k);
        benchmark::DoNotOptimize(m.size());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ooh_Insert)
    ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Arg(2'000'000)->Arg(5'000'000)
    ->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
static void BM_ankerl_Insert(benchmark::State& state) {
    const int N = (int)state.range(0);
    auto keys = make_keys(N);
    for (auto _ : state) {
        ankerl::unordered_dense::map<uint64_t, uint64_t> m;
        m.reserve(N);
        for (auto k : keys) m.emplace(k, k);
        benchmark::DoNotOptimize(m.size());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ankerl_Insert)
    ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Arg(2'000'000)->Arg(5'000'000)
    ->Unit(benchmark::kNanosecond);
#endif

// ── B. Find — parametrised hit rate ──────────────────────────────────────────
// state.range(0) = N, state.range(1) = hit_rate * 100

static void BM_ooh_Find(benchmark::State& state) {
    const int    N   = (int)state.range(0);
    const double hit = state.range(1) / 100.0;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, 1'000'000, hit);
    ooh::flat_map<uint64_t, uint64_t> m(N, 0.25);
    for (auto k : keys) m.insert(k, k);

    for (auto _ : state) {
        uint64_t acc = 0;
        for (auto x : q) { auto v = m.find(x); if (v) acc += *v; }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * (long long)q.size());
}
// Rows: N ∈ {100K, 500K, 1M, 2M, 5M} × hit ∈ {0, 50, 80, 100}
BENCHMARK(BM_ooh_Find)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 2'000'000, 5'000'000}, {0, 50, 80, 100}})
    ->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
static void BM_ankerl_Find(benchmark::State& state) {
    const int    N   = (int)state.range(0);
    const double hit = state.range(1) / 100.0;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, 1'000'000, hit);
    ankerl::unordered_dense::map<uint64_t, uint64_t> m;
    m.reserve(N);
    for (auto k : keys) m.emplace(k, k);

    for (auto _ : state) {
        uint64_t acc = 0;
        for (auto x : q) { auto it = m.find(x); if (it != m.end()) acc += it->second; }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * (long long)q.size());
}
BENCHMARK(BM_ankerl_Find)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 2'000'000, 5'000'000}, {0, 50, 80, 100}})
    ->Unit(benchmark::kNanosecond);
#endif

// ── C. find_ptr — zero-copy ───────────────────────────────────────────────────

static void BM_ooh_FindPtr(benchmark::State& state) {
    const int    N   = (int)state.range(0);
    const double hit = state.range(1) / 100.0;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, 1'000'000, hit);
    ooh::flat_map<uint64_t, uint64_t> m(N, 0.25);
    for (auto k : keys) m.insert(k, k);

    for (auto _ : state) {
        uint64_t acc = 0;
        for (auto x : q) { const uint64_t* p = m.find_ptr(x); if (p) acc += *p; }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * (long long)q.size());
}
BENCHMARK(BM_ooh_FindPtr)
    ->ArgsProduct({{100'000, 1'000'000, 5'000'000}, {0, 80, 100}})
    ->Unit(benchmark::kNanosecond);

// ── D. find_frozen — relaxed reads ───────────────────────────────────────────

static void BM_ooh_FindFrozen(benchmark::State& state) {
    const int    N   = (int)state.range(0);
    const double hit = state.range(1) / 100.0;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, 1'000'000, hit);
    ooh::flat_map<uint64_t, uint64_t> m(N, 0.25);
    for (auto k : keys) m.insert(k, k);
    m.freeze();

    for (auto _ : state) {
        uint64_t acc = 0;
        for (auto x : q) { auto v = m.find_frozen(x); if (v) acc += *v; }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * (long long)q.size());
}
BENCHMARK(BM_ooh_FindFrozen)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 2'000'000, 5'000'000}, {0, 50, 80, 100}})
    ->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
// ankerl has no frozen mode; plain find on read-only map for fair comparison
static void BM_ankerl_Find_Frozen(benchmark::State& state) {
    const int    N   = (int)state.range(0);
    const double hit = state.range(1) / 100.0;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, 1'000'000, hit);
    ankerl::unordered_dense::map<uint64_t, uint64_t> m;
    m.reserve(N);
    for (auto k : keys) m.emplace(k, k);

    for (auto _ : state) {
        uint64_t acc = 0;
        for (auto x : q) { auto it = m.find(x); if (it != m.end()) acc += it->second; }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * (long long)q.size());
}
BENCHMARK(BM_ankerl_Find_Frozen)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 2'000'000, 5'000'000}, {0, 50, 80, 100}})
    ->Unit(benchmark::kNanosecond);
#endif

// ── E. Multi-thread find_frozen — wait-free scaling ──────────────────────────

template <int THREADS>
static void BM_ooh_FindFrozen_MT(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int Q = 200'000;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, Q, 0.8);
    ooh::flat_map<uint64_t, uint64_t> m(N, 0.25);
    for (auto k : keys) m.insert(k, k);
    m.freeze();

    for (auto _ : state) {
        std::vector<std::thread> ts;
        std::atomic<uint64_t> tot{0};
        for (int t = 0; t < THREADS; ++t)
            ts.emplace_back([&] {
                uint64_t a = 0;
                for (auto x : q) { auto v = m.find_frozen(x); if (v) a += *v; }
                tot.fetch_add(a, std::memory_order_relaxed);
            });
        for (auto& th : ts) th.join();
        benchmark::DoNotOptimize(tot.load());
    }
    state.SetItemsProcessed(state.iterations() * (long long)THREADS * Q);
}
BENCHMARK_TEMPLATE(BM_ooh_FindFrozen_MT, 1) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ooh_FindFrozen_MT, 2) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ooh_FindFrozen_MT, 4) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ooh_FindFrozen_MT, 8) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
template <int THREADS>
static void BM_ankerl_FindFrozen_MT(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int Q = 200'000;
    auto keys = make_keys(N);
    auto q    = make_queries(keys, Q, 0.8);
    ankerl::unordered_dense::map<uint64_t, uint64_t> m;
    m.reserve(N);
    for (auto k : keys) m.emplace(k, k);

    for (auto _ : state) {
        std::vector<std::thread> ts;
        std::atomic<uint64_t> tot{0};
        for (int t = 0; t < THREADS; ++t)
            ts.emplace_back([&] {
                uint64_t a = 0;
                for (auto x : q) { auto it = m.find(x); if (it != m.end()) a += it->second; }
                tot.fetch_add(a, std::memory_order_relaxed);
            });
        for (auto& th : ts) th.join();
        benchmark::DoNotOptimize(tot.load());
    }
    state.SetItemsProcessed(state.iterations() * (long long)THREADS * Q);
}
BENCHMARK_TEMPLATE(BM_ankerl_FindFrozen_MT, 1) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ankerl_FindFrozen_MT, 2) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ankerl_FindFrozen_MT, 4) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ankerl_FindFrozen_MT, 8) ->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
#endif

// ── F. Multi-thread concurrent_insert — lock-free vs mutex ───────────────────

template <int THREADS>
static void BM_ooh_ConcurrentInsert(benchmark::State& state) {
    const int N = (int)state.range(0);
    auto all = make_keys(N, 5);
    std::vector<std::vector<uint64_t>> shards(THREADS);
    for (int i = 0; i < N; ++i) shards[i % THREADS].push_back(all[i]);

    for (auto _ : state) {
        ooh::flat_map<uint64_t, uint64_t> m(N + N / 4, 0.25);
        std::vector<std::thread> ts;
        for (int t = 0; t < THREADS; ++t)
            ts.emplace_back([&, t] { for (auto k : shards[t]) m.concurrent_insert(k, k); });
        for (auto& th : ts) th.join();
        benchmark::DoNotOptimize(m.size());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK_TEMPLATE(BM_ooh_ConcurrentInsert, 2) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ooh_ConcurrentInsert, 4) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ooh_ConcurrentInsert, 8) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
template <int THREADS>
static void BM_ankerl_ConcurrentInsert_Mutex(benchmark::State& state) {
    const int N = (int)state.range(0);
    auto all = make_keys(N, 5);
    std::vector<std::vector<uint64_t>> shards(THREADS);
    for (int i = 0; i < N; ++i) shards[i % THREADS].push_back(all[i]);

    for (auto _ : state) {
        ankerl::unordered_dense::map<uint64_t, uint64_t> m;
        m.reserve(N);
        std::mutex mx;
        std::vector<std::thread> ts;
        for (int t = 0; t < THREADS; ++t)
            ts.emplace_back([&, t] {
                for (auto k : shards[t]) { std::lock_guard lg(mx); m.emplace(k, k); }
            });
        for (auto& th : ts) th.join();
        benchmark::DoNotOptimize(m.size());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK_TEMPLATE(BM_ankerl_ConcurrentInsert_Mutex, 2) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ankerl_ConcurrentInsert_Mutex, 4) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_ankerl_ConcurrentInsert_Mutex, 8) ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)->Unit(benchmark::kNanosecond);
#endif

// ── G. Mixed RW: concurrent_insert (lock-free) + find vs shared_mutex ─────────

static void BM_ooh_RWMixed(benchmark::State& state) {
    const int N = (int)state.range(0);
    constexpr int WRITERS = 2, READERS = 6;
    auto all = make_keys(N, 7);
    std::vector<std::vector<uint64_t>> shards(WRITERS);
    for (int i = 0; i < N; ++i) shards[i % WRITERS].push_back(all[i]);
    auto q = make_queries(all, 300'000, 0.5);

    for (auto _ : state) {
        ooh::flat_map<uint64_t, uint64_t> m(N + N / 4, 0.25);
        std::atomic<bool> done{false};
        std::atomic<uint64_t> racc{0};

        std::vector<std::thread> writers;
        for (int t = 0; t < WRITERS; ++t)
            writers.emplace_back([&, t] { for (auto k : shards[t]) m.concurrent_insert(k, k); });

        std::vector<std::thread> readers;
        for (int r = 0; r < READERS; ++r)
            readers.emplace_back([&] {
                uint64_t a = 0;
                while (!done.load(std::memory_order_relaxed))
                    for (auto x : q) { auto v = m.find(x); if (v) a += *v; }
                racc.fetch_add(a, std::memory_order_relaxed);
            });

        for (auto& w : writers) w.join();
        done.store(true, std::memory_order_release);
        for (auto& r : readers) r.join();
        benchmark::DoNotOptimize(racc.load());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ooh_RWMixed)
    ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)
    ->Unit(benchmark::kNanosecond);

#if HAS_ANKERL
static void BM_ankerl_RWMixed_SharedMutex(benchmark::State& state) {
    const int N = (int)state.range(0);
    constexpr int WRITERS = 2, READERS = 6;
    auto all = make_keys(N, 7);
    std::vector<std::vector<uint64_t>> shards(WRITERS);
    for (int i = 0; i < N; ++i) shards[i % WRITERS].push_back(all[i]);
    auto q = make_queries(all, 300'000, 0.5);

    for (auto _ : state) {
        ankerl::unordered_dense::map<uint64_t, uint64_t> m;
        m.reserve(N);
        std::shared_mutex smx;
        std::atomic<bool> done{false};
        std::atomic<uint64_t> racc{0};

        std::vector<std::thread> writers;
        for (int t = 0; t < WRITERS; ++t)
            writers.emplace_back([&, t] {
                for (auto k : shards[t]) { std::unique_lock lk(smx); m.emplace(k, k); }
            });

        std::vector<std::thread> readers;
        for (int r = 0; r < READERS; ++r)
            readers.emplace_back([&] {
                uint64_t a = 0;
                while (!done.load(std::memory_order_relaxed)) {
                    std::shared_lock lk(smx);
                    for (auto x : q) { auto it = m.find(x); if (it != m.end()) a += it->second; }
                }
                racc.fetch_add(a, std::memory_order_relaxed);
            });

        for (auto& w : writers) w.join();
        done.store(true, std::memory_order_release);
        for (auto& r : readers) r.join();
        benchmark::DoNotOptimize(racc.load());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ankerl_RWMixed_SharedMutex)
    ->Arg(100'000)->Arg(500'000)->Arg(1'000'000)
    ->Unit(benchmark::kNanosecond);
#endif
