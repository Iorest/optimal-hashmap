# ooh — Optimal Open-Addressing Hash Map

[![CI](https://github.com/your-org/ooh/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ooh/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Header-only](https://img.shields.io/badge/header--only-✓-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**`ooh::flat_map`** — fixed-capacity, lock-free, header-only C++ hash map based on:

> *"Optimal Bounds for Open Addressing Without Reordering"*  
> Farach-Colton, Krapivin, Kuszmaul — [arXiv:2501.02305](https://arxiv.org/abs/2501.02305), 2025

| Property | Detail |
|----------|--------|
| **No rehash** | Capacity fixed at construction |
| **Lock-free insert** | `concurrent_insert()` — single CAS, no mutex |
| **Wait-free reads** | `find_frozen()` — zero atomic overhead after `freeze()` |
| **Thread-safe reads** | `find()` safe alongside `concurrent_insert()` |
| **Header-only** | `include/ooh/flat_map.hpp`, C++17, no deps |

---

## Performance (Apple M4 Pro, clang -O3)

vs [ankerl::unordered_dense](https://github.com/martinus/unordered_dense) v4.4.0, random `uint64_t` keys.

| Scenario | N | ooh | ankerl | winner |
|----------|---|-----|--------|--------|
| insert | 100K–1M | 6–12 ns | 8–20 ns | **ooh 1.3–1.7×** |
| insert | 2M–5M | 16–22 ns | 12–18 ns | ankerl 0.8× |
| find 80% hit | ≤1M | 5–11 ns | 8–12 ns | **ooh 1.1–2.1×** |
| find 100% hit | ≤500K | 4–6 ns | 8–12 ns | **ooh 1.3–3×** |
| find 80% hit | ≥2M | 13–24 ns | 12–18 ns | ankerl 0.8× |
| find 0% miss | ≥500K | 9–49 ns | 2–7 ns | ankerl 4–8×¹ |
| find_frozen 8T | 100K | **0.8 ns** | 1.7 ns | ooh 2× |
| concurrent_insert 8T | 1M | 104 ns | 74 ns (mutex) | ankerl 0.7× |
| mixed RW 2W+6R | 1M | **171 ns** | 34,000 ns (rwlock) | **ooh 200×** |

¹ No-Reordering must scan to EMPTY; Robin Hood exits early via probe-distance.
  Gap is fundamental (trade-off for lock-free inserts), not a bug.
  At 80–100% hit rate ooh equals or beats ankerl.

---

## Quick Start

```cpp
#include "ooh/flat_map.hpp"
```

**Single-writer build → frozen reads:**
```cpp
ooh::flat_map<uint64_t, MyData*> table(10'000'000);
for (auto& [k, v] : data) table.insert(k, v);
table.freeze();
// Any number of reader threads — wait-free, zero atomic overhead:
auto result = table.find_frozen(query_key);
```

**Multi-writer build:**
```cpp
ooh::flat_map<uint64_t, int32_t> index(50'000'000);
// Each thread inserts a disjoint key range:
parallel_for(shards, [&](auto& shard) {
    for (auto& [k, v] : shard) index.concurrent_insert(k, v);
});
index.freeze();
```

**Live concurrent insert + read:**
```cpp
ooh::flat_map<uint64_t, Record*> cache(1'000'000);
// Writer (lock-free):
cache.concurrent_insert(key, val);
// Readers (acquire-load, always safe):
auto r = cache.find(key);
```

---

## API Reference

### Construction
```cpp
flat_map(size_t max_items, double slack = 0.25);
// slack: empty fraction (0,1). 0.25→~75% load (default). 0.40→~60% (lower miss cost).

flat_map(InputIt first, InputIt last, size_t hint = 0, double slack = 0.25);
flat_map(std::initializer_list<std::pair<Key,Value>> il, double slack = 0.25);
```

### Writes (single-writer)
| Method | Returns | Notes |
|--------|---------|-------|
| `insert(key, val)` | `bool` | `false` if key exists |
| `emplace(key, args...)` | `insert_result` | In-place value construct |
| `try_emplace(key, args...)` | `insert_result` | Won't move key if present |
| `insert_or_assign(key, val)` | `bool` | `true` = inserted |
| `update(key, val)` | `bool` | `false` if absent |
| `operator[](key)` | `Value&` | Default-constructs if absent |
| `erase(key)` | `bool` | Tombstone; slot recycled on next insert |
| `clear()` | — | Requires no concurrent ops |

### Writes (multi-writer)
| Method | Notes |
|--------|-------|
| `concurrent_insert(key, val)` | Lock-free; keys must be disjoint across threads |

### Reads
| Method | Thread-safe with | Notes |
|--------|-----------------|-------|
| `find(key) → optional<Value>` | `concurrent_insert` | acquire-load |
| `find_ptr(key) → Value*` | `concurrent_insert` | zero-copy |
| `contains(key) → bool` | `concurrent_insert` | |
| `at(key) → Value&` | single-writer only | throws `out_of_range` |
| `freeze()` | — | seq_cst fence; call once after all writes |
| `find_frozen(key) → optional<Value>` | all readers | wait-free |
| `find_frozen_ptr(key) → const Value*` | all readers | wait-free, zero-copy |
| `contains_frozen(key) → bool` | all readers | wait-free |

### Utilities
```cpp
size_t size();  capacity();  max_size();
bool   empty(); full();
double load_factor();  max_load_factor();

m.for_each_bucket([](const Key& k, const Value& v) { ... });  // safe after erase
m.for_each([](const Key& k, const Value& v) { ... });          // WARNING: includes erased slots

auto s = m.probe_stats();  // s.avg_psl, s.max_psl, s.live_count
```

---

## Hash Function

`std::hash<Key>` is used by default.  For sequential/structured integer keys
(`std::hash<uint64_t>` is identity on libc++), use `ooh::hash<T>` which applies
Murmur3-finalizer mixing and also eliminates the internal fingerprint multiply:

```cpp
ooh::flat_map<uint64_t, V, ooh::hash<uint64_t>> m(N);
```

---

## Building

```sh
# Run tests
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build -V

# ASan/UBSan
cmake -B build -DOOH_ENABLE_ASAN=ON && cmake --build build && ctest --test-dir build -V

# TSan
cmake -B build -DOOH_ENABLE_TSAN=ON && cmake --build build && ctest --test-dir build -V

# Benchmarks (auto-fetches ankerl::unordered_dense)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOOH_BUILD_BENCH=ON
cmake --build build && ./build/bench_ooh
```

**CMake FetchContent:**
```cmake
FetchContent_Declare(ooh GIT_REPOSITORY https://github.com/your-org/ooh.git GIT_TAG v1.0.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(ooh)
target_link_libraries(myapp PRIVATE ooh::ooh)
```

---

## Design

**Why No-Reordering?**  Robin Hood hashing moves elements to keep probe distances short, but concurrent inserts would need to touch multiple slots atomically.  No-Reordering places each element in the first available slot and never moves it — enabling a single CAS per insert and true lock-free concurrency.

**Bucket layout** (8 bytes, `std::atomic<uint64_t>`):
```
[63..32] value_idx    index into KV array
[31..16] fingerprint  16-bit hash mix (0 = EMPTY)
[15.. 0] probe dist   1-based (df=0xFFFFFFFF = DELETED)
```
The 16-bit fingerprint filters ~99.998% of false key comparisons before touching the KV array.

---

## License

MIT — see [LICENSE](LICENSE).

```bibtex
@misc{farach2025optimal,
  title  = {Optimal Bounds for Open Addressing Without Reordering},
  author = {Farach-Colton, Martin and Krapivin, Andrew and Kuszmaul, William},
  year   = {2025}, url = {https://arxiv.org/abs/2501.02305}
}
```
