# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.1] — 2026-04-16

### Added
- GitHub Actions CI: multi-platform (Linux GCC/Clang, macOS, Windows MSVC/clang-cl)
- GitHub Actions CI: AddressSanitizer + UBSan, ThreadSanitizer jobs
- GitHub Actions CI: code coverage with lcov + Codecov upload
- GitHub Actions: CodeQL security analysis (weekly + on push)
- GitHub Actions: automated Release workflow triggered by version tags
- GitHub Actions: benchmark regression tracking on PRs (Google Benchmark)
- Google Benchmark integration (`-DOOH_BUILD_GBENCH=ON`) with hit-rate parametrisation
- Standalone bench: 0%/50%/80%/100% hit rates, multi-thread ladders, `shared_mutex` RW comparison
- Test suite section L: 10 new robustness tests (capacity overflow, probe chain integrity,
  moved-from safety, self-move-assign, frozen 100% hit concurrency, etc.)
- `find_ptr()` / `find_frozen_ptr()` zero-copy lookup API
- `OOH_FORCEINLINE` / `OOH_PREFETCH` / `OOH_LIKELY` portability macros (MSVC support)
- `vcpkg.json`, `CONTRIBUTING.md`, `SECURITY.md`, PR template, issue templates
- `.clang-format`, `.clang-tidy` configuration

### Fixed
- **Critical — data corruption**: `erase()` stored `vidx=0` in tombstone bucket instead
  of preserving original `value_idx`; caused all tombstone-reuse inserts to overwrite
  KV slot 0
- **Critical — data corruption**: `_place_cas` rollback on duplicate detection had the
  same `vidx=0` bug
- **Critical — resource leak**: `emplace()`, `try_emplace()`, `operator[]` tombstone
  paths used assignment instead of `~KV()` + placement-new, leaking non-trivial types
- **Critical — resource leak**: `concurrent_insert()` did not destroy the KV object
  when `_place_cas` failed (duplicate key or table full)
- **Memory management**: replaced `make_unique<KV[]>(max_items)` (default-constructs all
  slots) with `make_unique<KVSlot[]>(max_items)` (raw aligned bytes); only inserted slots
  are constructed via placement-new — same model as `std::vector`
- Move-assignment: source unique_ptrs not explicitly reset before in-place reconstruction
- `erase()` doc comment: concurrent `find()` IS safe alongside single-writer `erase()`
- CMake: version was `1.0.0` in `CMakeLists.txt` while header was `1.0.1`; now consistent
- README: `bench_ooh` path corrected (`build/bench_ooh`, not `build/bench/bench_ooh`)

### Changed
- CMake: `add_sanitizers` renamed to `ooh_add_sanitizers` (avoids name clashes)
- CMake: `FetchContent_Declare` for `unordered_dense` declared once, shared by both bench targets
- `for_each_bucket` promoted in README/docs as preferred iteration method over `for_each`

---

## [1.0.0] — 2025-01-15

### Added
- Initial release: `ooh::flat_map<Key, Value, Hash, KeyEqual>`
- Lock-free `concurrent_insert()` via single CAS
- Wait-free `find_frozen()` / `contains_frozen()` after `freeze()`
- Single-writer `insert`, `emplace`, `try_emplace`, `insert_or_assign`, `update`,
  `erase`, `operator[]`, `at`
- `for_each` (insertion-order) and `for_each_bucket` (bucket-order) iteration
- `probe_stats()` diagnostics
- Range constructor and initializer-list constructor
- CMake install support with `oohConfig.cmake`
- Full test suite (sections A–K, 42 tests)
- Benchmark suite vs `ankerl::unordered_dense`
