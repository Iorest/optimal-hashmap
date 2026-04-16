# Contributing to ooh

Thank you for your interest in contributing!  
This document explains how to get started, what the coding conventions are,  
and what the review process looks like.

---

## Table of Contents

1. [Ways to contribute](#ways-to-contribute)
2. [Development setup](#development-setup)
3. [Coding conventions](#coding-conventions)
4. [Running tests](#running-tests)
5. [Running benchmarks](#running-benchmarks)
6. [Submitting a pull request](#submitting-a-pull-request)
7. [Commit message format](#commit-message-format)

---

## Ways to contribute

| Type | Where |
|------|--------|
| Bug reports | [GitHub Issues](../../issues) — use the *Bug Report* template |
| Feature requests | [GitHub Issues](../../issues) — use the *Feature Request* template |
| Questions | [GitHub Discussions](../../discussions) |
| Code / docs | Pull Request against `master` |
| Security issues | [Private advisory](../../security/advisories/new) — **never** file a public issue |

---

## Development setup

```sh
git clone https://github.com/your-org/ooh.git
cd ooh

# Debug build with AddressSanitizer (recommended for development)
cmake -B build-dbg -DCMAKE_BUILD_TYPE=Debug \
      -DOOH_BUILD_TESTS=ON \
      -DOOH_ENABLE_ASAN=ON
cmake --build build-dbg --parallel

# Release build
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release \
      -DOOH_BUILD_TESTS=ON
cmake --build build-rel --parallel
```

### Optional: ThreadSanitizer

```sh
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug \
      -DOOH_BUILD_TESTS=ON \
      -DOOH_ENABLE_TSAN=ON
cmake --build build-tsan --parallel
ctest --test-dir build-tsan -V
```

### Dependencies

`ooh` itself has **zero dependencies** (C++17 + STL only).  
Benchmarks optionally pull in [ankerl/unordered_dense](https://github.com/martinus/unordered_dense) via CMake `FetchContent`.

---

## Coding conventions

- **Style**: follow `.clang-format` (based on LLVM, 4-space indent, 100-column limit).  
  Run `clang-format -i include/ooh/flat_map.hpp` before committing.
- **Naming**:  
  - Public API: `snake_case` (matching STL)  
  - Private helpers: `_snake_case` with leading underscore  
  - Types inside the class: `PascalCase`  
  - Macros: `OOH_ALL_CAPS`
- **No new external dependencies** in the library header itself.
- **No `using namespace std`** in headers.
- Every new public method needs a doc comment and a test case.
- Thread-safety guarantees must be stated explicitly in the doc comment.

---

## Running tests

```sh
# All tests
ctest --test-dir build-dbg -V --output-on-failure

# Specific test
ctest --test-dir build-dbg -V -R flat_map
```

Tests live in `tests/test_flat_map.cpp`.  
Add new test functions following the numbered sections already there.  
Each function must be called from `main()`.

---

## Running benchmarks

```sh
cmake -B build-bench -DCMAKE_BUILD_TYPE=Release \
      -DOOH_BUILD_BENCH=ON \
      -DOOH_BUILD_GBENCH=ON
cmake --build build-bench --parallel

# Custom bench (ooh vs ankerl)
./build-bench/bench_ooh

# Google Benchmark suite
./build-bench/bench_ooh_gbench --benchmark_filter=Insert
```

When submitting a performance-related PR, please include before/after numbers.

---

## Submitting a pull request

1. Fork the repository and create a branch: `git checkout -b fix/my-fix`.
2. Make your changes, write tests, and update docs if needed.
3. Ensure all checks pass locally (see *Running tests* above).
4. Open a PR against `master`.  
   Fill in the PR template — especially the checklist.
5. A maintainer will review within a few business days.  
   Please be patient and responsive to review comments.

---

## Commit message format

```
<type>(<scope>): <short summary>

[optional body]

[optional footer: Fixes #123]
```

**Types**: `fix`, `feat`, `perf`, `test`, `docs`, `refactor`, `build`, `ci`  
**Scope** (optional): `flat_map`, `bench`, `cmake`, `ci`, `docs`

Examples:
```
fix(flat_map): destroy KV on tombstone reuse for non-trivial types
perf(flat_map): use __builtin_expect on probe loop EMPTY check
test(flat_map): add string-key edge cases
docs: clarify erase() thread-safety precondition
```

---

## Code of Conduct

Be kind and constructive.  
Harassment or disrespectful behaviour will not be tolerated.
