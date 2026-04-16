---
name: Pull Request
about: Describe your changes

---

## Summary

<!-- One paragraph describing what this PR does and why -->

## Type of change

- [ ] Bug fix (non-breaking, fixes an issue)
- [ ] New feature (non-breaking, adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation / comment only
- [ ] Performance improvement
- [ ] Refactoring (no functional change)

## Checklist

- [ ] The code compiles cleanly with `-Wall -Wextra -Wpedantic`
- [ ] New / modified functionality is covered by tests in `tests/test_flat_map.cpp`
- [ ] All existing tests pass (`ctest --test-dir build -V`)
- [ ] AddressSanitizer clean (`-DOOH_ENABLE_ASAN=ON` Debug build)
- [ ] ThreadSanitizer clean (`-DOOH_ENABLE_TSAN=ON` Debug build) if concurrency was touched
- [ ] Public API changes are reflected in `README.md` API Reference section
- [ ] Performance-sensitive changes include benchmark data (before/after)

## Test output

```
# paste: ctest --test-dir build -V
```

## Benchmark delta (if applicable)

```
# paste: ./build/bench_ooh
```
