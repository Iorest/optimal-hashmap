# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ Active  |

## Reporting a Vulnerability

**Please do NOT file a public GitHub issue for security vulnerabilities.**

Use [GitHub Private Vulnerability Reporting](../../security/advisories/new) instead.

Include:
- A description of the vulnerability and its impact
- Steps to reproduce (a minimal C++ snippet if possible)
- Affected version(s)
- Any suggested mitigation

We aim to acknowledge reports within **48 hours** and provide a fix within **14 days** for critical issues.

## Scope

Security issues in `ooh` typically relate to:
- Memory safety (out-of-bounds, use-after-free, data races)
- Undefined behaviour reachable from public API
- Denial-of-service via hash collision amplification

## Out of Scope

- Performance regressions (use GitHub Issues)
- Feature requests (use GitHub Issues / Discussions)
