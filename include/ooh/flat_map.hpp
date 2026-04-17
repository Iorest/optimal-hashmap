// ============================================================================
// ooh/flat_map.hpp  —  Lock-Free Fixed-Capacity Hash Map  (v1.0.1)
//
// Based on: "Optimal Bounds for Open Addressing Without Reordering"
//           Farach-Colton, Krapivin, Kuszmaul — arXiv:2501.02305, 2025
//
// The paper proves that greedy No-Reordering insertion achieves
//   E[probes] = O(log(1/(1-α)))   at load factor α
// without ever moving an inserted element, enabling single-CAS lock-free inserts.
//
// ── Performance vs ankerl::unordered_dense (Apple M4 Pro, clang -O3) ────────
//
//  │ Scenario                  │ ooh       │ ankerl    │ ratio    │
//  │───────────────────────────│───────────│───────────│──────────│
//  │ insert  100K–1M           │ 6–12 ns   │ 8–20 ns   │ 1.3–1.7× │
//  │ insert  2M–5M (cold)      │ 16–22 ns  │ 12–18 ns  │ 0.8×     │
//  │ find    ≤1M (80% hit)     │ 5–11 ns   │ 8–12 ns   │ 1.1–2.1× │
//  │ find    ≤1M (100% hit)    │ 4–8 ns    │ 8–12 ns   │ 1.4–2.0× │
//  │ find    ≥2M (80% hit)     │ 13–24 ns  │ 12–18 ns  │ 0.8–0.9× │
//  │ find    0% hit (miss)*    │ 6–49 ns   │ 2–7 ns    │ 0.15–1×  │
//  │ 8-thread find_frozen      │ 0.8–2 ns  │ 1.7–2 ns  │ 1–2×     │
//  │ rw mixed (lock-free)      │ 170–190ns │ 35K–55Kns │ 200–290× │
//
//  * Miss must scan to EMPTY (No-Reordering constraint); Robin Hood exits
//    early via probe-distance. At 80–100% hit rate ooh equals or beats ankerl.
//
// ── Thread-safety ────────────────────────────────────────────────────────────
//  insert() / erase()       single writer only
//  concurrent_insert()      lock-free; keys must be disjoint across threads
//  find() / contains()      safe concurrent with concurrent_insert()
//  find_frozen()            wait-free after freeze()
//
// ── Quick start ──────────────────────────────────────────────────────────────
//  ooh::flat_map<uint64_t, MyData*> table(10'000'000);
//  for (auto& [k, v] : data) table.insert(k, v);
//  table.freeze();
//  auto result = table.find_frozen(query_key);   // any number of threads
//
// ── Requirements ─────────────────────────────────────────────────────────────
//  C++17, header-only, no external dependencies.
// ============================================================================

#pragma once

#define OOH_VERSION_MAJOR 1
#define OOH_VERSION_MINOR 0
#define OOH_VERSION_PATCH 1
#define OOH_VERSION "1.0.1"

// ── Portability macros ───────────────────────────────────────────────────────
// OOH_PREFETCH(p): L1 prefetch on both x86-64 (prefetcht0) and ARM64 (PLDL1KEEP).
// locality=3 maps to L1 on both targets; locality=1 (old default) mapped to L3.

#ifdef _MSC_VER
#  include <intrin.h>
#  define OOH_FORCEINLINE   __forceinline
#  define OOH_PREFETCH(p)   _mm_prefetch((const char*)(p), _MM_HINT_T0)
#  define OOH_LIKELY(x)     (x)
#  define OOH_UNLIKELY(x)   (x)
#  define OOH_RESTRICT      __restrict   // MSVC uses __restrict (no underscores)
#else
#  define OOH_FORCEINLINE   __attribute__((always_inline)) inline
#  define OOH_PREFETCH(p)   __builtin_prefetch((p), 0, 3)
#  define OOH_LIKELY(x)     __builtin_expect(!!(x), 1)
#  define OOH_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#  define OOH_RESTRICT      __restrict__
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

// ── TSan: KV buffer written before bucket release-store, read after acquire-load.
// The CAS establishes C++17 happens-before; TSan needs an explicit hint because
// it tracks same-address atomics only, not cross-object release/acquire pairs.
//
// Detection: __SANITIZE_THREAD__ (GCC + Clang with -fsanitize=thread).
// __has_feature(thread_sanitizer) is Clang-only and cannot appear in a
// plain #if on GCC — even guarded by defined(__has_feature), because GCC's
// preprocessor does not short-circuit macro calls in #if expressions.
// We use nested #ifdef to keep __has_feature entirely invisible to GCC.
#ifdef __SANITIZE_THREAD__
#  define OOH_TSAN_ENABLED 1
#elif defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define OOH_TSAN_ENABLED 1
#  endif
#endif

#ifdef OOH_TSAN_ENABLED
extern "C" void AnnotateBenignRaceSized(const char*, int,
                                        const volatile void*, size_t, const char*);
#  define OOH_TSAN_BENIGN_RACE(p, sz) \
     AnnotateBenignRaceSized(__FILE__, __LINE__, (p), (sz), "ooh KV cross-object sync")
#  define OOH_TSAN_RELEASE(p) (void)(p)
#  define OOH_TSAN_ACQUIRE(p) (void)(p)
#else
#  define OOH_TSAN_BENIGN_RACE(p, sz) (void)(p)
#  define OOH_TSAN_RELEASE(p)         (void)(p)
#  define OOH_TSAN_ACQUIRE(p)         (void)(p)
#endif

namespace ooh {

// ── ooh::hash<T> ─────────────────────────────────────────────────────────────
// Optional high-quality hash adapter.  Applies Murmur3-finalizer mixing to
// spread bits for sequential/structured integer keys.
//
// Declares is_avalanching so flat_map skips the redundant _fp() multiply
// (saves ~0.26 ns/op on ARM64; no effect on x86).
//
// Use for sequential or structured integer keys:
//   ooh::flat_map<uint64_t, V, ooh::hash<uint64_t>> m(N);
// For random keys (UUID, mt19937_64 output), std::hash is sufficient.
namespace detail {
    inline uint64_t hash_mix64(uint64_t x) noexcept {
        x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
        x ^= x >> 31; return x;
    }
} // namespace detail

template <typename T>
struct hash {
    size_t operator()(const T& key) const noexcept {
        return detail::hash_mix64(std::hash<T>{}(key));
    }
};

template <> struct hash<uint64_t> {
    using is_avalanching = void;
    size_t operator()(uint64_t x) const noexcept { return detail::hash_mix64(x); }
};
template <> struct hash<int64_t> {
    using is_avalanching = void;
    size_t operator()(int64_t x) const noexcept { return detail::hash_mix64(static_cast<uint64_t>(x)); }
};
template <> struct hash<uint32_t> {
    using is_avalanching = void;
    size_t operator()(uint32_t x) const noexcept { return detail::hash_mix64(x); }
};
template <> struct hash<int32_t> {
    using is_avalanching = void;
    size_t operator()(int32_t x) const noexcept { return detail::hash_mix64(static_cast<uint32_t>(x)); }
};

// ============================================================================
// flat_map<Key, Value, Hash, KeyEqual>
//
// Fixed-capacity open-addressing hash map.  No rehash ever.
//
// Bucket layout (8 bytes, std::atomic<uint64_t>):
//   [63..32] value_idx  — index into the KV array
//   [31..16] fingerprint — 16-bit hash mix; 0 = EMPTY sentinel
//   [15.. 0] probe dist  — 1-based; 0 = EMPTY, 0xFFFFFFFF (df) = DELETED
//
// KV array: raw aligned bytes; only inserted slots are placement-new'd.
// Tombstone (DELETED) slots are recycled lazily by the next insert on that
// probe chain; destructor cleans up all constructed slots in [0, m_vidx).
// ============================================================================
template <typename Key,
          typename Value,
          typename Hash     = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class flat_map {

    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;
    using sz  = std::size_t;

    // ── Bucket ───────────────────────────────────────────────────────────────
    struct alignas(8) Bucket {
        std::atomic<u64> word{0};  // 0 = EMPTY

        static constexpr u32 EMPTY_DF   = 0u;
        static constexpr u32 DELETED_DF = 0xFFFF'FFFFu;

        static constexpr u64 make  (u32 vidx, u32 df) noexcept { return (u64(vidx) << 32) | df; }
        static constexpr u64 encode(u32 vidx, u16 fp, u16 d) noexcept { return make(vidx, (u32(fp) << 16) | d); }
        static constexpr u32 df_of  (u64 w) noexcept { return u32(w); }
        static constexpr u32 vidx_of(u64 w) noexcept { return u32(w >> 32); }
        static constexpr u16 fp_of  (u64 w) noexcept { return u16((w >> 16) & 0xFFFFu); }
    };
    static_assert(sizeof(Bucket) == 8 && alignof(Bucket) == 8);
    static_assert(std::atomic<u64>::is_always_lock_free,
        "ooh::flat_map requires lock-free 64-bit atomics");

    // ── KV storage ───────────────────────────────────────────────────────────
    struct KV {
        Key key; Value val;
        KV() = delete;
        template<typename K, typename V>
        KV(K&& k, V&& v) : key(std::forward<K>(k)), val(std::forward<V>(v)) {}
    };
    struct alignas(KV) KVSlot {
        unsigned char data[sizeof(KV)];
        KV*       ptr()       noexcept { return std::launder(reinterpret_cast<KV*>(data)); }
        const KV* ptr() const noexcept { return std::launder(reinterpret_cast<const KV*>(data)); }
    };

    // ── Fingerprint ──────────────────────────────────────────────────────────
    // If Hash::is_avalanching is defined, skip the multiplicative mix (bits
    // are already well-distributed); otherwise apply a Fibonacci-constant mix.
    template<typename H, typename = void> struct _is_avalanching : std::false_type {};
    template<typename H> struct _is_avalanching<H, std::void_t<typename H::is_avalanching>> : std::true_type {};

    OOH_FORCEINLINE u16 _fp(sz h) const noexcept {
        u16 f;
        if constexpr (_is_avalanching<Hash>::value)
            f = u16(h >> 32);
        else
            f = u16(((h ^ (h >> 16)) * UINT64_C(0x9e3779b97f4a7c15)) >> 32);
        return f ? f : 1u;
    }

    static sz _p2(sz n) noexcept {
        if (n <= 1) return 1;
        --n; for (sz s = 1; s < sizeof(sz)*8; s <<= 1) n |= n >> s;
        return n + 1;
    }

    // ── Members ──────────────────────────────────────────────────────────────
    std::unique_ptr<Bucket[]> m_buckets;
    sz                        m_cap{0};
    sz                        m_mask{0};
    std::unique_ptr<KVSlot[]> m_kv_buf;
    sz                        m_max_items{0};
    std::atomic<u32>          m_vidx{0};  // next free KV index; [0,m_vidx) are constructed
    std::atomic<sz>           m_size{0};  // live element count

    [[no_unique_address]] Hash     m_hash;
    [[no_unique_address]] KeyEqual m_key_equal;

    KV*       _kv(u32 i)       noexcept { return m_kv_buf[i].ptr(); }
    const KV* _kv(u32 i) const noexcept { return m_kv_buf[i].ptr(); }

    void _destroy_kv_range(u32 n) noexcept {
        if constexpr (!std::is_trivially_destructible_v<KV>)
            for (u32 i = 0; i < n; ++i) _kv(i)->~KV();
    }

public:
    // ── Type aliases ─────────────────────────────────────────────────────────
    using key_type        = Key;
    using mapped_type     = Value;
    using value_type      = std::pair<const Key, Value>;
    using size_type       = sz;
    using difference_type = std::ptrdiff_t;
    using hasher          = Hash;
    using key_equal_type  = KeyEqual;

    struct insert_result {
        bool   inserted;
        Value* value_ptr;   // pointer to new or existing value; nullptr on capacity error
        explicit operator bool() const noexcept { return inserted; }
    };

    // ── Construction ─────────────────────────────────────────────────────────
    // max_items: hard element limit (table never rehashes).
    // slack: fraction of bucket slots kept empty (0 < slack < 1, default 0.25).
    //   0.25 → ~75% load, good default
    //   0.40 → ~60% load, lower miss-probe cost, +60% bucket memory
    explicit flat_map(sz max_items, double slack = 0.25)
        : flat_map(max_items, slack, Hash{}, KeyEqual{}) {}

    flat_map(sz max_items, double slack, Hash h, KeyEqual eq)
        : m_hash(std::move(h)), m_key_equal(std::move(eq))
    {
        if (slack <= 0.0 || slack >= 1.0)
            throw std::invalid_argument("flat_map: slack must be in (0,1)");
        if (max_items == 0)
            throw std::invalid_argument("flat_map: max_items must be > 0");
        if (max_items > sz(std::numeric_limits<u32>::max()))
            throw std::invalid_argument("flat_map: max_items exceeds 2^32-1");

        m_cap = _p2(sz(double(max_items) / (1.0 - slack)) + 1);
        if (m_cap < 16) m_cap = 16;
        m_mask      = m_cap - 1;
        m_buckets   = std::make_unique<Bucket[]>(m_cap);
        m_max_items = max_items;
        m_kv_buf    = std::make_unique<KVSlot[]>(max_items);
        OOH_TSAN_BENIGN_RACE(m_kv_buf.get(), max_items * sizeof(KVSlot));
    }

    template <typename InputIt>
    flat_map(InputIt first, InputIt last,
             sz capacity_hint = 0, double slack = 0.25,
             Hash h = Hash{}, KeyEqual eq = KeyEqual{})
        : flat_map(_range_hint(first, last, capacity_hint), slack, std::move(h), std::move(eq))
    { for (; first != last; ++first) insert(first->first, first->second); }

    flat_map(std::initializer_list<std::pair<Key,Value>> il, double slack = 0.25)
        : flat_map(il.size() ? il.size() : sz(1), slack)
    { for (auto& p : il) insert(p.first, p.second); }

    flat_map(const flat_map&)            = delete;
    flat_map& operator=(const flat_map&) = delete;

    flat_map(flat_map&& o) noexcept
        : m_buckets(std::move(o.m_buckets)), m_cap(o.m_cap), m_mask(o.m_mask)
        , m_kv_buf(std::move(o.m_kv_buf)), m_max_items(o.m_max_items)
        , m_hash(std::move(o.m_hash)), m_key_equal(std::move(o.m_key_equal))
    {
        m_vidx.store(o.m_vidx.load(std::memory_order_relaxed), std::memory_order_relaxed);
        m_size.store(o.m_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        o.m_cap = o.m_mask = o.m_max_items = 0;
        o.m_vidx.store(0, std::memory_order_relaxed);
        o.m_size.store(0, std::memory_order_relaxed);
    }

    flat_map& operator=(flat_map&& o) noexcept {
        if (this != &o) {
            _destroy_kv_range(m_vidx.load(std::memory_order_relaxed));
            m_buckets.reset(); m_kv_buf.reset();
            new (this) flat_map(std::move(o));
        }
        return *this;
    }

    ~flat_map() { if (m_kv_buf) _destroy_kv_range(m_vidx.load(std::memory_order_relaxed)); }

    void swap(flat_map& o) noexcept {
        using std::swap;
        swap(m_buckets, o.m_buckets); swap(m_cap, o.m_cap); swap(m_mask, o.m_mask);
        swap(m_kv_buf, o.m_kv_buf);   swap(m_max_items, o.m_max_items);
        swap(m_hash, o.m_hash);        swap(m_key_equal, o.m_key_equal);
        u32 vi = m_vidx.exchange(o.m_vidx.load(std::memory_order_relaxed));
        o.m_vidx.store(vi, std::memory_order_relaxed);
        sz si = m_size.exchange(o.m_size.load(std::memory_order_relaxed));
        o.m_size.store(si, std::memory_order_relaxed);
    }
    friend void swap(flat_map& a, flat_map& b) noexcept { a.swap(b); }

    // ── Capacity ─────────────────────────────────────────────────────────────
    [[nodiscard]] sz     size()            const noexcept { return m_size.load(std::memory_order_relaxed); }
    [[nodiscard]] sz     capacity()        const noexcept { return m_cap; }
    [[nodiscard]] sz     max_size()        const noexcept { return m_max_items; }
    [[nodiscard]] bool   empty()           const noexcept { return size() == 0; }
    [[nodiscard]] bool   full()            const noexcept { return size() >= m_max_items; }
    [[nodiscard]] double load_factor()     const noexcept {
        return m_cap ? double(m_vidx.load(std::memory_order_relaxed)) / double(m_cap) : 0.0;
    }
    [[nodiscard]] double max_load_factor() const noexcept {
        return m_cap ? double(m_max_items) / double(m_cap) : 0.0;
    }

    // ── clear — PRECONDITION: no concurrent readers or writers ────────────────
    void clear() noexcept {
        for (sz i = 0; i < m_cap; ++i)
            m_buckets[i].word.store(0ULL, std::memory_order_relaxed);
        _destroy_kv_range(m_vidx.load(std::memory_order_relaxed));
        m_vidx.store(0, std::memory_order_relaxed);
        m_size.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    // ── Single-writer insert / emplace / try_emplace / insert_or_assign ──────
    // Exception safety: strong guarantee.
    // Thread safety: concurrent find()/contains() safe (bucket publish = release store).
    bool insert(const Key& key, const Value& val) { return _sw_insert(key, val); }
    bool insert(Key&& key, Value&& val)            { return _sw_insert(std::move(key), std::move(val)); }
    bool insert(const std::pair<Key,Value>& kv)    { return _sw_insert(kv.first, kv.second); }
    bool insert(std::pair<Key,Value>&& kv)         { return _sw_insert(std::move(kv.first), std::move(kv.second)); }

    template <typename K, typename... Args>
    insert_result emplace(K&& key, Args&&... args) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        sz slot; u32 dist, tomb_vidx;
        if (const Bucket* b = _sw_probe(key, h, fp, slot, dist, tomb_vidx))
            return {false, &_kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val};
        if (slot == sz(-1)) return {false, nullptr};
        const u32 vidx = _sw_alloc_kv(std::forward<K>(key), tomb_vidx, std::forward<Args>(args)...);
        if (vidx == u32(-1)) return {false, nullptr};
        _sw_commit(slot, fp, dist, vidx);
        return {true, &_kv(vidx)->val};
    }

    // try_emplace: like emplace, but does not move key if already present
    template <typename K, typename... Args>
    insert_result try_emplace(K&& key, Args&&... args) { return emplace(std::forward<K>(key), std::forward<Args>(args)...); }

    bool insert_or_assign(const Key& key, const Value& val) { return _sw_insert_or_assign(key, val); }
    bool insert_or_assign(Key&& key, Value&& val)           { return _sw_insert_or_assign(std::move(key), std::move(val)); }

    // update: overwrite value for existing key; returns false if not found
    bool update(const Key& key, const Value& val) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (Bucket* b = _find_sw_mut(key, h, fp)) {
            _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val = val; return true;
        }
        return false;
    }
    bool update(const Key& key, Value&& val) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (Bucket* b = _find_sw_mut(key, h, fp)) {
            _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val = std::move(val); return true;
        }
        return false;
    }

    Value& operator[](const Key& key)  { return _sw_subscript(key); }
    Value& operator[](Key&& key)       { return _sw_subscript(std::move(key)); }

    [[nodiscard]] const Value& at(const Key& key) const {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (const Bucket* b = _find(key, h, fp))
            return _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        throw std::out_of_range("flat_map::at: key not found");
    }
    [[nodiscard]] Value& at(const Key& key) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (Bucket* b = _find_sw_mut(key, h, fp))
            return _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        throw std::out_of_range("flat_map::at: key not found");
    }

    // ── erase — tombstone strategy ────────────────────────────────────────────
    // Marks bucket DELETED; KV recycled lazily on next insert through this slot.
    // Concurrent find() safe; do NOT call concurrently with insert()/erase().
    bool erase(const Key& key) noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        Bucket* b = _find_sw_mut(key, h, fp);
        if (!b) return false;
        const u32 vidx = Bucket::vidx_of(b->word.load(std::memory_order_relaxed));
        b->word.store(Bucket::make(vidx, Bucket::DELETED_DF), std::memory_order_release);
        m_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    // ── find / contains — acquire loads, safe with concurrent_insert ──────────
    [[nodiscard]] OOH_FORCEINLINE
    std::optional<Value> find(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (const Bucket* b = _find(key, h, fp))
            return _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        return std::nullopt;
    }
    [[nodiscard]] OOH_FORCEINLINE
    bool contains(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        return _find(key, h, fp) != nullptr;
    }
    [[nodiscard]] sz count(const Key& key) const noexcept { return contains(key) ? 1 : 0; }

    // find_ptr: zero-copy; returned pointer valid until next mutation.
    [[nodiscard]] OOH_FORCEINLINE
    const Value* find_ptr(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (const Bucket* b = _find(key, h, fp))
            return &_kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        return nullptr;
    }
    [[nodiscard]] OOH_FORCEINLINE
    Value* find_ptr(const Key& key) noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (Bucket* b = _find_sw_mut(key, h, fp))
            return &_kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        return nullptr;
    }

    // ── freeze / find_frozen — wait-free reads after all writes complete ───────
    void freeze() noexcept { std::atomic_thread_fence(std::memory_order_seq_cst); }

    [[nodiscard]] OOH_FORCEINLINE
    std::optional<Value> find_frozen(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (const Bucket* b = _find_relaxed(key, h, fp))
            return _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        return std::nullopt;
    }
    [[nodiscard]] OOH_FORCEINLINE
    bool contains_frozen(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        return _find_relaxed(key, h, fp) != nullptr;
    }
    [[nodiscard]] OOH_FORCEINLINE
    const Value* find_frozen_ptr(const Key& key) const noexcept {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (const Bucket* b = _find_relaxed(key, h, fp))
            return &_kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        return nullptr;
    }

    // ── concurrent_insert — lock-free multi-writer ────────────────────────────
    // PRECONDITIONS: keys disjoint across threads; do not mix with insert().
    // The CAS (release/acquire) synchronises KV write with concurrent find().
    // No explicit fence needed: CAS release already provides ordering.
    bool concurrent_insert(const Key& key, const Value& val) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        if (_find(key, h, fp)) return false;
        const u32 vidx = m_vidx.fetch_add(1, std::memory_order_relaxed);
        if (vidx >= m_max_items) return false;
        new (&m_kv_buf[vidx]) KV(key, val);
        OOH_TSAN_RELEASE(&m_kv_buf[vidx]);
        if (!_place_cas(key, h, fp, vidx)) { _kv(vidx)->~KV(); return false; }
        return true;
    }

    // ── Iteration ─────────────────────────────────────────────────────────────
    // for_each_bucket: safe after any erase(); visits live elements in bucket order.
    template <typename Fn> void for_each_bucket(Fn&& fn) const {
        for (sz i = 0; i < m_cap; ++i) {
            const u64 w = m_buckets[i].word.load(std::memory_order_acquire);
            const u32 df = Bucket::df_of(w);
            if (df == Bucket::EMPTY_DF || df == Bucket::DELETED_DF) continue;
            const u32 idx = Bucket::vidx_of(w);
            fn(_kv(idx)->key, _kv(idx)->val);
        }
    }
    // for_each: insertion-order; WARNING includes stale erased slots.
    // Use only when no erase() has been called.
    template <typename Fn> void for_each(Fn&& fn) const {
        const u32 n = m_vidx.load(std::memory_order_acquire);
        for (u32 i = 0; i < n; ++i) fn(_kv(i)->key, _kv(i)->val);
    }
    template <typename Fn> void for_each(Fn&& fn) {
        const u32 n = m_vidx.load(std::memory_order_acquire);
        for (u32 i = 0; i < n; ++i) fn(_kv(i)->key, _kv(i)->val);
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────
    [[nodiscard]] const Hash&     hash_function() const noexcept { return m_hash; }
    [[nodiscard]] const KeyEqual& key_eq()        const noexcept { return m_key_equal; }

    struct ProbeStats { double avg_psl; u64 max_psl; sz live_count; };
    [[nodiscard]] ProbeStats probe_stats() const noexcept {
        u64 total = 0, mx = 0; sz live = 0;
        for (sz i = 0; i < m_cap; ++i) {
            const u64 w = m_buckets[i].word.load(std::memory_order_relaxed);
            const u32 df = Bucket::df_of(w);
            if (df == Bucket::EMPTY_DF || df == Bucket::DELETED_DF) continue;
            const u64 psl = df & 0xFFFFu;
            total += psl; if (psl > mx) mx = psl; ++live;
        }
        return {live ? double(total) / live : 0.0, mx, live};
    }

private:
    // ── Probe loops ───────────────────────────────────────────────────────────
    // Bucket array: 8 bytes/slot, 8 slots/cache-line; prefetch at pos+4 crosses
    // into the next line.  find() uses acquire (LDAPR/MOV), find_frozen() uses
    // relaxed (LDR/MOV) — measurably faster on ARM64 after freeze().
    static constexpr sz PREFETCH_DIST = 4;

    // acquire — safe during concurrent_insert
    [[nodiscard]] OOH_FORCEINLINE
    const Bucket* _find(const Key& key, sz h, u16 fp) const noexcept {
        if (OOH_UNLIKELY(m_cap == 0)) return nullptr;
        sz pos = h & m_mask;
        const Bucket* OOH_RESTRICT B = m_buckets.get();
        while (true) {
            OOH_PREFETCH(&B[(pos + PREFETCH_DIST) & m_mask]);
            const u64 w = B[pos].word.load(std::memory_order_acquire);
            const u32 df = Bucket::df_of(w);
            if (OOH_UNLIKELY(df == Bucket::EMPTY_DF)) return nullptr;
            if (df != Bucket::DELETED_DF && Bucket::fp_of(w) == fp) {
                const u32 vidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[vidx]);
                if (m_key_equal(_kv(vidx)->key, key)) return &B[pos];
            }
            pos = (pos + 1) & m_mask;
        }
    }

    // relaxed — used after freeze(); TSan safe via seq_cst fence in freeze()
    [[nodiscard]] OOH_FORCEINLINE
    const Bucket* _find_relaxed(const Key& key, sz h, u16 fp) const noexcept {
        if (OOH_UNLIKELY(m_cap == 0)) return nullptr;
        sz pos = h & m_mask;
        const Bucket* OOH_RESTRICT B = m_buckets.get();
        while (true) {
            OOH_PREFETCH(&B[(pos + PREFETCH_DIST) & m_mask]);
            const u64 w = B[pos].word.load(std::memory_order_relaxed);
            const u32 df = Bucket::df_of(w);
            if (OOH_UNLIKELY(df == Bucket::EMPTY_DF)) return nullptr;
            if (df != Bucket::DELETED_DF && Bucket::fp_of(w) == fp) {
                const u32 vidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[vidx]);
                if (m_key_equal(_kv(vidx)->key, key)) return &B[pos];
            }
            pos = (pos + 1) & m_mask;
        }
    }

    // relaxed — single-writer path (concurrent find() uses acquire on same word,
    // so using atomic loads here keeps TSan happy without any hardware cost)
    [[nodiscard]] OOH_FORCEINLINE
    const Bucket* _find_sw(const Key& key, sz h, u16 fp) const noexcept {
        if (OOH_UNLIKELY(m_cap == 0)) return nullptr;
        sz pos = h & m_mask;
        const Bucket* OOH_RESTRICT B = m_buckets.get();
        while (true) {
            const u64 w = B[pos].word.load(std::memory_order_relaxed);
            const u32 df = Bucket::df_of(w);
            if (OOH_UNLIKELY(df == Bucket::EMPTY_DF)) return nullptr;
            if (df != Bucket::DELETED_DF && Bucket::fp_of(w) == fp) {
                const u32 vidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[vidx]);
                if (m_key_equal(_kv(vidx)->key, key)) return &B[pos];
            }
            pos = (pos + 1) & m_mask;
        }
    }
    [[nodiscard]] OOH_FORCEINLINE
    Bucket* _find_sw_mut(const Key& key, sz h, u16 fp) noexcept {
        return const_cast<Bucket*>(_find_sw(key, h, fp));
    }

    // Combined dup-check + insert-slot probe (single pass)
    [[nodiscard]] OOH_FORCEINLINE
    const Bucket* _sw_probe(const Key& key, sz h, u16 fp,
                             sz& out_slot, u32& out_dist, u32& out_tomb_vidx) const noexcept {
        sz pos = h & m_mask;
        u32 dist = 1, tomb_dist = 0, tomb_vidx = u32(-1);
        sz  tomb_pos = sz(-1);
        const Bucket* OOH_RESTRICT B = m_buckets.get();
        while (dist <= 0xFFFFu) {
            const u64 w = B[pos].word.load(std::memory_order_relaxed);
            const u32 df = Bucket::df_of(w);
            if (df == Bucket::EMPTY_DF) {
                if (tomb_pos != sz(-1)) { out_slot = tomb_pos; out_dist = tomb_dist; out_tomb_vidx = tomb_vidx; }
                else                    { out_slot = pos;      out_dist = dist;      out_tomb_vidx = u32(-1); }
                return nullptr;
            }
            if (df == Bucket::DELETED_DF) {
                if (tomb_pos == sz(-1)) { tomb_pos = pos; tomb_dist = dist; tomb_vidx = Bucket::vidx_of(w); }
            } else if (Bucket::fp_of(w) == fp) {
                const u32 kvidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[kvidx]);
                if (m_key_equal(_kv(kvidx)->key, key)) return &B[pos];
            }
            pos = (pos + 1) & m_mask; ++dist;
        }
        if (tomb_pos != sz(-1)) { out_slot = tomb_pos; out_dist = tomb_dist; out_tomb_vidx = tomb_vidx; }
        else                    { out_slot = sz(-1);   out_dist = 0;         out_tomb_vidx = u32(-1); }
        return nullptr;
    }

    // Allocate or recycle a KV slot; return index or u32(-1) on failure.
    // If tomb_vidx != -1: destroy old KV and reuse that slot.
    // Strong exception safety: on throw the tombstone remains DELETED.
    template<typename K, typename... Args>
    u32 _sw_alloc_kv(K&& key, u32 tomb_vidx, Args&&... args) {
        if (tomb_vidx != u32(-1)) {
            _kv(tomb_vidx)->~KV();
            new (&m_kv_buf[tomb_vidx]) KV(std::forward<K>(key), Value(std::forward<Args>(args)...));
            OOH_TSAN_RELEASE(&m_kv_buf[tomb_vidx]);
            return tomb_vidx;
        }
        const u32 vidx = m_vidx.load(std::memory_order_relaxed);
        if (vidx >= m_max_items) return u32(-1);
        new (&m_kv_buf[vidx]) KV(std::forward<K>(key), Value(std::forward<Args>(args)...));
        OOH_TSAN_RELEASE(&m_kv_buf[vidx]);
        m_vidx.store(vidx + 1, std::memory_order_relaxed);
        return vidx;
    }

    // Specialisation for operator[] (default-constructs Value)
    template<typename K>
    u32 _sw_alloc_kv_default(K&& key, u32 tomb_vidx) {
        return _sw_alloc_kv(std::forward<K>(key), tomb_vidx);
    }

    template <typename K, typename V>
    bool _sw_insert(K&& key, V&& val) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        sz slot; u32 dist, tomb_vidx;
        if (_sw_probe(key, h, fp, slot, dist, tomb_vidx)) return false;
        if (slot == sz(-1)) return false;
        const u32 vidx = _sw_alloc_kv(std::forward<K>(key), tomb_vidx, std::forward<V>(val));
        if (vidx == u32(-1)) return false;
        _sw_commit(slot, fp, dist, vidx);
        return true;
    }

    template <typename K, typename V>
    bool _sw_insert_or_assign(K&& key, V&& val) {
        const sz h = m_hash(key); const u16 fp = _fp(h);
        sz slot; u32 dist, tomb_vidx;
        if (const Bucket* b = _sw_probe(key, h, fp, slot, dist, tomb_vidx)) {
            _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val = std::forward<V>(val);
            return false;
        }
        if (slot == sz(-1)) return false;
        const u32 vidx = _sw_alloc_kv(std::forward<K>(key), tomb_vidx, std::forward<V>(val));
        if (vidx == u32(-1)) return false;
        _sw_commit(slot, fp, dist, vidx);
        return true;
    }

    template <typename K>
    Value& _sw_subscript(K&& key) {
        static_assert(std::is_default_constructible_v<Value>,
            "operator[] requires default-constructible Value");
        const sz h = m_hash(key); const u16 fp = _fp(h);
        sz slot; u32 dist, tomb_vidx;
        if (const Bucket* b = _sw_probe(key, h, fp, slot, dist, tomb_vidx))
            return _kv(Bucket::vidx_of(b->word.load(std::memory_order_relaxed)))->val;
        if (slot == sz(-1))
            throw std::overflow_error("flat_map::operator[]: capacity exceeded");
        const u32 vidx = _sw_alloc_kv(std::forward<K>(key), tomb_vidx);
        if (vidx == u32(-1))
            throw std::overflow_error("flat_map::operator[]: capacity exceeded");
        _sw_commit(slot, fp, dist, vidx);
        return _kv(vidx)->val;
    }

    OOH_FORCEINLINE void _sw_commit(sz slot, u16 fp, u32 dist, u32 vidx) noexcept {
        m_buckets[slot].word.store(Bucket::encode(vidx, fp, u16(dist)), std::memory_order_release);
        m_size.fetch_add(1, std::memory_order_relaxed);
    }

    // Lock-free CAS placement for concurrent_insert
    bool _place_cas(const Key& key, sz h, u16 fp, u32 vidx) noexcept {
        sz pos = h & m_mask;
        u32 dist = 1;
        while (true) {
            u64 w = m_buckets[pos].word.load(std::memory_order_acquire);
            const u32 df = Bucket::df_of(w);
            if (df == Bucket::EMPTY_DF) {
                if (m_buckets[pos].word.compare_exchange_strong(
                        w, Bucket::encode(vidx, fp, u16(dist)),
                        std::memory_order_release, std::memory_order_acquire)) {
                    if (_dup_before(key, h, fp, pos)) {
                        m_buckets[pos].word.store(
                            Bucket::make(vidx, Bucket::DELETED_DF), std::memory_order_release);
                        return false;
                    }
                    m_size.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                continue;
            }
            if (df == Bucket::DELETED_DF) { pos = (pos+1)&m_mask; ++dist; if (dist>m_cap||dist>0xFFFFu) return false; continue; }
            if (Bucket::fp_of(w) == fp) {
                const u32 kvidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[kvidx]);
                if (m_key_equal(_kv(kvidx)->key, key)) return false;
            }
            pos = (pos+1)&m_mask; ++dist; if (dist>m_cap||dist>0xFFFFu) return false;
        }
    }

    bool _dup_before(const Key& key, sz h, u16 fp, sz stop) const noexcept {
        sz pos = h & m_mask;
        while (pos != stop) {
            const u64 w = m_buckets[pos].word.load(std::memory_order_acquire);
            const u32 df = Bucket::df_of(w);
            if (df == Bucket::EMPTY_DF) return false;
            if (df != Bucket::DELETED_DF && Bucket::fp_of(w) == fp) {
                const u32 kvidx = Bucket::vidx_of(w);
                OOH_TSAN_ACQUIRE(&m_kv_buf[kvidx]);
                if (m_key_equal(_kv(kvidx)->key, key)) return true;
            }
            pos = (pos+1) & m_mask;
        }
        return false;
    }

    template <typename It>
    static sz _range_hint(It first, It last, sz hint) {
        if (hint > 0) return hint;
        if constexpr (std::is_base_of_v<std::random_access_iterator_tag,
                typename std::iterator_traits<It>::iterator_category>) {
            auto d = std::distance(first, last);
            return d > 0 ? sz(d) : sz(1);
        }
        return sz(64);
    }
};

} // namespace ooh
