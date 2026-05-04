// Bruteforce.cpp
// Window-restricted divisor search for semiprime N = P*Q.
// Sized for N up to 128 bits with P and Q up to 64 bits each.
//
// Pipeline:
//   (1) intersect P-side and Q-side windows  -> tight search interval
//   (2) wheel sieve mod 2*3*5*7*11 = 2310    -> ~79% of candidates skipped
//   (3) parallel workers, chunks ordered by distance from centroid
//   (4) first divisor wins; for a semiprime N, any divisor in (1, N) is P or Q
//
// The wheel sieve is correct whenever gcd(N, 2*3*5*7*11) == 1, which holds
// automatically when both prime factors exceed 11 — true for every test where
// P and Q are both ~half the bit-length of a non-trivial N.
//
// Build (g++ / MinGW):  g++ -O3 -std=c++17 -pthread -o Bruteforce.exe Bruteforce.cpp
// Usage:                Bruteforce N P_pred Q_pred [delta_p] [delta_q] [threads]
//                       Bruteforce -pq P Q P_pred Q_pred [delta_p] [delta_q] [threads]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstring>

using u64  = uint64_t;
using u128 = __uint128_t;

// ---------- u128 string helpers ----------
static u128 parse_u128(const char* s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (u128)(*s - '0');
        ++s;
    }
    return v;
}

// Decimal of v written into buf (NUL-terminated). buf must hold >= 40 bytes.
static int print_u128(char* buf, u128 v) {
    if (v == 0) { buf[0] = '0'; buf[1] = 0; return 1; }
    char tmp[40];
    int n = 0;
    while (v > 0) {
        tmp[n++] = (char)('0' + (int)(v % 10));
        v /= 10;
    }
    for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
    buf[n] = 0;
    return n;
}

static inline u64 sat_u64(u128 v) {
    return (v > (u128)UINT64_MAX) ? UINT64_MAX : (u64)v;
}

// ---------- Wheel sieve mod W = 2*3*5*7*11 = 2310 ----------
static constexpr u64 W = 2310;
static std::vector<u64> wheel_residues;  // r in [0, W) with gcd(r, W) = 1

static void build_wheel() {
    if (!wheel_residues.empty()) return;
    static const u64 wheel_primes[] = {2, 3, 5, 7, 11};
    wheel_residues.reserve(480);  // phi(2310) = 480
    for (u64 r = 1; r < W; r += 2) {
        bool coprime = true;
        for (u64 p : wheel_primes) {
            if (r % p == 0) { coprime = false; break; }
        }
        if (coprime) wheel_residues.push_back(r);
    }
}

// ---------- Window intersection ----------
struct Window {
    u64 lo;  // inclusive
    u64 hi;  // inclusive (lo > hi means empty)
};

static Window intersect_windows(u128 N, u64 P_pred, u64 Q_pred,
                                u64 delta_p, u64 delta_q) {
    // P-side window
    u64 lo_p = (P_pred > delta_p) ? std::max<u64>(2, P_pred - delta_p) : 2;
    u64 hi_p = (P_pred + delta_p < P_pred) ? UINT64_MAX : (P_pred + delta_p);

    // Q-side window translated to P:  P in [N/(Q_pred + dq), N/(Q_pred - dq)]
    u64 lo_q = (Q_pred > delta_q) ? std::max<u64>(2, Q_pred - delta_q) : 2;
    u64 hi_q = (Q_pred + delta_q < Q_pred) ? UINT64_MAX : (Q_pred + delta_q);
    u64 lo_p_from_q = sat_u64(N / (u128)hi_q);
    u64 hi_p_from_q = sat_u64(N / (u128)lo_q);

    Window w;
    w.lo = std::max<u64>(2, std::max(lo_p, lo_p_from_q));
    w.hi = std::min(hi_p, hi_p_from_q);
    if (w.lo > w.hi) w.hi = (w.lo > 0) ? (w.lo - 1) : 0;  // empty
    return w;
}

// ---------- Per-chunk wheel-sieved divisibility search ----------
static u64 search_chunk(u128 N, u64 lo, u64 hi, std::atomic<bool>& stop) {
    if (lo > hi) return 0;
    u64 base = (lo / W) * W;
    while (base <= hi) {
        if (stop.load(std::memory_order_relaxed)) return 0;
        for (u64 r : wheel_residues) {
            u64 p = base + r;
            if (p < lo) continue;
            if (p > hi) break;
            if (N % (u128)p == 0) return p;
        }
        u64 next = base + W;
        if (next < base) break;  // u64 overflow at top of range
        base = next;
    }
    return 0;
}

int main(int argc, char** argv) {
    u128 N;
    u64  P_pred, Q_pred, delta_p, delta_q;
    unsigned T;

    bool pq_mode = (argc > 1 && strcmp(argv[1], "-pq") == 0);

    if (pq_mode) {
        // Bruteforce -pq P Q P_pred Q_pred [delta_p] [delta_q] [threads]
        if (argc < 6) {
            fprintf(stderr,
                "Usage: %s -pq P Q P_pred Q_pred [delta_p] [delta_q] [threads]\n", argv[0]);
            return 1;
        }
        u64 P_true = strtoull(argv[2], nullptr, 10);
        u64 Q_true = strtoull(argv[3], nullptr, 10);
        N       = (u128)P_true * (u128)Q_true;
        P_pred  = strtoull(argv[4], nullptr, 10);
        Q_pred  = strtoull(argv[5], nullptr, 10);
        delta_p = (argc > 6) ? strtoull(argv[6], nullptr, 10) : 1000;
        delta_q = (argc > 7) ? strtoull(argv[7], nullptr, 10) : 1000;
        T       = (argc > 8) ? (unsigned)atoi(argv[8])
                             : std::max(1u, std::thread::hardware_concurrency());
        char nbuf[48]; print_u128(nbuf, N);
        fprintf(stderr, "[pq-mode] N = %s\n", nbuf);
    } else {
        // Bruteforce N P_pred Q_pred [delta_p] [delta_q] [threads]
        if (argc < 4) {
            fprintf(stderr,
                "Usage: %s N P_pred Q_pred [delta_p] [delta_q] [threads]\n"
                "       %s -pq P Q P_pred Q_pred [delta_p] [delta_q] [threads]\n",
                argv[0], argv[0]);
            return 1;
        }
        N       = parse_u128(argv[1]);
        P_pred  = strtoull(argv[2], nullptr, 10);
        Q_pred  = strtoull(argv[3], nullptr, 10);
        delta_p = (argc > 4) ? strtoull(argv[4], nullptr, 10) : 1000;
        delta_q = (argc > 5) ? strtoull(argv[5], nullptr, 10) : 1000;
        T       = (argc > 6) ? (unsigned)atoi(argv[6])
                             : std::max(1u, std::thread::hardware_concurrency());
    }

    auto t_total_start = std::chrono::steady_clock::now();

    // (1) Window intersection
    Window w = intersect_windows(N, P_pred, Q_pred, delta_p, delta_q);
    if (w.lo > w.hi) {
        fprintf(stderr,
            "[abort] P-window and Q-window do not intersect — predictions "
            "inconsistent with N. Re-anneal.\n");
        return 2;
    }

    fprintf(stderr,
        "[window] P in [%llu, %llu]  size=%llu  threads=%u\n",
        (unsigned long long)w.lo, (unsigned long long)w.hi,
        (unsigned long long)(w.hi - w.lo + 1), T);

    build_wheel();

    // (2) Centroid: combined estimate of P from both hints.
    u64 p_from_q = sat_u64(N / (u128)Q_pred);
    u64 center   = (u64)(((u128)P_pred + (u128)p_from_q) / 2);
    if (center < w.lo) center = w.lo;
    if (center > w.hi) center = w.hi;

    // (3) Build chunks; sort by distance from centroid (closest first).
    constexpr u64 CHUNK = 1u << 14;  // 16384 candidates per chunk
    struct Chunk { u64 lo, hi; };
    std::vector<Chunk> chunks;
    {
        u64 lo = w.lo;
        while (true) {
            u64 hi = (lo > UINT64_MAX - (CHUNK - 1))
                       ? w.hi
                       : std::min<u64>(lo + CHUNK - 1, w.hi);
            chunks.push_back({lo, hi});
            if (hi >= w.hi) break;
            lo = hi + 1;
        }
    }
    auto chunk_dist = [center](Chunk c) -> u64 {
        if (center < c.lo) return c.lo - center;
        if (center > c.hi) return center - c.hi;
        return 0;
    };
    std::sort(chunks.begin(), chunks.end(),
              [&](Chunk a, Chunk b) { return chunk_dist(a) < chunk_dist(b); });

    // (4) Parallel search; first divisor wins.
    std::atomic<size_t> next_chunk{0};
    std::atomic<bool>   stop{false};
    std::atomic<u64>    found_p{0};

    auto worker = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (i >= chunks.size()) return;
            u64 p = search_chunk(N, chunks[i].lo, chunks[i].hi, stop);
            if (p) {
                u64 expected = 0;
                if (found_p.compare_exchange_strong(expected, p)) {
                    stop.store(true, std::memory_order_relaxed);
                }
                return;
            }
        }
    };

    auto t_search_start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (unsigned t = 0; t < T; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();

    auto t_end = std::chrono::steady_clock::now();
    double search_secs = std::chrono::duration<double>(t_end - t_search_start).count();
    double total_secs  = std::chrono::duration<double>(t_end - t_total_start).count();

    if (u64 p = found_p.load()) {
        u64  q       = (u64)(N / (u128)p);
        u128 product = (u128)p * (u128)q;
        char nbuf[48], pbuf[48];
        print_u128(nbuf, N);
        print_u128(pbuf, product);

        printf("P = %llu\n",   (unsigned long long)p);
        printf("Q = %llu\n",   (unsigned long long)q);
        printf("P*Q = %s\n",   pbuf);
        printf("N   = %s\n",   nbuf);
        printf("search time = %.6f s\n", search_secs);
        printf("total time  = %.6f s\n", total_secs);
        return (product == N) ? 0 : 4;
    }

    fprintf(stderr,
        "[fail] No divisor in window. Either delta is too small or the "
        "prediction is outside its claimed error bound.\n"
        "search time = %.6f s\ntotal time  = %.6f s\n",
        search_secs, total_secs);
    return 3;
}
