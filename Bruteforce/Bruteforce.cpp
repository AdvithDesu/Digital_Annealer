// Bruteforce.cpp
// Unbounded outward-spiral divisor search for semiprime N = P*Q.
// Sized for N up to 128 bits with P and Q up to 64 bits each.
//
// Strategy:
//   - Compute centroid = avg(P_pred, N/Q_pred) as best starting estimate.
//   - Threads claim chunks via an atomic index, alternating above/below
//     the centroid (even index -> up, odd index -> down), so candidates
//     closest to the prediction are always tested first.
//   - Wheel sieve mod 2310 skips ~79% of candidates with no divisibility test.
//   - Runs until the factor is found; no window or delta needed.
//
// Build:  g++ -O3 -std=c++17 -pthread -o Bruteforce Bruteforce.cpp
// Usage:  Bruteforce  N P_pred Q_pred [threads]
//         Bruteforce  -pq P Q P_pred Q_pred [threads]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>

using u64  = uint64_t;
using u128 = __uint128_t;

// ---------- u128 helpers ----------
static u128 parse_u128(const char* s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (u128)(*s - '0'); ++s; }
    return v;
}
static void print_u128(char* buf, u128 v) {
    if (v == 0) { buf[0]='0'; buf[1]=0; return; }
    char tmp[40]; int n=0;
    while (v > 0) { tmp[n++]=(char)('0'+(int)(v%10)); v/=10; }
    for (int i=0;i<n;++i) buf[i]=tmp[n-1-i]; buf[n]=0;
}
static inline u64 sat_u64(u128 v) {
    return (v > (u128)UINT64_MAX) ? UINT64_MAX : (u64)v;
}

// ---------- Wheel sieve mod 2310 ----------
static constexpr u64 W = 2310;
static std::vector<u64> wheel_residues;

static void build_wheel() {
    if (!wheel_residues.empty()) return;
    static const u64 wp[] = {2,3,5,7,11};
    wheel_residues.reserve(480);
    for (u64 r=1; r<W; r+=2) {
        bool ok=true;
        for (u64 p:wp) if (r%p==0){ok=false;break;}
        if (ok) wheel_residues.push_back(r);
    }
}

// ---------- Single-chunk search ----------
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
        if (next < base) break;  // u64 overflow
        base = next;
    }
    return 0;
}

int main(int argc, char** argv) {
    u128 N; u64 P_pred, Q_pred; unsigned T;

    bool pq_mode = (argc > 1 && strcmp(argv[1], "-pq") == 0);

    if (pq_mode) {
        if (argc < 6) {
            fprintf(stderr,
                "Usage: %s -pq P Q P_pred Q_pred [threads]\n", argv[0]);
            return 1;
        }
        u64 a = strtoull(argv[2], nullptr, 10);
        u64 b = strtoull(argv[3], nullptr, 10);
        N = (u128)a * (u128)b;
        u64 pp = strtoull(argv[4], nullptr, 10);
        u64 qp = strtoull(argv[5], nullptr, 10);
        P_pred = (pp <= qp) ? pp : qp;
        Q_pred = (pp <= qp) ? qp : pp;
        T = (argc > 6) ? (unsigned)atoi(argv[6])
                       : std::max(1u, std::thread::hardware_concurrency());
        char nbuf[48]; print_u128(nbuf, N);
        fprintf(stderr, "[pq-mode] N = %s\n", nbuf);
    } else {
        if (argc < 4) {
            fprintf(stderr,
                "Usage: %s N P_pred Q_pred [threads]\n"
                "       %s -pq P Q P_pred Q_pred [threads]\n", argv[0], argv[0]);
            return 1;
        }
        N      = parse_u128(argv[1]);
        P_pred = strtoull(argv[2], nullptr, 10);
        Q_pred = strtoull(argv[3], nullptr, 10);
        T = (argc > 4) ? (unsigned)atoi(argv[4])
                       : std::max(1u, std::thread::hardware_concurrency());
    }

    auto t_total_start = std::chrono::steady_clock::now();
    build_wheel();

    // Centroid: average of P_pred and N/Q_pred (two independent estimates of P)
    u64 p_from_q = sat_u64(N / (u128)Q_pred);
    u64 center   = (u64)(((u128)P_pred + (u128)p_from_q) / 2);
    if (center < 2) center = 2;

    fprintf(stderr, "[search] center=%llu  threads=%u\n",
            (unsigned long long)center, T);

    // Spiral outward from center. Chunk k:
    //   even k -> [center + (k/2)*C,     center + (k/2+1)*C - 1]  (upward)
    //   odd  k -> [center - (k+1)/2*C,   center - (k-1)/2*C - 1]  (downward)
    constexpr u64 CHUNK = 1u << 14;
    std::atomic<u64> chunk_idx{0};
    std::atomic<bool> stop{false};
    std::atomic<u64>  found_p{0};

    auto worker = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            u64 k  = chunk_idx.fetch_add(1, std::memory_order_relaxed);
            u64 lo, hi;

            if (k % 2 == 0) {
                // Upward chunk
                u64 step = k / 2;
                u64 offset = step * CHUNK;
                if (step > 0 && offset / CHUNK != step) continue;  // overflow
                lo = center + offset;
                if (lo < center) continue;                          // overflow
                hi = (lo + CHUNK - 1 < lo) ? UINT64_MAX : lo + CHUNK - 1;
            } else {
                // Downward chunk
                u64 step   = (k + 1) / 2;
                u64 offset = step * CHUNK;
                if (step > 0 && offset / CHUNK != step) continue;  // overflow
                if (center < offset) continue;                      // below 0
                lo = center - offset;
                hi = center - (step - 1) * CHUNK - 1;
                lo = std::max<u64>(lo, 2);
                if (lo > hi) continue;
            }

            u64 p = search_chunk(N, lo, hi, stop);
            if (p) {
                u64 expected = 0;
                if (found_p.compare_exchange_strong(expected, p))
                    stop.store(true, std::memory_order_relaxed);
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
        u64  q = (u64)(N / (u128)p);
        u128 product = (u128)p * (u128)q;
        char nbuf[48], pbuf[48];
        print_u128(nbuf, N); print_u128(pbuf, product);
        printf("P = %llu\n",          (unsigned long long)p);
        printf("Q = %llu\n",          (unsigned long long)q);
        printf("P*Q = %s\n",          pbuf);
        printf("N   = %s\n",          nbuf);
        printf("search time = %.6f s\n", search_secs);
        printf("total time  = %.6f s\n", total_secs);
        return (product == N) ? 0 : 4;
    }

    fprintf(stderr, "[fail] Factor not found (exhausted u64 range).\n"
        "search time = %.6f s\ntotal time  = %.6f s\n", search_secs, total_secs);
    return 3;
}
