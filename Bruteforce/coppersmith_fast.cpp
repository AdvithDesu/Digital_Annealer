// coppersmith_fast.cpp
// Dependency-free hand-rolled Coppersmith factor recovery.
//
// Same algorithm as coppersmith.cpp (Coppersmith "factoring with known high
// bits" in an outward-spiralling grid around a guess for P), but with a
// self-contained, fixed-precision LLL instead of NTL. Links nothing but the
// C++ standard library: uses __int128 (compiler built-in) for N and the
// divisibility tests, a fixed-width BigInt for the lattice entries, and
// long double for the Gram-Schmidt arithmetic.
//
// Why long double is enough for the GSO: the Gram-Schmidt mu coefficients are
// bounded; only the magnitudes of the b*_i norms are large, and those are
// carried by the floating-point exponent, not the mantissa. The precision
// needed scales with the lattice DIMENSION, not the entry bit-length (the L2
// result). NTL's LLL_XD does this with a 53-bit mantissa and works; on the
// GH200 long double is IEEE binary128 (113-bit mantissa) -- strictly more.
// Every candidate is verified with an exact N % cand, so any residual float
// error can only cause a false negative (a missed block), never a false factor.
//
// Build (Linux / GH200):
//   g++ -O3 -std=c++17 -mcpu=neoverse-v2 -pthread coppersmith_fast.cpp -o coppersmith_fast
//   (x86 dev box: drop -mcpu, or use -march=native)
//
// Usage:
//   coppersmith_fast  N  guessP        [threads] [m] [t] [safety]
//   coppersmith_fast  -pq P Q guessP   [threads] [m] [t] [safety]   (N = P*Q)
//   coppersmith_fast  --selftest [bits=64] [threads]
//
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <complex>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <random>
#include <memory>

using std::max;
using std::min;
using u64  = uint64_t;
using u128 = unsigned __int128;
using i128 = __int128;

// ============================ fixed-width BigInt ============================
// Sign-magnitude. d[] little-endian, only d[0..len-1] are meaningful; limbs at
// or above len are never read, so we never have to clear them. LIMBS is sized
// for the largest intermediate during lattice build and reduction:
// entries are <= N^m (<= 2m limbs for 128-bit N), build products reach ~2.5m
// limbs, so LIMBS=28 covers m up to ~10 with margin.
static constexpr int LIMBS = 40;

struct BigInt {
    int sign;            // -1, 0, +1
    int len;             // number of significant limbs
    u64 d[LIMBS];

    void setzero() { sign = 0; len = 0; }
    void set_one() { sign = 1; len = 1; d[0] = 1; }

    void from_u64(u64 v) {
        if (v == 0) { setzero(); return; }
        sign = 1; len = 1; d[0] = v;
    }
    void from_u128(u128 v) {
        if (v == 0) { setzero(); return; }
        sign = 1;
        d[0] = (u64)v; d[1] = (u64)(v >> 64);
        len = d[1] ? 2 : 1;
    }
    void from_i128(i128 v) {
        if (v == 0) { setzero(); return; }
        if (v < 0) { from_u128((u128)(-v)); sign = -1; }
        else       { from_u128((u128)v);    sign = 1;  }
    }
    // x assumed integer-valued (caller rounds). Decomposes into limbs.
    void from_longdouble(long double x) {
        setzero();
        int s = (x < 0) ? -1 : 1;
        long double t = fabsl(x);
        int i = 0;
        while (t >= 1.0L && i < LIMBS) {
            long double q   = floorl(ldexpl(t, -64));  // floor(t / 2^64)
            long double rem = t - ldexpl(q, 64);       // low 64 bits
            d[i++] = (u64)rem;
            t = q;
        }
        len = i;
        while (len > 0 && d[len - 1] == 0) len--;
        sign = (len == 0) ? 0 : s;
    }
};

static long double to_ld(const BigInt& x) {
    if (x.sign == 0) return 0.0L;
    int lo = (x.len > 3) ? x.len - 3 : 0;
    long double r = 0;
    for (int i = x.len - 1; i >= lo; i--)
        r = r * 18446744073709551616.0L + (long double)x.d[i];   // *2^64
    r = ldexpl(r, 64 * lo);
    return (x.sign < 0) ? -r : r;
}

static int cmp_mag(const BigInt& a, const BigInt& b) {
    if (a.len != b.len) return a.len < b.len ? -1 : 1;
    for (int i = a.len - 1; i >= 0; i--)
        if (a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
    return 0;
}

// res = |a| + |b|   (alias-safe: reads index i before writing it)
static void add_mag(const BigInt& a, const BigInt& b, BigInt& res) {
    int n = max(a.len, b.len);
    u128 carry = 0;
    for (int i = 0; i < n; i++) {
        u128 s = carry + (i < a.len ? a.d[i] : 0) + (i < b.len ? b.d[i] : 0);
        res.d[i] = (u64)s;
        carry = s >> 64;
    }
    if (carry) { res.d[n] = (u64)carry; n++; }
    res.len = n;
}

// res = |a| - |b|, requires |a| >= |b|  (alias-safe)
static void sub_mag(const BigInt& a, const BigInt& b, BigInt& res) {
    i128 borrow = 0;
    int n = a.len;
    for (int i = 0; i < n; i++) {
        i128 s = (i128)a.d[i] - (i < b.len ? (i128)b.d[i] : 0) - borrow;
        if (s < 0) { s += ((i128)1 << 64); borrow = 1; }
        else borrow = 0;
        res.d[i] = (u64)s;
    }
    while (n > 0 && res.d[n - 1] == 0) n--;
    res.len = n;
}

static void add(const BigInt& a, const BigInt& b, BigInt& res);  // fwd

// res = a + b
static void add(const BigInt& a, const BigInt& b, BigInt& res) {
    if (a.sign == 0) { res = b; return; }
    if (b.sign == 0) { res = a; return; }
    if (a.sign == b.sign) { add_mag(a, b, res); res.sign = a.sign; return; }
    int c = cmp_mag(a, b);
    if (c == 0) { res.setzero(); return; }
    if (c > 0) { sub_mag(a, b, res); res.sign = a.sign; }
    else       { sub_mag(b, a, res); res.sign = b.sign; }
}

// res = a - b
static void sub(const BigInt& a, const BigInt& b, BigInt& res) {
    BigInt nb = b; nb.sign = -nb.sign;   // copy so b is untouched even if res==b
    add(a, nb, res);
}

// res = a * b   (res must NOT alias a or b)
static void mul(const BigInt& a, const BigInt& b, BigInt& res) {
    if (a.sign == 0 || b.sign == 0) { res.setzero(); return; }
    if (a.len + b.len > LIMBS) { res.setzero(); return; }  // safety: must not happen
    int n = a.len + b.len;
    for (int i = 0; i < n; i++) res.d[i] = 0;
    for (int i = 0; i < a.len; i++) {
        u128 carry = 0;
        for (int j = 0; j < b.len; j++) {
            u128 p = (u128)a.d[i] * b.d[j] + res.d[i + j] + carry;
            res.d[i + j] = (u64)p;
            carry = p >> 64;
        }
        res.d[i + b.len] += (u64)carry;
    }
    while (n > 0 && res.d[n - 1] == 0) n--;
    res.len = n;
    res.sign = (n == 0) ? 0 : a.sign * b.sign;
}

// =============================== parameters ===============================
struct Params { int m, t, d, deg; double beta; };

static int    g_D;          // lattice dimension (= Params.d), for MAXD checks
static constexpr int MAXD = 18;

// =============================== LLL engine ===============================
// Floating-point LLL: exact-integer basis b[][], long double GSO (mu, Bnorm).
struct Lattice {
    int D;
    BigInt b[MAXD][MAXD];        // basis rows x coords
    BigInt fp[MAXD][MAXD];       // workspace: coeffs of (a+x)^i
    long double mu[MAXD][MAXD];
    long double Bnorm[MAXD];
    long double ldv[MAXD][MAXD]; // long double cache of b for GSO
    BigInt t1, t2, prod, R;      // scratch

    // --- build the Howgrave-Graham basis for f(x)=a+x around centre 'a' ---
    void build(i128 a, const Params& P,
               const std::vector<BigInt>& Npow, const std::vector<BigInt>& Xpow) {
        D = P.d;
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) b[i][j].setzero();

        BigInt A; A.from_i128(a);

        // fp[i][*] = coefficients (ascending) of (a + x)^i
        fp[0][0].set_one();
        for (int i = 1; i <= P.m; i++) {
            for (int j = 0; j <= i; j++) {
                if (j <= i - 1) mul(fp[i - 1][j], A, t1); else t1.setzero();
                if (j >= 1)     add(t1, fp[i - 1][j - 1], fp[i][j]);
                else            fp[i][j] = t1;
            }
        }
        // g_i rows:  N^(m-i) * f(x)^i , column j scaled by X^j
        for (int i = 0; i <= P.m; i++)
            for (int j = 0; j <= i; j++) {
                mul(fp[i][j], Npow[P.m - i], t1);
                mul(t1, Xpow[j], b[i][j]);
            }
        // h_k rows:  x^k * f(x)^m , column (c+k) scaled by X^(c+k)
        for (int k = 1; k <= P.t; k++) {
            int row = P.m + k;
            for (int c = 0; c <= P.m; c++)
                mul(fp[P.m][c], Xpow[c + k], b[row][c + k]);
        }
    }

    // Initial GSO from the LOWER-TRIANGULAR build: for a triangular basis the
    // Gram-Schmidt is exact in closed form, b*_i = d_i e_i, so
    //   Bnorm[i] = d_i^2          (d_i = diagonal entry b[i][i])
    //   mu[i][j] = b[i][j] / d_j  (j < i)
    // This avoids the catastrophic cancellation of the naive recurrence, which
    // is fatal here because b*_i can be hundreds of bits smaller than b_i.
    // (Only valid right after build(), before any reduction; lll() calls it first.)
    void gso() {
        for (int j = 0; j < D; j++) {
            long double dj = to_ld(b[j][j]);
            Bnorm[j] = dj * dj;
            if (Bnorm[j] < 1e-300L) Bnorm[j] = 1e-300L;
        }
        for (int i = 0; i < D; i++)
            for (int j = 0; j < i; j++)
                mu[i][j] = (b[i][j].sign == 0) ? 0.0L
                                               : to_ld(b[i][j]) / to_ld(b[j][j]);
    }

    // size-reduce b_k against b_l (l < k)
    void reduce(int k, int l) {
        if (fabsl(mu[k][l]) <= 0.5L) return;
        long double r = roundl(mu[k][l]);
        R.from_longdouble(r);
        if (R.sign != 0) {
            for (int c = 0; c < D; c++) {
                mul(R, b[l][c], prod);
                sub(b[k][c], prod, b[k][c]);
            }
        }
        for (int j = 0; j < l; j++) mu[k][j] -= r * mu[l][j];
        mu[k][l] -= r;
    }

    // swap b_k and b_{k-1}, update GSO in place (standard formulas)
    void swap_rows(int k) {
        for (int c = 0; c < D; c++) std::swap(b[k][c], b[k - 1][c]);
        long double nu = mu[k][k - 1];
        long double Bp = Bnorm[k] + nu * nu * Bnorm[k - 1];
        if (Bp < 1e-300L) Bp = 1e-300L;
        mu[k][k - 1] = nu * Bnorm[k - 1] / Bp;
        Bnorm[k]     = Bnorm[k - 1] * Bnorm[k] / Bp;
        Bnorm[k - 1] = Bp;
        for (int j = 0; j < k - 1; j++) std::swap(mu[k - 1][j], mu[k][j]);
        for (int i = k + 1; i < D; i++) {
            long double tt = mu[i][k];
            mu[i][k]     = mu[i][k - 1] - nu * tt;
            mu[i][k - 1] = tt + mu[k][k - 1] * mu[i][k];
        }
    }

    void lll(double delta = 0.99) {
        gso();
        int k = 1;
        long long iter = 0, cap = 200LL * D * D + 1000;
        while (k < D) {
            if (++iter > cap) break;     // safety against float-induced stalls
            reduce(k, k - 1);
            if (Bnorm[k] >= (delta - mu[k][k - 1] * mu[k][k - 1]) * Bnorm[k - 1]) {
                for (int l = k - 2; l >= 0; l--) reduce(k, l);
                k++;
            } else {
                swap_rows(k);
                k = (k - 1 > 1) ? k - 1 : 1;
            }
        }
    }
};

// =========================== root finding (DK) ===========================
// Real roots of the SCALED polynomial G(y)=sum coeff[j] y^j (y = x/X). The
// X-scaling balances the coefficients, so this is well-conditioned in double.
static void find_real_roots(const std::vector<double>& coeff, double ybound,
                            std::vector<double>& out) {
    long n = (long)coeff.size() - 1;
    while (n >= 1 && coeff[n] == 0.0) n--;
    if (n < 1) return;

    std::vector<std::complex<double>> c(n + 1);
    for (long j = 0; j <= n; j++) c[j] = std::complex<double>(coeff[j], 0.0);
    std::complex<double> lead = c[n];
    if (std::abs(lead) == 0.0) return;
    for (auto& z : c) z /= lead;

    std::vector<std::complex<double>> r(n);
    std::complex<double> seed(0.4, 0.9), p(1.0, 0.0);
    for (long k = 0; k < n; k++) { p *= seed; r[k] = p; }

    for (int it = 0; it < 300; it++) {
        double maxd = 0.0;
        for (long i = 0; i < n; i++) {
            std::complex<double> val(1.0, 0.0);
            for (long j = n - 1; j >= 0; j--) val = val * r[i] + c[j];
            std::complex<double> den(1.0, 0.0);
            for (long j = 0; j < n; j++) if (j != i) den *= (r[i] - r[j]);
            if (std::abs(den) == 0.0) continue;
            std::complex<double> delta = val / den;
            r[i] -= delta;
            maxd = max(maxd, std::abs(delta));
        }
        if (maxd < 1e-13) break;
    }
    for (long i = 0; i < n; i++) {
        double re = r[i].real(), im = r[i].imag();
        if (fabs(im) < 1e-6 * (1.0 + fabs(re)) && fabs(re) <= ybound)
            out.push_back(re);
    }
}

// =============================== search state ===============================
static std::atomic<bool>               g_found{false};
static std::atomic<unsigned long long> g_next_k{0};
static std::atomic<unsigned long long> g_blocks_done{0};
static u128                            g_factor;
static std::mutex                      g_factor_mtx;

// Test one block centred at 'a'. Returns true and sets outFactor on success.
static bool test_block(Lattice& lat, i128 a, const Params& P, u128 N,
                       const std::vector<BigInt>& Npow,
                       const std::vector<BigInt>& Xpow,
                       double Xdouble, u128& outFactor) {
    lat.build(a, P, Npow, Xpow);
    lat.lll();

    std::vector<double> coeff, roots;
    for (int row = 0; row < P.d; row++) {
        coeff.assign(P.d, 0.0);
        int hi = -1;
        for (int j = 0; j < P.d; j++) {
            if (lat.b[row][j].sign != 0) { coeff[j] = (double)to_ld(lat.b[row][j]); hi = j; }
        }
        if (hi < 1) continue;
        coeff.resize(hi + 1);
        roots.clear();
        find_real_roots(coeff, 1.5, roots);
        for (double y : roots) {
            long base = (long)llround(y * Xdouble);
            for (long off = -4; off <= 4; off++) {
                i128 x0   = (i128)base + off;
                i128 cand = a + x0;
                if (cand <= 1 || (u128)cand >= N) continue;
                if (N % (u128)cand == 0) { outFactor = (u128)cand; return true; }
            }
        }
    }
    return false;
}

// ============================= u128 string I/O =============================
static u128 parse_u128(const std::string& s) {
    u128 v = 0;
    for (char ch : s) if (ch >= '0' && ch <= '9') v = v * 10 + (u128)(ch - '0');
    return v;
}
static std::string u128_str(u128 v) {
    if (v == 0) return "0";
    char buf[44]; int n = 0;
    while (v > 0) { buf[n++] = (char)('0' + (int)(v % 10)); v /= 10; }
    std::string r; for (int i = n - 1; i >= 0; i--) r += buf[i];
    return r;
}

// =============================== driver ===============================
static double x_exponent(const Params& P) {
    double m = P.m, t = P.t, d = P.d;
    double E   = m * (m + 1) / 2.0 + t * m + t * (t + 1) / 2.0;
    double num = P.beta * m * d - m * (m + 1) / 2.0;
    return num / E;
}

static int run_search(u128 N, u128 guess, unsigned T, Params P, double safety) {
    P.d   = P.m + 1 + P.t;
    P.deg = P.m + P.t;
    if (P.d > MAXD) { fprintf(stderr, "[error] dim %d > MAXD %d; lower m/t.\n", P.d, MAXD); return 2; }

    // Quick win.
    if (guess > 1 && guess < N && N % guess == 0) {
        u128 q = N / guess;
        std::cout << "P = " << u128_str(min(guess, q)) << "\nQ = " << u128_str(max(guess, q)) << "\n";
        std::cout << "(guess was exact)\n";
        return 0;
    }

    double log2N = log2((double)N);
    double Xexp  = x_exponent(P);
    double Xbits = Xexp * log2N;
    if (Xbits < 1.0) { fprintf(stderr, "[error] X<2 (Xbits=%.2f); raise m/t.\n", Xbits); return 2; }

    u64 Xscale = (u64)1 << (int)floor(Xbits);
    u64 step   = Xscale / (u64)llround(safety);
    if (step < 1) step = 1;

    // Precompute N^e and X^e as BigInt (shared, read-only).
    std::vector<BigInt> Npow(P.m + 1), Xpow(P.deg + 1);
    BigInt Nb;  Nb.from_u128(N);
    BigInt Xb;  Xb.from_u64(Xscale);
    Npow[0].set_one(); for (int e = 1; e <= P.m;   e++) mul(Npow[e - 1], Nb, Npow[e]);
    Xpow[0].set_one(); for (int e = 1; e <= P.deg; e++) mul(Xpow[e - 1], Xb, Xpow[e]);

    double Xdouble = (double)Xscale;
    i128   center  = (i128)guess;

    fprintf(stderr,
        "[config] m=%d t=%d dim=%d  beta=%.3f  log2(N)=%.1f\n"
        "         X ~ 2^%.1f   step ~ 2^%.1f   threads=%u  safety=%.1f\n"
        "         (unbounded grid: runs until P is found)\n",
        P.m, P.t, P.d, P.beta, log2N, Xbits, log2((double)step), T, safety);

    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        auto lat = std::make_unique<Lattice>();
        u128 factor;
        while (!g_found.load(std::memory_order_relaxed)) {
            unsigned long long k = g_next_k.fetch_add(1, std::memory_order_relaxed);
            i128 a;
            if (k % 2 == 0) a = center + (i128)((u128)(k / 2) * step);
            else            a = center - (i128)((u128)((k + 1) / 2) * step);
            if (a <= 2 || (u128)a >= N) { g_blocks_done.fetch_add(1, std::memory_order_relaxed); continue; }

            if (test_block(*lat, a, P, N, Npow, Xpow, Xdouble, factor)) {
                std::lock_guard<std::mutex> lk(g_factor_mtx);
                if (!g_found.load()) { g_factor = factor; g_found.store(true); }
                return;
            }
            g_blocks_done.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto monitor = [&]() {
        unsigned long long last = 0;
        while (!g_found.load()) {
            for (int s = 0; s < 25 && !g_found.load(); s++)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (g_found.load()) break;
            unsigned long long done = g_blocks_done.load();
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double rate = (done - last) / 5.0; last = done;
            unsigned long long k = g_next_k.load();
            double radius = (double)(k / 2 + 1) * (double)step;
            fprintf(stderr, "[search] blocks=%llu  rate=%.0f blk/s  radius~2^%.1f  elapsed=%.0fs\n",
                    done, rate, log2(radius + 1.0), secs);
        }
    };

    std::thread mon(monitor);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < T; i++) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    g_found.store(true);
    mon.join();

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    u128 factor;
    { std::lock_guard<std::mutex> lk(g_factor_mtx); factor = g_factor; }
    if (factor > 1 && factor < N && N % factor == 0) {
        u128 q = N / factor;
        std::cout << "P = " << u128_str(min(factor, q)) << "\nQ = " << u128_str(max(factor, q)) << "\n";
        fprintf(stderr, "[done] found in %.3f s  (%llu blocks)\n", secs, g_blocks_done.load());
        return 0;
    }
    fprintf(stderr, "[fail] exhausted %llu blocks.\n", g_blocks_done.load());
    return 3;
}

// =============================== selftest ===============================
static u64 mulmod(u64 a, u64 b, u64 m) { return (u128)a * b % m; }
static u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1; a %= m;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
static bool is_prime(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2,3,5,7,11,13,17,19,23,29,31,37}) { if (n % p == 0) return n == p; }
    u64 d = n - 1; int s = 0; while (!(d & 1)) { d >>= 1; s++; }
    for (u64 a : {2,3,5,7,11,13,17,19,23,29,31,37}) {
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool ok = false;
        for (int i = 1; i < s; i++) { x = mulmod(x, x, n); if (x == n - 1) { ok = true; break; } }
        if (!ok) return false;
    }
    return true;
}
static u64 rand_prime(int bits, std::mt19937_64& rng) {
    u64 lo = (u64)1 << (bits - 1);
    u64 hi = (bits >= 64) ? ~0ULL : (((u64)1 << bits) - 1);
    while (true) {
        u64 v = lo + (rng() % (hi - lo));
        v |= 1;
        if (is_prime(v)) return v;
    }
}

static int selftest(int bits, unsigned T) {
    std::mt19937_64 rng(0xC0FFEE);
    u64 p = rand_prime(bits, rng);
    u64 q = rand_prime(bits, rng);
    u128 N = (u128)p * q;

    int sbits = max(4, bits / 2 - 12);
    u64 delta = (rng() % ((u64)1 << sbits)) + 1;
    u128 guess = (rng() & 1) ? (u128)p + delta : (u128)p - delta;

    fprintf(stderr, "[selftest] bits=%d  (small perturbation: correctness check)\n", bits);
    std::cerr << "  p     = " << p << "\n  q     = " << q
              << "\n  N     = " << u128_str(N)
              << "\n  guess = " << u128_str(guess) << "  (|p-guess| = " << delta << ")\n";

    Params P{1, 2, 0, 0, 0.49};
    int rc = run_search(N, guess, T, P, 2.0);
    if (rc == 0) fprintf(stderr, "[selftest] PASS (recovery validated)\n");
    return rc;
}

// ================================= debug =================================
// Run a single block at a = centre and dump the reduced basis / roots.
static int debug1(u128 N, i128 centre, Params P, long x0true) {
    P.d = P.m + 1 + P.t; P.deg = P.m + P.t;
    double log2N = log2((double)N);
    double Xbits = x_exponent(P) * log2N;
    u64 Xscale = (u64)1 << (int)floor(Xbits);
    std::vector<BigInt> Npow(P.m + 1), Xpow(P.deg + 1);
    BigInt Nb; Nb.from_u128(N); BigInt Xb; Xb.from_u64(Xscale);
    Npow[0].set_one(); for (int e = 1; e <= P.m;   e++) mul(Npow[e-1], Nb, Npow[e]);
    Xpow[0].set_one(); for (int e = 1; e <= P.deg; e++) mul(Xpow[e-1], Xb, Xpow[e]);

    auto lat = std::make_unique<Lattice>();
    double Xdouble = (double)Xscale;
    double y0 = (double)x0true / Xdouble;
    fprintf(stderr, "[debug1] a=%s  Xscale=2^%d  dim=%d  x0true=%ld  y0=%.8g\n",
            u128_str((u128)centre).c_str(), (int)floor(Xbits), P.d, x0true, y0);
    lat->build(centre, P, Npow, Xpow);
    lat->lll();
    for (int row = 0; row < P.d; row++) {
        std::vector<double> coeff(P.d, 0.0), roots;
        int hi = -1;
        for (int j = 0; j < P.d; j++)
            if (lat->b[row][j].sign != 0) { coeff[j] = (double)to_ld(lat->b[row][j]); hi = j; }
        if (hi < 1) { fprintf(stderr, "  row %d: (skip)\n", row); continue; }
        coeff.resize(hi + 1);
        // Evaluate G(y0) RELATIVE to the largest term, to see if y0 is a true root.
        long double Gy = 0, scale = 0, yp = 1;
        for (int j = 0; j <= hi; j++) { long double term = (long double)coeff[j] * yp; Gy += term; scale = max(scale, fabsl(term)); yp *= y0; }
        find_real_roots(coeff, 1.5, roots);
        fprintf(stderr, "  row %d: deg=%d  c0_bits=%.0f  lead_bits=%.0f  |G(y0)|/scale=%.2e  nroots=%zu",
                row, hi, log2(fabs(coeff[0]) + 1), log2(fabs(coeff[hi]) + 1),
                scale > 0 ? (double)(fabsl(Gy) / scale) : 0.0, roots.size());
        for (double y : roots) fprintf(stderr, "  y=%.6g", y);
        fprintf(stderr, "\n");
    }
    return 0;
}

// ================================= main =================================
int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc >= 4 && std::string(argv[1]) == "--debug1") {
        u128 N = parse_u128(argv[2]); u128 guess = parse_u128(argv[3]);
        long x0true = (argc > 4) ? atol(argv[4]) : 0;
        Params P{4,4,0,0,0.49};
        return debug1(N, (i128)guess, P, x0true);
    }
    if (argc >= 2 && std::string(argv[1]) == "--selftest") {
        int bits = (argc >= 3) ? atoi(argv[2]) : 64;
        unsigned T = (argc >= 4) ? (unsigned)atoi(argv[3]) : max(1u, std::thread::hardware_concurrency());
        return selftest(bits, T);
    }

    bool pq = (argc >= 2 && std::string(argv[1]) == "-pq");
    if ((pq && argc < 5) || (!pq && argc < 3)) {
        fprintf(stderr,
            "Usage: %s N guessP [threads] [m] [t] [safety]\n"
            "       %s -pq P Q guessP [threads] [m] [t] [safety]   (N = P*Q)\n"
            "       %s --selftest [bits=64] [threads]\n", argv[0], argv[0], argv[0]);
        return 1;
    }

    u128 N, guess;
    int base;
    if (pq) {
        u128 p = parse_u128(argv[2]), q = parse_u128(argv[3]);
        guess = parse_u128(argv[4]);
        N = p * q; base = 5;
        fprintf(stderr, "[pq-mode] N = %s\n", u128_str(N).c_str());
    } else {
        N = parse_u128(argv[1]); guess = parse_u128(argv[2]); base = 3;
    }
    if (N <= 3) { fprintf(stderr, "[error] N too small.\n"); return 1; }

    unsigned T = (argc > base) ? (unsigned)atoi(argv[base]) : max(1u, std::thread::hardware_concurrency());
    Params P;
    P.m    = (argc > base + 1) ? atoi(argv[base + 1]) : 1;
    P.t    = (argc > base + 2) ? atoi(argv[base + 2]) : 2;
    P.beta = 0.49;
    double safety = (argc > base + 3) ? atof(argv[base + 3]) : 2.0;
    if (P.m < 1) P.m = 1;
    if (P.t < 1) P.t = 1;
    if (safety < 1.0) safety = 1.0;

    return run_search(N, guess, T, P, safety);
}
