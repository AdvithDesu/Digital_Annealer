// coppersmith.cpp
// Factor a semiprime N = P*Q given an approximate guess for one factor P,
// using Coppersmith's "factoring with known high bits" (LLL lattice) method
// applied in an outward-spiralling grid around the guess.
//
// Idea
// ----
// Coppersmith recovers an exact factor P of N from an approximation a (= guess)
// PROVIDED |P - a| < ~N^(1/4).  A single guess that is only "within 2%" is far
// too coarse for one shot (2% of a 64-bit P is ~2^58, while N^(1/4) ~ 2^32 for a
// 128-bit N).  So instead we tile the number line with overlapping Coppersmith
// blocks, each of half-width X (~N^(1/4)), and spiral outward from the guess,
// testing the closest blocks first.  Each block is one small LLL reduction.
// We keep searching until P is found -- the 2% figure is NOT assumed anywhere;
// if the real error is larger, the spiral simply runs longer.
//
// Per block:
//   - Build the Howgrave-Graham lattice for f(x) = a + x with multiplicity m
//     and t extra x-shifts (dimension d = m + 1 + t).
//   - LLL-reduce it.
//   - Each reduced row is a polynomial g(x) that vanishes at x0 = P - a over Z.
//     Find g's small real roots, snap to integers, and test gcd/divisibility.
//   - On a hit, store the factor and stop all threads.
//
// Tuning knobs (m, t, safety) trade per-block cost against block count; the
// defaults aim for a few hours on a many-core machine for the 128-bit case.
//
// Build (Linux / GH200):
//   g++ -O3 -std=c++17 -march=native -pthread coppersmith.cpp -o coppersmith -lntl -lgmp -lm
//   (needs:  apt-get install libntl-dev libgmp-dev   -- or build NTL with NTL_THREADS=on)
//
// Usage:
//   coppersmith  N  guessP  [threads] [m] [t] [safety]
//   coppersmith  --selftest [bits=64] [threads]
//
// Output:
//   P = <factor>
//   Q = <cofactor>
//
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <complex>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace NTL;

// ----------------------------- configuration -----------------------------
struct Params {
    long   m;        // multiplicity
    long   t;        // number of extra x-shifts
    long   d;        // lattice dimension = m + 1 + t
    long   deg;      // max polynomial degree = m + t
    double beta;     // P >= N^beta (0.49 default: covers slightly-imbalanced 64-bit factors)
};

// --------------------------- global search state -------------------------
static atomic<bool>               g_found{false};
static atomic<unsigned long long> g_next_k{0};
static atomic<unsigned long long> g_blocks_done{0};
static ZZ                         g_factor;
static mutex                      g_factor_mtx;

// ---------------------------- math helpers -------------------------------

// fpow[i] = coefficients (ascending) of (a + x)^i, for i = 0..m
static void build_fpow(const ZZ& a, long m, vector<vector<ZZ>>& fpow) {
    fpow.assign(m + 1, {});
    ZZ one; set(one);                         // one = 1
    fpow[0] = vector<ZZ>{ one };
    for (long i = 1; i <= m; i++) {
        const vector<ZZ>& prev = fpow[i - 1];
        vector<ZZ> cur(prev.size() + 1);      // default ZZ = 0
        for (size_t j = 0; j < prev.size(); j++) {
            cur[j]     += prev[j] * a;   // * a
            cur[j + 1] += prev[j];       // * x
        }
        fpow[i] = std::move(cur);
    }
}

// Build the (d x d) Howgrave-Graham basis for f(x)=a+x, root mod a divisor >= N^beta.
// Rows (each vanishes mod P^m at x0 = P - a):
//   g_i(x) = N^(m-i) * f(x)^i           for i = 0..m          (degrees 0..m)
//   h_k(x) = x^k     * f(x)^m           for k = 1..t          (degrees m+1..m+t)
// Column j is scaled by X^j so that the short vector encodes a true integer root.
static void build_matrix(const ZZ& a, const Params& P,
                         const vector<ZZ>& Npow, const vector<ZZ>& Xpow,
                         vector<vector<ZZ>>& fpow, mat_ZZ& B) {
    build_fpow(a, P.m, fpow);
    const long d = P.d;
    for (long r = 0; r < d; r++)
        for (long c = 0; c < d; c++)
            B[r][c] = 0;

    ZZ tmp;
    // g_i rows
    for (long i = 0; i <= P.m; i++) {
        const vector<ZZ>& fi = fpow[i];           // length i+1
        const ZZ& Npw = Npow[P.m - i];
        for (long j = 0; j <= i; j++) {
            mul(tmp, fi[j], Npw);
            mul(tmp, tmp, Xpow[j]);
            B[i][j] = tmp;
        }
    }
    // h_k rows
    const vector<ZZ>& fm = fpow[P.m];             // length m+1
    for (long k = 1; k <= P.t; k++) {
        long row = P.m + k;
        for (long c = 0; c <= P.m; c++) {
            long j = c + k;
            mul(tmp, fm[c], Xpow[j]);
            B[row][j] = tmp;
        }
    }
}

// Find real roots (via Durand-Kerner) of the SCALED polynomial
//   G(y) = sum_j coeff[j] y^j ,   y = x / X.
// The lattice columns are X-scaled precisely so these coefficients are balanced
// (all ~P^m), which makes double-precision root finding well-conditioned. The
// true root is y0 = x0 / X in [-1, 1]; the caller recovers x0 = round(y0 * X).
// Roots are kept within |y| <= ybound (slightly above 1 for safety margin).
static void find_real_roots(const vector<double>& coeff, double ybound,
                            vector<double>& out) {
    long n = (long)coeff.size() - 1;
    while (n >= 1 && coeff[n] == 0.0) n--;
    if (n < 1) return;

    vector<complex<double>> c(n + 1);
    for (long j = 0; j <= n; j++) c[j] = complex<double>(coeff[j], 0.0);

    complex<double> lead = c[n];
    if (abs(lead) == 0.0) return;
    for (auto& z : c) z /= lead;                  // make monic

    vector<complex<double>> r(n);
    complex<double> seed(0.4, 0.9), p(1.0, 0.0);
    for (long k = 0; k < n; k++) { p *= seed; r[k] = p; }

    for (int it = 0; it < 300; it++) {
        double maxd = 0.0;
        for (long i = 0; i < n; i++) {
            // monic Horner: c[n]==1
            complex<double> val(1.0, 0.0);
            for (long j = n - 1; j >= 0; j--) val = val * r[i] + c[j];
            complex<double> den(1.0, 0.0);
            for (long j = 0; j < n; j++) if (j != i) den *= (r[i] - r[j]);
            if (abs(den) == 0.0) continue;
            complex<double> delta = val / den;
            r[i] -= delta;
            maxd = max(maxd, abs(delta));
        }
        if (maxd < 1e-13) break;
    }

    for (long i = 0; i < n; i++) {
        double re = r[i].real(), im = r[i].imag();
        if (fabs(im) < 1e-6 * (1.0 + fabs(re)) && fabs(re) <= ybound)
            out.push_back(re);
    }
}

// Test one block centred at 'a'. Returns true and sets 'outFactor' on success.
static bool test_block(const ZZ& a, const Params& P, const ZZ& N,
                       const vector<ZZ>& Npow, const vector<ZZ>& Xpow,
                       double Xdouble, vector<vector<ZZ>>& fpow,
                       mat_ZZ& B, ZZ& outFactor) {
    build_matrix(a, P, Npow, Xpow, fpow, B);
    LLL_XD(B, 0.99);

    ZZ x0, cand, rem;
    vector<double> coeff, roots;
    for (long row = 0; row < P.d; row++) {
        // Scaled coefficients straight from the (X-scaled) lattice row -> balanced.
        coeff.assign(P.d, 0.0);
        long hi = -1;
        for (long j = 0; j < P.d; j++) {
            if (IsZero(B[row][j])) continue;
            coeff[j] = to_double(B[row][j]);
            hi = j;
        }
        if (hi < 1) continue;
        coeff.resize(hi + 1);

        roots.clear();
        find_real_roots(coeff, 1.5, roots);          // roots in y = x/X
        for (double y : roots) {
            long base = (long)llround(y * Xdouble);   // back to x0
            for (long off = -4; off <= 4; off++) {
                conv(x0, base + off);
                cand = a + x0;
                if (cand <= 1 || cand >= N) continue;
                rem = N % cand;
                if (IsZero(rem)) { outFactor = cand; return true; }
            }
        }
    }
    return false;
}

// ------------------------------ search driver ----------------------------

static void worker(ZZ N, ZZ center, ZZ step, Params P,
                   vector<ZZ> Npow, vector<ZZ> Xpow, double Xbound,
                   unsigned long long maxblocks) {
    mat_ZZ B;
    B.SetDims(P.d, P.d);
    vector<vector<ZZ>> fpow;
    ZZ a, factor;

    while (!g_found.load(memory_order_relaxed)) {
        unsigned long long k = g_next_k.fetch_add(1, memory_order_relaxed);
        if (maxblocks && k >= maxblocks) return;

        // Spiral: even k -> upward, odd k -> downward (closest blocks first).
        if (k % 2 == 0) {
            long half = (long)(k / 2);
            a = center + step * conv<ZZ>(half);
        } else {
            long half = (long)((k + 1) / 2);
            a = center - step * conv<ZZ>(half);
        }
        if (a <= 2 || a >= N) { g_blocks_done.fetch_add(1, memory_order_relaxed); continue; }

        if (test_block(a, P, N, Npow, Xpow, Xbound, fpow, B, factor)) {
            lock_guard<mutex> lk(g_factor_mtx);
            if (!g_found.load()) { g_factor = factor; g_found.store(true); }
            return;
        }
        g_blocks_done.fetch_add(1, memory_order_relaxed);
    }
}

static void monitor(ZZ step, double Xbits) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    unsigned long long last = 0;
    while (!g_found.load()) {
        this_thread::sleep_for(seconds(5));
        if (g_found.load()) break;
        unsigned long long done = g_blocks_done.load();
        double secs = duration<double>(steady_clock::now() - t0).count();
        double rate = (done - last) / 5.0;
        last = done;
        // current spiral radius ~ (k/2)*step ; report its bit-length
        unsigned long long k = g_next_k.load();
        ZZ radius = step * conv<ZZ>((long)(k / 2 + 1));
        fprintf(stderr,
            "[search] blocks=%llu  rate=%.0f blk/s  radius~2^%.1f  elapsed=%.0fs\n",
            done, rate, (double)NumBits(radius), secs);
    }
}

// ------------------------------ parameter setup --------------------------

// Reachable root half-width exponent: X = N^Xexp, with
//   Xexp = (beta*m*d - m(m+1)/2) / E,   E = m(m+1)/2 + t*m + t(t+1)/2.
static double x_exponent(const Params& P) {
    double m = P.m, t = P.t, d = P.d;
    double E   = m * (m + 1) / 2.0 + t * m + t * (t + 1) / 2.0;
    double num = P.beta * m * d - m * (m + 1) / 2.0;
    return num / E;
}

static int run_search(const ZZ& N, const ZZ& guess, unsigned T,
                      Params P, double safety, unsigned long long maxblocks) {
    P.d   = P.m + 1 + P.t;
    P.deg = P.m + P.t;

    // Quick wins before launching the lattice search.
    if (guess > 1 && guess < N && N % guess == 0) {
        ZZ q = N / guess;
        cout << "P = " << guess << "\nQ = " << q << "\n";
        cout << "(guess was exact)\n";
        return 0;
    }
    {
        ZZ g = GCD(guess, N);
        if (g > 1 && g < N) { ZZ q = N / g; cout << "P = " << g << "\nQ = " << q << "\n(via gcd)\n"; return 0; }
    }

    double log2N = log(to_double(N)) / log(2.0);
    double Xexp  = x_exponent(P);
    double Xbits = Xexp * log2N;
    if (Xbits < 1.0) {
        fprintf(stderr, "[error] parameters give X < 2 (Xbits=%.2f); raise m/t.\n", Xbits);
        return 2;
    }

    // X = 2^floor(Xbits)  (the lattice scaling / recoverable half-width)
    ZZ Xscale = power2_ZZ((long)floor(Xbits));
    // Block step = X / safety, so any factor lands well inside a block's radius.
    ZZ step = Xscale / conv<ZZ>((long)llround(safety));
    if (step < 1) set(step);

    // Precompute N^e (e=0..m) and X^e (e=0..m+t).
    vector<ZZ> Npow(P.m + 1), Xpow(P.deg + 1);
    set(Npow[0]); for (long e = 1; e <= P.m;   e++) Npow[e] = Npow[e - 1] * N;
    set(Xpow[0]); for (long e = 1; e <= P.deg; e++) Xpow[e] = Xpow[e - 1] * Xscale;

    double Xbound = to_double(Xscale);

    fprintf(stderr,
        "[config] m=%ld t=%ld dim=%ld  beta=%.3f  log2(N)=%.1f\n"
        "         X ~ 2^%.1f (recoverable half-width)   step ~ 2^%.1f\n"
        "         threads=%u  safety=%.1f%s\n",
        P.m, P.t, P.d, P.beta, log2N, Xbits, (double)NumBits(step), T, safety,
        maxblocks ? "" : "  (unbounded: runs until P is found)");

    auto t0 = chrono::steady_clock::now();

    thread mon(monitor, step, Xbits);
    vector<thread> threads;
    for (unsigned i = 0; i < T; i++)
        threads.emplace_back(worker, N, guess, step, P, Npow, Xpow, Xbound, maxblocks);
    for (auto& th : threads) th.join();
    g_found.store(true);          // ensure monitor exits even if maxblocks hit
    mon.join();

    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    ZZ factor;
    { lock_guard<mutex> lk(g_factor_mtx); factor = g_factor; }
    if (factor > 1 && factor < N && N % factor == 0) {
        ZZ q = N / factor;
        ZZ P_, Q_;
        if (factor <= q) { P_ = factor; Q_ = q; } else { P_ = q; Q_ = factor; }
        cout << "P = " << P_ << "\nQ = " << Q_ << "\n";
        fprintf(stderr, "[done] found in %.3f s  (%llu blocks)\n",
                secs, (unsigned long long)g_blocks_done.load());
        return 0;
    }

    fprintf(stderr, "[fail] exhausted %llu blocks without finding a factor.\n",
            (unsigned long long)g_blocks_done.load());
    return 3;
}

// ------------------------------- selftest --------------------------------

static int selftest(long bits, unsigned T) {
    ZZ p = GenPrime_ZZ(bits);
    ZZ q = GenPrime_ZZ(bits);
    ZZ N = p * q;

    // Correctness check only: perturb p by a SMALL amount (well inside one
    // Coppersmith block, ~2^(bits/2-8)) so recovery happens within a handful of
    // blocks and returns instantly. This validates the lattice/root math.
    // To measure the real full-2% spiral, run the binary with a 2%-off guess.
    long sbits = max(4L, bits / 2 - 12);
    ZZ delta = RandomBnd(power2_ZZ(sbits));
    ZZ guess = (RandomBnd(2) == 0) ? p + delta : p - delta;

    fprintf(stderr, "[selftest] bits=%ld  (small perturbation: correctness check)\n  p     = ", bits);
    cerr << p << "\n  q     = " << q << "\n  N     = " << N
         << "\n  guess = " << guess << "  (|p-guess| ~ 2^" << (double)NumBits(delta) << ")\n";

    Params P{4, 4, 0, 0, 0.49};
    int rc = run_search(N, guess, T, P, 2.0, /*maxblocks=*/0);
    if (rc == 0) fprintf(stderr, "[selftest] PASS (Coppersmith recovery validated)\n");
    return rc;
}

// --------------------------------- main ----------------------------------

int main(int argc, char** argv) {
    if (argc >= 2 && string(argv[1]) == "--selftest") {
        long bits = (argc >= 3) ? atol(argv[2]) : 64;
        unsigned T = (argc >= 4) ? (unsigned)atoi(argv[3])
                                 : max(1u, thread::hardware_concurrency());
        return selftest(bits, T);
    }

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s N guessP [threads] [m] [t] [safety]\n"
            "       %s --selftest [bits=64] [threads]\n", argv[0], argv[0]);
        return 1;
    }

    ZZ N, guess;
    { stringstream ss(argv[1]); ss >> N; }
    { stringstream ss(argv[2]); ss >> guess; }
    if (N <= 3) { fprintf(stderr, "[error] N too small.\n"); return 1; }

    unsigned T   = (argc > 3) ? (unsigned)atoi(argv[3])
                              : max(1u, thread::hardware_concurrency());
    Params P;
    P.m    = (argc > 4) ? atol(argv[4]) : 4;
    P.t    = (argc > 5) ? atol(argv[5]) : 4;
    P.beta = 0.49;
    double safety = (argc > 6) ? atof(argv[6]) : 2.0;
    if (P.m < 1) P.m = 1;
    if (P.t < 1) P.t = 1;
    if (safety < 1.0) safety = 1.0;

    return run_search(N, guess, T, P, safety, /*maxblocks=*/0);
}
