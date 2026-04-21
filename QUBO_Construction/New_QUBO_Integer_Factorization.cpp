// New_QUBO_Integer_Factorization.cpp
//
// Clean-room C++ reimplementation of QUBO_Integer_Factorization.py.
// Builds the CSR Ising representation (row_ptr, col_idx, J_values, h) plus
// the bookkeeping side-lists (ass, expr, mul) needed for post-processing.
//
// Design:
//   - All symbolic polynomial arithmetic uses __int128 coefficients
//     (requires GCC/Clang; does NOT compile under MSVC without adaptation).
//   - Conversion to `double` happens only at the final CSR emission step,
//     matching the numeric type the rest of the SA pipeline consumes.
//   - No external dependencies.
//
// Usage:
//   g++ -O2 -std=c++17 New_QUBO_Integer_Factorization.cpp -o new_qubo
//   ./new_qubo <N>         # N is a decimal integer (may exceed 64 bits)
//   ./new_qubo             # defaults to 159197
//
// Outputs in the current working directory:
//   row_ptr_<N>.csv, col_idx_<N>.csv, J_values_<N>.csv, h_vector_<N>.csv

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using i128 = __int128;
using u128 = unsigned __int128;

// ============================================================
// Section 1 — int128 helpers
// ============================================================

static i128 i128_abs(i128 x) { return x < 0 ? -x : x; }

static i128 i128_gcd(i128 a, i128 b) {
    a = i128_abs(a);
    b = i128_abs(b);
    while (b) { i128 t = a % b; a = b; b = t; }
    return a;
}

static std::string u128_to_string(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) { s += char('0' + int(v % 10)); v /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}

static std::string i128_to_string(i128 v) {
    if (v == 0) return "0";
    bool neg = v < 0;
    u128 u = neg ? (u128)(-v) : (u128)v;
    std::string s = u128_to_string(u);
    if (neg) s = "-" + s;
    return s;
}

static u128 parse_u128(const std::string& s) {
    u128 r = 0;
    for (char c : s) {
        if (c < '0' || c > '9')
            throw std::runtime_error("non-digit in N: " + s);
        r = r * 10 + u128(c - '0');
    }
    return r;
}

static int u128_bit_length(u128 v) {
    int n = 0;
    while (v > 0) { n++; v >>= 1; }
    return n;
}

static double u128_to_double(u128 v) {
    if (v == 0) return 0.0;
    uint64_t hi = (uint64_t)(v >> 64);
    uint64_t lo = (uint64_t)v;
    return (double)hi * 18446744073709551616.0 + (double)lo;
}

static bool u128_bit(u128 v, int i) { return (v >> i) & 1; }

// ============================================================
// Section 2 — BinaryPoly
// ============================================================
//
// A polynomial over binary variables. Because x^2 = x, each monomial is
// uniquely identified by a sorted, unique vector<string> of variable names.
// The empty vector represents the constant term.

using Mono = std::vector<std::string>;

struct BinaryPoly {
    // Ordered map so iteration is deterministic (sorted by monomial key).
    std::map<Mono, i128> terms;

    BinaryPoly() = default;

    static BinaryPoly constant(i128 c) {
        BinaryPoly p;
        if (c != 0) p.terms[Mono{}] = c;
        return p;
    }

    static BinaryPoly var(const std::string& name) {
        BinaryPoly p;
        p.terms[Mono{name}] = 1;
        return p;
    }

    static BinaryPoly monomial(const Mono& m, i128 c) {
        BinaryPoly p;
        if (c != 0) p.terms[m] = c;
        return p;
    }

    bool is_zero() const { return terms.empty(); }

    i128 constant_term() const {
        auto it = terms.find(Mono{});
        return (it == terms.end()) ? (i128)0 : it->second;
    }

    std::set<std::string> free_symbols() const {
        std::set<std::string> s;
        for (auto& kv : terms) for (auto& v : kv.first) s.insert(v);
        return s;
    }

    void add_term(const Mono& m, i128 c) {
        if (c == 0) return;
        auto it = terms.find(m);
        if (it == terms.end()) terms.emplace(m, c);
        else { it->second += c; if (it->second == 0) terms.erase(it); }
    }

    BinaryPoly& operator+=(const BinaryPoly& o) {
        for (auto& kv : o.terms) add_term(kv.first, kv.second);
        return *this;
    }
    BinaryPoly& operator-=(const BinaryPoly& o) {
        for (auto& kv : o.terms) add_term(kv.first, -kv.second);
        return *this;
    }
    BinaryPoly operator+(const BinaryPoly& o) const { auto r = *this; r += o; return r; }
    BinaryPoly operator-(const BinaryPoly& o) const { auto r = *this; r -= o; return r; }
    BinaryPoly operator-() const {
        BinaryPoly r;
        for (auto& kv : terms) r.terms[kv.first] = -kv.second;
        return r;
    }

    // Union of two sorted, unique monomial keys — automatic x^2 = x.
    static Mono mul_mono(const Mono& a, const Mono& b) {
        Mono r;
        r.reserve(a.size() + b.size());
        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
        return r;
    }

    BinaryPoly operator*(const BinaryPoly& o) const {
        BinaryPoly r;
        for (auto& a : terms)
            for (auto& b : o.terms)
                r.add_term(mul_mono(a.first, b.first), a.second * b.second);
        return r;
    }

    BinaryPoly& operator*=(i128 s) {
        if (s == 0) { terms.clear(); return *this; }
        for (auto& kv : terms) kv.second *= s;
        return *this;
    }

    BinaryPoly squared() const { return (*this) * (*this); }

    i128 gcd_of_coeffs() const {
        i128 g = 0;
        for (auto& kv : terms) g = i128_gcd(g, kv.second);
        return (g == 0) ? (i128)1 : g;
    }

    void divide_by(i128 g) {
        if (g == 1 || g == 0) return;
        for (auto& kv : terms) kv.second /= g;
    }

    // Substitute a monomial m (size >= 1) with a polynomial expr.
    // For each term containing m as a subset, remove m's variables and
    // multiply the term by expr.  Single-variable case handles v -> poly.
    BinaryPoly substitute(const Mono& m, const BinaryPoly& expr) const {
        BinaryPoly result;
        for (auto& kv : terms) {
            const Mono& k = kv.first;
            if (std::includes(k.begin(), k.end(), m.begin(), m.end())) {
                Mono rest;
                std::set_difference(k.begin(), k.end(),
                                    m.begin(), m.end(),
                                    std::back_inserter(rest));
                for (auto& ek : expr.terms) {
                    result.add_term(mul_mono(rest, ek.first),
                                    kv.second * ek.second);
                }
            } else {
                result.add_term(k, kv.second);
            }
        }
        return result;
    }

    BinaryPoly substitute_var(const std::string& v, const BinaryPoly& expr) const {
        return substitute(Mono{v}, expr);
    }

    // Canonical string form for tracing / debugging (matches Python _canon).
    std::string canon() const {
        i128 c0 = constant_term();
        std::vector<std::pair<Mono, i128>> parts;
        parts.reserve(terms.size());
        for (auto& kv : terms) if (!kv.first.empty()) parts.push_back(kv);
        std::sort(parts.begin(), parts.end());
        std::string s = i128_to_string(c0);
        for (auto& p : parts) {
            i128 c = p.second;
            std::string sign = (c >= 0) ? "+" : "-";
            i128 a = i128_abs(c);
            s += " ";
            s += sign;
            s += " ";
            s += i128_to_string(a);
            s += "*";
            for (size_t i = 0; i < p.first.size(); ++i) {
                if (i) s += "*";
                s += p.first[i];
            }
        }
        return s;
    }
};

// Convenience: is the monomial key a "p/q" monomial (for tracking lists)?
static bool mono_has_p_or_q(const Mono& m) {
    for (auto& v : m)
        if (!v.empty() && (v[0] == 'p' || v[0] == 'q')) return true;
    return false;
}

static bool mono_has_s(const Mono& m) {
    for (auto& v : m)
        if (v.size() >= 2 && v[0] == 's' && v[1] == '_') return true;
    return false;
}

// ============================================================
// Section 3 — Variable setup + column clauses
// ============================================================

struct SetupResult {
    int n_p;
    int n_q;
    std::set<std::pair<int,int>> s_vars; // (i, i+j) keys
};

// Mirrors initialize_variables.
static SetupResult initialize_variables(u128 N) {
    int bits = u128_bit_length(N);
    double lg;
    if (bits <= 53) {
        lg = std::log2(u128_to_double(N));
    } else {
        // Stable log2 for values beyond double's exact range.
        // log2(N) = bit_length - 1 + log2(N / 2^(bit_length-1)).
        int shift = bits - 53;
        u128 top = N >> shift;
        lg = (double)shift + std::log2((double)(uint64_t)top);
    }
    int n_q = (int)std::ceil(0.5 * lg);
    int n_p = n_q;

    SetupResult r;
    r.n_p = n_p;
    r.n_q = n_q;

    for (int i = 1; i < n_p + n_q; ++i) {
        int L_i = std::max(i - n_p + 1, 0);
        int L_min = std::min(i, n_q - 1);
        int num_prod_terms = L_min - L_i + 1;
        int num_carry_in = i - 1;
        int max_sum_in_col = num_prod_terms + num_carry_in;
        if (max_sum_in_col > 1) {
            int num_carry_out_bits = (int)std::floor(std::log2((double)max_sum_in_col));
            for (int j = 1; j <= num_carry_out_bits; ++j) {
                if (i + j < n_p + n_q)
                    r.s_vars.insert({i, i + j});
            }
        }
    }
    return r;
}

// p_k and q_k as BinaryPoly.  p_0 = q_0 = p_{n_p-1} = q_{n_q-1} = 1 are constants.
static BinaryPoly p_var(int k, int n_p) {
    if (k == 0 || k == n_p - 1) return BinaryPoly::constant(1);
    return BinaryPoly::var("p_" + std::to_string(k));
}
static BinaryPoly q_var(int k, int n_q) {
    if (k == 0 || k == n_q - 1) return BinaryPoly::constant(1);
    return BinaryPoly::var("q_" + std::to_string(k));
}
static std::string s_name(int i, int j) {
    return "s_" + std::to_string(i) + "_" + std::to_string(j);
}

// Mirrors generate_column_clauses.
static std::vector<BinaryPoly> generate_column_clauses(
    u128 N, int n_p, int n_q, const std::set<std::pair<int,int>>& s_vars)
{
    int n_m = n_p + n_q;
    // N_bits[i] = (i-th bit of N), LSB first, zero-padded to n_m.
    std::vector<int> N_bits(n_m, 0);
    for (int i = 0; i < n_m; ++i) N_bits[i] = u128_bit(N, i) ? 1 : 0;

    std::vector<BinaryPoly> clauses;
    clauses.reserve(n_m - 1);

    for (int i = 1; i < n_m; ++i) {
        BinaryPoly clause = BinaryPoly::constant(N_bits[i]);

        // -- cross products p_j * q_{i-j}
        for (int j = 0; j < n_p; ++j) {
            int k = i - j;
            if (k >= 0 && k < n_q) {
                BinaryPoly prod = p_var(j, n_p) * q_var(k, n_q);
                clause -= prod;
            }
        }
        // -- input carries s_{k,i}
        for (int k = 1; k < i; ++k) {
            if (s_vars.count({k, i})) {
                clause -= BinaryPoly::var(s_name(k, i));
            }
        }
        // -- output carries 2^j * s_{i,i+j}
        int j = 1;
        while (s_vars.count({i, i + j})) {
            BinaryPoly carry = BinaryPoly::var(s_name(i, i + j));
            carry *= (i128)((u128)1 << j);
            clause += carry;
            // The Python code defensively appends s_{i,i+j} as its own clause
            // when i+j > n_p+n_q-1; with our initialize_variables that cannot
            // happen (the guard is baked in), so the guard here is trivial.
            if (i + j > n_p + n_q - 1) {
                clauses.push_back(BinaryPoly::var(s_name(i, i + j)));
            }
            j += 1;
        }
        clauses.push_back(clause);
    }
    return clauses;
}

// ============================================================
// Section 4 — Simplification rules
// ============================================================
//
// Each rule inspects a single clause and returns a set of monomial
// substitutions (Mono -> BinaryPoly).  The simplifier applies them to all
// clauses and restarts its pass.  For rule_3 we additionally return
// mul_ass, which tracks product-equality constraints that are *not*
// substituted into clauses but are preserved for post-processing.

using SubstMap = std::map<Mono, BinaryPoly>;

// ---- Rule 1 & 2 ----
static SubstMap apply_rule_1_and_2(const BinaryPoly& clause) {
    SubstMap out;
    std::vector<std::pair<Mono, i128>> non_const;
    i128 cst = 0;
    for (auto& kv : clause.terms) {
        if (kv.first.empty()) cst = kv.second;
        else non_const.push_back(kv);
    }
    if (non_const.empty()) return out;

    int n = (int)non_const.size();

    auto all_coef = [&](i128 want) {
        for (auto& kv : non_const) if (kv.second != want) return false;
        return true;
    };

    // Rule 1:  sum = n  =>  each = 1
    if (all_coef(-1) && cst == (i128)n) {
        for (auto& kv : non_const) out[kv.first] = BinaryPoly::constant(1);
        return out;
    }
    if (all_coef(1) && -cst == (i128)n) {
        for (auto& kv : non_const) out[kv.first] = BinaryPoly::constant(1);
        return out;
    }
    // Rule 2:  sum = 0  =>  each = 0
    if (all_coef(1) && cst == 0) {
        for (auto& kv : non_const) out[kv.first] = BinaryPoly::constant(0);
        return out;
    }
    if (all_coef(-1) && cst == 0) {
        for (auto& kv : non_const) out[kv.first] = BinaryPoly::constant(0);
        return out;
    }
    return out;
}

// ---- Rule 4 ----
static SubstMap apply_rule_4(const BinaryPoly& clause) {
    SubstMap out;
    i128 cst = 0;
    std::vector<std::pair<Mono, i128>> pos, neg;
    for (auto& kv : clause.terms) {
        if (kv.first.empty()) cst = kv.second;
        else if (kv.second > 0) pos.push_back(kv);
        else if (kv.second < 0) neg.push_back(kv);
    }
    i128 pos_sum = 0, neg_mag_sum = 0;
    for (auto& kv : pos) pos_sum += kv.second;
    for (auto& kv : neg) neg_mag_sum += -kv.second;

    for (auto& kv : pos) {
        if (kv.second > neg_mag_sum - cst) out[kv.first] = BinaryPoly::constant(0);
        if (kv.second > pos_sum + cst)     out[kv.first] = BinaryPoly::constant(1);
    }
    for (auto& kv : neg) {
        i128 mag = -kv.second;
        if (mag > pos_sum + cst)          out[kv.first] = BinaryPoly::constant(0);
        if (mag > neg_mag_sum - cst)      out[kv.first] = BinaryPoly::constant(1);
    }
    return out;
}

// ---- Rule 3 ----
// Match clause == m1 + m2 - 2*m3  (or its negation).  Each mi is a
// monomial with coefficient +/-1 or +/-2 as listed, no constant term.
// If any of the matched monomials is a product, the equality is tracked
// as a mul_ass entry rather than a direct substitution.
struct Rule3Result {
    SubstMap substs;
    SubstMap mul_ass;
};

static Rule3Result apply_rule_3(const BinaryPoly& clause) {
    Rule3Result r;
    // Collect non-constant monomials with their coefficients.
    i128 cst = 0;
    std::vector<std::pair<Mono, i128>> nc;
    for (auto& kv : clause.terms) {
        if (kv.first.empty()) cst = kv.second;
        else nc.push_back(kv);
    }
    if (cst != 0) return r;
    if (nc.size() != 3) return r;

    // Identify the (+1, +1, -2) layout, possibly after negation.
    auto match_one_two = [&](int sign) -> int {
        // returns the index of the (-2*sign) term, or -1 if layout mismatch.
        int idx_two = -1;
        int plus_count = 0;
        for (int i = 0; i < 3; ++i) {
            i128 c = nc[i].second * sign;
            if (c == 1) plus_count++;
            else if (c == -2) { if (idx_two != -1) return -1; idx_two = i; }
            else return -1;
        }
        return (plus_count == 2 && idx_two != -1) ? idx_two : -1;
    };

    int idx_two = match_one_two(+1);
    int sign = +1;
    if (idx_two < 0) {
        idx_two = match_one_two(-1);
        sign = -1;
    }
    if (idx_two < 0) return r;
    (void)sign; // only used to verify pattern; equalities are sign-symmetric.

    // v1, v2 have coefficient +sign, v3 has coefficient -2*sign.
    std::vector<Mono> vs;
    for (int i = 0; i < 3; ++i) if (i != idx_two) vs.push_back(nc[i].first);
    vs.push_back(nc[idx_two].first);
    // vs = {v1, v2, v3}

    // Classify each vi as: single variable (is_Symbol), product (is_Mul), or other.
    auto is_symbol = [&](const Mono& m) { return m.size() == 1; };
    auto is_mul    = [&](const Mono& m) { return m.size() >= 2; };

    // Separate into s-vars (single-var monos whose name starts with s_) and
    // other single-var monos + products.  Order within each group preserves
    // the original order in vs.
    std::vector<int> s_idx, other_idx;
    std::vector<int> mul_idx;
    for (int i = 0; i < 3; ++i) {
        if (is_mul(vs[i])) mul_idx.push_back(i);
        else if (mono_has_s(vs[i])) s_idx.push_back(i);
        else other_idx.push_back(i);
    }

    auto mono_to_poly = [](const Mono& m) {
        return BinaryPoly::monomial(m, 1);
    };

    if (mul_idx.empty()) {
        // All three are single-variable monomials.
        if (!s_idx.empty()) {
            // Prefer to eliminate s_ variables.
            Mono target = other_idx.empty() ? vs[s_idx[0]] : vs[other_idx[0]];
            for (int si : s_idx) {
                if (vs[si] != target)
                    r.substs[vs[si]] = mono_to_poly(target);
            }
            if (other_idx.size() > 1) {
                r.substs[vs[other_idx[1]]] = mono_to_poly(vs[other_idx[0]]);
            }
        } else {
            // No s vars; set v1 = v2, v3 = v2.
            r.substs[vs[0]] = mono_to_poly(vs[1]);
            r.substs[vs[2]] = mono_to_poly(vs[1]);
        }
        return r;
    }

    // There is at least one product monomial -- follow the Python branch.
    // In Python the logic uses s_vars / other_vars ordering from vs.
    std::vector<int> s_vars_idx;
    std::vector<int> other_vars_idx;
    for (int i = 0; i < 3; ++i) {
        if (!is_mul(vs[i]) && mono_has_s(vs[i])) s_vars_idx.push_back(i);
        else if (!is_mul(vs[i])) other_vars_idx.push_back(i);
        else other_vars_idx.push_back(i); // muls go into "other_vars" per Python
    }

    if (s_vars_idx.size() == 2) {
        r.substs[vs[s_vars_idx[0]]] = mono_to_poly(vs[s_vars_idx[1]]);
        r.mul_ass[vs[s_vars_idx[0]]] = mono_to_poly(vs[mul_idx[0]]);
        r.mul_ass[vs[s_vars_idx[1]]] = mono_to_poly(vs[mul_idx[0]]);
    } else if (s_vars_idx.size() == 1) {
        int si = s_vars_idx[0];
        // other_vars has two entries (one mul, one symbol-or-mul).
        if (other_vars_idx.size() < 2) return r; // safety
        int o0 = other_vars_idx[0], o1 = other_vars_idx[1];
        if (is_symbol(vs[o0])) {
            r.substs[vs[si]] = mono_to_poly(vs[o0]);
            r.mul_ass[vs[o1]] = mono_to_poly(vs[o0]);
        } else if (is_symbol(vs[o1])) {
            r.substs[vs[si]] = mono_to_poly(vs[o1]);
            r.mul_ass[vs[o0]] = mono_to_poly(vs[o1]);
        } else {
            r.mul_ass[vs[si]] = mono_to_poly(vs[o0]);
            r.mul_ass[vs[o1]] = mono_to_poly(vs[o0]);
        }
    } else {
        // zero s-vars among vs
        if (mul_idx.size() == 3) {
            r.mul_ass[vs[0]] = mono_to_poly(vs[2]);
            r.mul_ass[vs[1]] = mono_to_poly(vs[2]);
        } else if (mul_idx.size() == 2) {
            if (other_vars_idx.size() >= 1 && is_symbol(vs[other_vars_idx[0]])) {
                r.mul_ass[vs[other_vars_idx[1]]] = mono_to_poly(vs[other_vars_idx[0]]);
                r.mul_ass[vs[other_vars_idx[2]]] = mono_to_poly(vs[other_vars_idx[0]]);
            } else if (other_vars_idx.size() >= 2 && is_symbol(vs[other_vars_idx[1]])) {
                r.mul_ass[vs[other_vars_idx[0]]] = mono_to_poly(vs[other_vars_idx[1]]);
                r.mul_ass[vs[other_vars_idx[2]]] = mono_to_poly(vs[other_vars_idx[1]]);
            } else {
                r.mul_ass[vs[other_vars_idx[0]]] = mono_to_poly(vs[other_vars_idx[2]]);
                r.mul_ass[vs[other_vars_idx[1]]] = mono_to_poly(vs[other_vars_idx[2]]);
            }
        } else {
            // exactly one mul
            int m = mul_idx[0];
            std::vector<int> rest;
            for (int i = 0; i < 3; ++i) if (i != m) rest.push_back(i);
            if (is_mul(vs[rest[0]])) {
                r.mul_ass[vs[rest[0]]] = mono_to_poly(vs[rest[1]]);
                r.substs[vs[m]]         = mono_to_poly(vs[rest[1]]);
            } else if (is_mul(vs[rest[1]])) {
                r.mul_ass[vs[rest[1]]] = mono_to_poly(vs[rest[0]]);
                r.substs[vs[m]]         = mono_to_poly(vs[rest[0]]);
            } else {
                r.mul_ass[vs[m]]        = mono_to_poly(vs[rest[1]]);
                r.substs[vs[rest[0]]]  = mono_to_poly(vs[rest[1]]);
            }
        }
    }
    return r;
}

// ---- Rule 6 ----
// Pair-complementary: y = 1 - x when the combined magnitude cannot both be
// satisfied nor both denied.
static SubstMap apply_rule_6(const BinaryPoly& clause) {
    SubstMap out;
    i128 cst = 0;
    std::vector<std::pair<Mono, i128>> pos, neg_mag;
    for (auto& kv : clause.terms) {
        if (kv.first.empty()) cst = kv.second;
        else if (kv.second > 0) pos.push_back({kv.first, kv.second});
        else if (kv.second < 0) neg_mag.push_back({kv.first, -kv.second});
    }
    i128 pos_sum = 0, neg_sum = 0;
    for (auto& kv : pos) pos_sum += kv.second;
    for (auto& kv : neg_mag) neg_sum += kv.second;

    auto cmp_desc = [](const std::pair<Mono,i128>& a, const std::pair<Mono,i128>& b) {
        return a.second > b.second;
    };
    auto pos_sort = pos; std::sort(pos_sort.begin(), pos_sort.end(), cmp_desc);
    auto neg_sort = neg_mag; std::sort(neg_sort.begin(), neg_sort.end(), cmp_desc);

    auto set_complement = [&](const Mono& a, const Mono& b) {
        // Prefer to eliminate the s-containing side.
        Mono lhs = a;
        Mono rhs = b;
        if (mono_has_s(a))      { lhs = a; rhs = b; }
        else if (mono_has_s(b)) { lhs = b; rhs = a; }
        // constant 1 - rhs
        BinaryPoly v = BinaryPoly::constant(1);
        v -= BinaryPoly::monomial(rhs, 1);
        out[lhs] = v;
    };

    if (pos_sum > neg_sum - cst && cst < 0) {
        for (size_t i = 0; i < pos_sort.size(); ++i) {
            bool stop = false;
            for (size_t j = i + 1; j < pos_sort.size(); ++j) {
                i128 cy = pos_sort[i].second;
                i128 cx = pos_sort[j].second;
                if (cy + cx > neg_sum - cst) {
                    if (-cst - pos_sum + cy + cx > 0) {
                        set_complement(pos_sort[i].first, pos_sort[j].first);
                        stop = true;
                        break;
                    }
                } else {
                    stop = true;
                    break;
                }
            }
            if (stop) break;
        }
    }
    if (neg_sum > pos_sum + cst && cst > 0) {
        for (size_t i = 0; i < neg_sort.size(); ++i) {
            bool stop = false;
            for (size_t j = i + 1; j < neg_sort.size(); ++j) {
                i128 cy = neg_sort[i].second;
                i128 cx = neg_sort[j].second;
                if (cy + cx > pos_sum + cst) {
                    if (neg_sum - cst - cy - cx < 0) {
                        set_complement(neg_sort[i].first, neg_sort[j].first);
                        stop = true;
                        break;
                    }
                } else {
                    stop = true;
                    break;
                }
            }
            if (stop) break;
        }
    }
    return out;
}

// ---- Parity rule ----
static SubstMap apply_parity_rule(const BinaryPoly& clause) {
    SubstMap out;
    // Collect monomials whose coefficient is odd.
    std::vector<std::pair<Mono, i128>> odd; // coeffs normalized to +/-1
    i128 cst_odd = 0;
    for (auto& kv : clause.terms) {
        if (kv.first.empty()) {
            if ((i128_abs(kv.second) % 2) == 1)
                cst_odd = ((kv.second % 2) + 2) % 2 == 0 ? 0 : (kv.second > 0 ? 1 : -1);
        } else {
            i128 r = kv.second % 2;
            if (r != 0) odd.push_back({kv.first, (kv.second > 0 ? 1 : -1)});
        }
    }
    if (odd.size() != 2) return out;

    // Must have two single-variable monomials (the Python check is
    // match[x_i].is_Symbol).
    if (odd[0].first.size() != 1 || odd[1].first.size() != 1) return out;

    const Mono& A = odd[0].first;
    const Mono& B = odd[1].first;
    i128 sa = odd[0].second;
    i128 sb = odd[1].second;

    auto set_equal = [&](const Mono& lhs, const Mono& rhs) {
        // Prefer to eliminate an s-variable.
        Mono elim = lhs;
        Mono keep = rhs;
        if (mono_has_s(lhs))      { elim = lhs; keep = rhs; }
        else if (mono_has_s(rhs)) { elim = rhs; keep = lhs; }
        out[elim] = BinaryPoly::monomial(keep, 1);
    };
    auto set_complement = [&](const Mono& lhs, const Mono& rhs) {
        Mono elim = lhs;
        Mono keep = rhs;
        if (mono_has_s(lhs))      { elim = lhs; keep = rhs; }
        else if (mono_has_s(rhs)) { elim = rhs; keep = lhs; }
        BinaryPoly v = BinaryPoly::constant(1);
        v -= BinaryPoly::monomial(keep, 1);
        out[elim] = v;
    };

    if (cst_odd == 0) {
        // x1 + x2 even  =>  x1 = x2
        if (sa == sb) set_equal(A, B);
        else          set_equal(A, B); // x1 - x2 = 0 also implies equality
    } else {
        // x1 + x2 odd  =>  x1 + x2 = 1  =>  x1 = 1 - x2
        set_complement(A, B);
    }
    return out;
}

// ---- Replacement rule (fallback) ----
// Only eliminates s-variables whose coefficient is +/- 1 in some clause.
static SubstMap apply_replacement(const std::vector<BinaryPoly>& clauses) {
    SubstMap out;
    for (auto& c : clauses) {
        if (c.is_zero()) continue;
        // Iterate variables in the clause.  Since clause.terms is a map,
        // iteration is deterministic by monomial key.
        std::set<std::string> vars;
        for (auto& kv : c.terms) for (auto& v : kv.first) vars.insert(v);
        for (auto& v : vars) {
            // Skip non-s vars.
            if (!(v.size() >= 2 && v[0] == 's' && v[1] == '_')) continue;
            // Find clause's coefficient on the single-variable monomial {v}.
            auto it = c.terms.find(Mono{v});
            if (it == c.terms.end()) continue;
            i128 coef = it->second;
            if (coef != 1 && coef != -1) continue;
            // Solve c = 0 for v:  v = -(rest) / coef.
            BinaryPoly rest = c;
            rest.terms.erase(Mono{v});   // drop the v term
            // v = -rest / coef
            BinaryPoly val = rest;
            if (coef == 1)  val *= (i128)(-1);
            else            /* coef == -1 */ ;  // v = -rest / -1 = rest
            out[Mono{v}] = val;
            return out;
        }
    }
    return out;
}

// ============================================================
// Section 5 — Clause simplifier main loop
// ============================================================

struct ConstraintEntry {
    Mono key;
    BinaryPoly value;
};

struct SimplifyResult {
    std::vector<BinaryPoly> clauses;
    std::vector<ConstraintEntry> assignments;  // p/q = constant (0 or 1)
    std::vector<ConstraintEntry> expressions;  // p/q = expression
    std::vector<ConstraintEntry> mul_constraints;
};

static void apply_substs_to_all(std::vector<BinaryPoly>& cls, const SubstMap& subs) {
    for (auto& c : cls) {
        for (auto& kv : subs) {
            c = c.substitute(kv.first, kv.second);
        }
    }
}

// After value-preserving substitutions, the power rule (x^n = x) is
// already baked into BinaryPoly by construction; nothing more to do.
static void apply_power_rule_noop(std::vector<BinaryPoly>& /*cls*/) {}

static SimplifyResult clause_simplifier(std::vector<BinaryPoly> clauses) {
    SimplifyResult R;
    int max_iter = 2 * (int)clauses.size();

    auto record_p_or_q = [&](std::vector<ConstraintEntry>& sink,
                             const SubstMap& subs) {
        for (auto& kv : subs) {
            if (mono_has_p_or_q(kv.first)) {
                sink.push_back({kv.first, kv.second});
            }
        }
    };

    // Expand "product = 1" into per-factor assignments.
    auto expand_product_equals_one = [](SubstMap& subs,
                                        std::vector<ConstraintEntry>& ass_sink) -> SubstMap {
        SubstMap expanded = subs;
        for (auto& kv : subs) {
            const Mono& m = kv.first;
            i128 v = kv.second.constant_term();
            bool is_const_one =
                kv.second.terms.size() == 1 &&
                kv.second.terms.count(Mono{}) &&
                v == 1;
            if (m.size() >= 2 && is_const_one && mono_has_p_or_q(m)) {
                for (auto& var : m) {
                    Mono single{var};
                    expanded[single] = BinaryPoly::constant(1);
                    ass_sink.push_back({single, BinaryPoly::constant(1)});
                }
            }
        }
        return expanded;
    };

    for (int it = 0; it < max_iter; ++it) {
        int fired = 0;

        // (a) GCD-divide each clause
        for (auto& c : clauses) {
            if (c.is_zero()) continue;
            i128 g = c.gcd_of_coeffs();
            if (g > 1) c.divide_by(g);
        }

        // (b) Rule 1 & 2 — first matching clause
        for (size_t ci = 0; ci < clauses.size() && !fired; ++ci) {
            if (clauses[ci].is_zero()) continue;
            auto subs = apply_rule_1_and_2(clauses[ci]);
            if (subs.empty()) continue;
            // Track p/q assignments first (products before per-factor expansion).
            for (auto& kv : subs) {
                if (mono_has_p_or_q(kv.first))
                    R.assignments.push_back({kv.first, kv.second});
            }
            auto full = expand_product_equals_one(subs, R.assignments);
            apply_substs_to_all(clauses, full);
            fired = 1;
        }
        if (fired) continue;

        // (c) Rule 4
        for (size_t ci = 0; ci < clauses.size() && !fired; ++ci) {
            if (clauses[ci].is_zero()) continue;
            auto subs = apply_rule_4(clauses[ci]);
            if (subs.empty()) continue;
            for (auto& kv : subs) {
                if (mono_has_p_or_q(kv.first))
                    R.assignments.push_back({kv.first, kv.second});
            }
            auto full = expand_product_equals_one(subs, R.assignments);
            apply_substs_to_all(clauses, full);
            fired = 1;
        }
        if (fired) continue;

        // (d) Rule 3
        for (size_t ci = 0; ci < clauses.size() && !fired; ++ci) {
            if (clauses[ci].is_zero()) continue;
            auto r3 = apply_rule_3(clauses[ci]);
            if (!r3.mul_ass.empty()) {
                for (auto& kv : r3.mul_ass) {
                    R.mul_constraints.push_back({kv.first, kv.second});
                }
            }
            if (r3.substs.empty()) continue;
            record_p_or_q(R.expressions, r3.substs);
            apply_substs_to_all(clauses, r3.substs);
            apply_power_rule_noop(clauses);
            fired = 1;
        }
        if (fired) continue;

        // (e) Rule 6
        for (size_t ci = 0; ci < clauses.size() && !fired; ++ci) {
            if (clauses[ci].is_zero()) continue;
            auto subs = apply_rule_6(clauses[ci]);
            if (subs.empty()) continue;
            record_p_or_q(R.expressions, subs);
            apply_substs_to_all(clauses, subs);
            apply_power_rule_noop(clauses);
            fired = 1;
        }
        if (fired) continue;

        // (f) Parity
        for (size_t ci = 0; ci < clauses.size() && !fired; ++ci) {
            if (clauses[ci].is_zero()) continue;
            auto subs = apply_parity_rule(clauses[ci]);
            if (subs.empty()) continue;
            record_p_or_q(R.expressions, subs);
            apply_substs_to_all(clauses, subs);
            apply_power_rule_noop(clauses);
            fired = 1;
        }
        if (fired) continue;

        // (g) Replacement — fallback, s-vars only
        {
            auto subs = apply_replacement(clauses);
            if (subs.empty()) break;
            record_p_or_q(R.expressions, subs); // usually none for s-only
            apply_substs_to_all(clauses, subs);
            apply_power_rule_noop(clauses);
        }
    }

    // Drop zero clauses
    std::vector<BinaryPoly> kept;
    kept.reserve(clauses.size());
    for (auto& c : clauses) if (!c.is_zero()) kept.push_back(std::move(c));
    R.clauses = std::move(kept);
    return R;
}

// ============================================================
// Section 6 — Hamiltonian  H = sum of C_i^2
// ============================================================

static BinaryPoly qubo_hamiltonian_from_clauses(const std::vector<BinaryPoly>& clauses) {
    BinaryPoly H;
    for (auto& c : clauses) {
        if (c.is_zero()) continue;
        H += c.squared();
    }
    return H;
}

// ============================================================
// Section 7 — Quadrization
// ============================================================
//
// Input: H as a BinaryPoly of degree <= 4.
// Output: QUBO as linear[var] + quadratic[(u,v)] maps (u <= v lexicographically),
// after introducing auxiliary variables for every degree-3 and degree-4
// monomial, using the paper's reductions (equations 9, 11, and the 4-local
// positive/negative formulae).
//
// Auxiliaries are named w0, w1, ... and w#0, w#1, ...  (the latter only
// emitted in the positive-quartic reduction).  The index increments once
// per unique high-degree monomial processed.

struct QUBO {
    std::map<std::string, i128> linear;
    // Key is (a, b) with a <= b lexicographically.
    std::map<std::pair<std::string, std::string>, i128> quadratic;
};

static void qubo_add_linear(QUBO& q, const std::string& v, i128 c) {
    if (c == 0) return;
    auto it = q.linear.find(v);
    if (it == q.linear.end()) q.linear[v] = c;
    else { it->second += c; if (it->second == 0) q.linear.erase(it); }
}

static void qubo_add_quadratic(QUBO& q, const std::string& a, const std::string& b, i128 c) {
    if (c == 0) return;
    if (a == b) { qubo_add_linear(q, a, c); return; }
    auto key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    auto it = q.quadratic.find(key);
    if (it == q.quadratic.end()) q.quadratic[key] = c;
    else { it->second += c; if (it->second == 0) q.quadratic.erase(it); }
}

// Positive cubic:  +k * x1*x2*x3
//   -> aux w (linear +k), w-c:+k, w-b:-k, w-a:-k, a-b:+k
static void reduce_p3(QUBO& q, const std::string& a, const std::string& b,
                      const std::string& c, const std::string& w, i128 k) {
    qubo_add_linear(q, w, k);
    qubo_add_quadratic(q, w, c,  k);
    qubo_add_quadratic(q, w, b, -k);
    qubo_add_quadratic(q, w, a, -k);
    qubo_add_quadratic(q, a, b,  k);
}

// Negative cubic:  -|k| * x1*x2*x3  (k < 0 passed in)
//   -> aux w (linear -2|k|), w-a: +|k|, w-b: +|k|, w-c: +|k|
// Using the Python convention where `interaction = k` (negative),
// n3 uses  w: -2*interaction (= +2|k|), w-*: +interaction (= -|k|).
// That is equivalent up to sign of w; the Ising energy is the same.
static void reduce_n3(QUBO& q, const std::string& a, const std::string& b,
                      const std::string& c, const std::string& w, i128 k_neg) {
    // Replicates Python's n3 exactly with interaction = k_neg.
    qubo_add_linear(q, w, -2 * k_neg);
    qubo_add_quadratic(q, w, a, k_neg);
    qubo_add_quadratic(q, w, b, k_neg);
    qubo_add_quadratic(q, w, c, k_neg);
}

// Positive quartic:  +k * x1*x2*x3*x4
// Python p4:
//   w: +k, w#: +k
//   w-a:-k, w#-d:+k, w-b:-k, w#-c:-k, w-c:+k, w#-w:-k
//   a-b:+k
static void reduce_p4(QUBO& q,
                      const std::string& a, const std::string& b,
                      const std::string& c, const std::string& d,
                      const std::string& w, const std::string& wh, i128 k) {
    qubo_add_linear(q, w,  k);
    qubo_add_linear(q, wh, k);
    qubo_add_quadratic(q, w,  a, -k);
    qubo_add_quadratic(q, wh, d,  k);
    qubo_add_quadratic(q, w,  b, -k);
    qubo_add_quadratic(q, wh, c, -k);
    qubo_add_quadratic(q, w,  c,  k);
    qubo_add_quadratic(q, wh, w, -k);
    qubo_add_quadratic(q, a,  b,  k);
}

// Negative quartic:  -|k| * x1*x2*x3*x4  (k_neg < 0)
// Python n4: w: -3*interaction, w-*: +interaction each.
static void reduce_n4(QUBO& q,
                      const std::string& a, const std::string& b,
                      const std::string& c, const std::string& d,
                      const std::string& w, i128 k_neg) {
    qubo_add_linear(q, w, -3 * k_neg);
    qubo_add_quadratic(q, w, a, k_neg);
    qubo_add_quadratic(q, w, b, k_neg);
    qubo_add_quadratic(q, w, c, k_neg);
    qubo_add_quadratic(q, w, d, k_neg);
}

static QUBO quadrizate(const BinaryPoly& H) {
    QUBO q;
    // First copy through all linear / quadratic / constant terms; bucket the
    // cubics and quartics for aux-var reduction.
    std::vector<std::pair<Mono, i128>> high; // deg 3 or 4
    for (auto& kv : H.terms) {
        const Mono& k = kv.first;
        i128 c = kv.second;
        if (c == 0) continue;
        if (k.empty()) continue; // constant offset -- dropped (matches pyqubo)
        if (k.size() == 1) qubo_add_linear(q, k[0], c);
        else if (k.size() == 2) qubo_add_quadratic(q, k[0], k[1], c);
        else if (k.size() == 3 || k.size() == 4) high.push_back(kv);
        else {
            throw std::runtime_error("Unexpected monomial degree " +
                                     std::to_string(k.size()));
        }
    }

    // Deterministic aux-var numbering: sort by monomial key.
    std::sort(high.begin(), high.end());

    int n = 0;
    for (auto& kv : high) {
        const Mono& k = kv.first;
        i128 c = kv.second;
        std::string w = "w" + std::to_string(n);
        std::string wh = "w#" + std::to_string(n);
        if (k.size() == 3) {
            if (c > 0) reduce_p3(q, k[0], k[1], k[2], w, c);
            else       reduce_n3(q, k[0], k[1], k[2], w, c);
        } else { // size == 4
            if (c > 0) reduce_p4(q, k[0], k[1], k[2], k[3], w, wh, c);
            else       reduce_n4(q, k[0], k[1], k[2], k[3], w, c);
        }
        n += 1;
    }
    return q;
}

// ============================================================
// Section 8 — QUBO -> CSR Ising
// ============================================================

struct CSRIsing {
    std::vector<int>    row_ptr;
    std::vector<int>    col_idx;
    std::vector<double> values;
    std::vector<double> h;
    double              offset;
    std::vector<std::string> index_to_var;
};

static CSRIsing qubo_to_csr_ising(const QUBO& qubo) {
    // --- variable ordering: sorted by string ---
    std::set<std::string> var_set;
    for (auto& kv : qubo.linear)    var_set.insert(kv.first);
    for (auto& kv : qubo.quadratic) {
        var_set.insert(kv.first.first);
        var_set.insert(kv.first.second);
    }
    std::vector<std::string> sorted_vars(var_set.begin(), var_set.end());
    // std::set iteration already sorts by < on string.
    std::unordered_map<std::string, int> var_to_idx;
    var_to_idx.reserve(sorted_vars.size() * 2);
    for (int i = 0; i < (int)sorted_vars.size(); ++i)
        var_to_idx[sorted_vars[i]] = i;

    int size = (int)sorted_vars.size();

    // --- Q_acc: integer accumulation of raw QUBO entries ---
    // Keyed by (i, j).  Off-diagonal only on one side (whichever we inserted).
    std::map<std::pair<int,int>, i128> Q_acc;
    for (auto& kv : qubo.linear) {
        int i = var_to_idx[kv.first];
        Q_acc[{i, i}] += kv.second;
    }
    for (auto& kv : qubo.quadratic) {
        int i = var_to_idx[kv.first.first];
        int j = var_to_idx[kv.first.second];
        Q_acc[{i, j}] += kv.second;
    }

    // --- Q_sym = (Q + Q^T) / 2  as doubles ---
    std::map<std::pair<int,int>, double> Q_sym;
    std::set<std::pair<int,int>> visited;
    for (auto& kv : Q_acc) {
        int i = kv.first.first;
        int j = kv.first.second;
        if (visited.count({i, j})) continue;
        if (i == j) {
            Q_sym[{i, i}] = (double)(long double)kv.second;
            visited.insert({i, i});
        } else {
            double vij = (double)(long double)kv.second;
            auto it = Q_acc.find({j, i});
            double vji = (it == Q_acc.end()) ? 0.0 : (double)(long double)it->second;
            double sym = 0.5 * (vij + vji);
            Q_sym[{i, j}] = sym;
            Q_sym[{j, i}] = sym;
            visited.insert({i, j});
            visited.insert({j, i});
        }
    }

    // --- h, neighbors, offset ---
    CSRIsing out;
    out.h.assign(size, 0.0);
    out.offset = 0.0;
    std::vector<std::vector<std::pair<int, double>>> neighbors(size);

    for (auto& kv : Q_sym) {
        int i = kv.first.first;
        int j = kv.first.second;
        double Qij = kv.second;
        if (i == j) {
            out.h[i]    += 0.5  * Qij;
            out.offset  += 0.5  * Qij;
        } else {
            double Jij = Qij / 2.0;
            neighbors[i].push_back({j, Jij});
            out.h[i]   += 0.5  * Qij;
            out.offset += 0.25 * Qij;
        }
    }

    // --- CSR construction ---
    out.row_ptr.assign(size + 1, 0);
    out.col_idx.clear();
    out.values.clear();
    int nnz = 0;
    for (int i = 0; i < size; ++i) {
        out.row_ptr[i] = nnz;
        auto& row = neighbors[i];
        std::sort(row.begin(), row.end(),
                  [](const std::pair<int,double>& a, const std::pair<int,double>& b) {
                      return a.first < b.first;
                  });
        for (auto& p : row) {
            out.col_idx.push_back(p.first);
            out.values.push_back(p.second);
            nnz++;
        }
    }
    out.row_ptr[size] = nnz;
    out.index_to_var = std::move(sorted_vars);
    return out;
}

// ============================================================
// Section 9 — File I/O (matches the Python save_csr_and_h format)
// ============================================================

static void write_int_column(const std::string& path, const std::vector<int>& v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    for (int x : v) f << x << "\n";
}

static void write_double_column(const std::string& path, const std::vector<double>& v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    f << std::fixed << std::setprecision(8);
    for (double x : v) f << x << "\n";
}

static void write_double_row(const std::string& path, const std::vector<double>& v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    // Python: pd.DataFrame(h.reshape(1,-1)).to_csv(..., header=False, index=False)
    // produces a single CSV row.  We match with commas and full precision.
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) f << ",";
        f << std::setprecision(17) << v[i];
    }
    f << "\n";
}

static void save_csr_and_h(const CSRIsing& c, const std::string& N_str) {
    write_int_column   ("row_ptr_"  + N_str + ".csv", c.row_ptr);
    write_int_column   ("col_idx_"  + N_str + ".csv", c.col_idx);
    write_double_column("J_values_" + N_str + ".csv", c.values);
    write_double_row   ("h_vector_" + N_str + ".csv", c.h);
}

// ============================================================
// Section 10 — Main
// ============================================================

int main(int argc, char** argv) {
    try {
        std::string N_str = (argc >= 2) ? argv[1] : "159197";
        u128 N = parse_u128(N_str);

        auto t0 = std::chrono::steady_clock::now();

        auto setup = initialize_variables(N);
        std::cout << "Factoring N = " << N_str
                  << " (" << u128_bit_length(N) << " bits)\n";
        std::cout << "Assuming n_p = " << setup.n_p
                  << ", n_q = " << setup.n_q << "\n";

        auto clauses = generate_column_clauses(N, setup.n_p, setup.n_q, setup.s_vars);
        std::cout << "Generated " << clauses.size() << " column clauses\n";

        auto simp = clause_simplifier(std::move(clauses));
        std::cout << "After simplification: "
                  << simp.clauses.size() << " non-zero clauses, "
                  << simp.assignments.size() << " assignments, "
                  << simp.expressions.size() << " expressions, "
                  << simp.mul_constraints.size() << " mul-constraints\n";

        auto H = qubo_hamiltonian_from_clauses(simp.clauses);
        std::cout << "Hamiltonian has " << H.terms.size() << " monomials\n";

        auto qubo = quadrizate(H);
        std::set<std::string> qubo_vars;
        for (auto& kv : qubo.linear)    qubo_vars.insert(kv.first);
        for (auto& kv : qubo.quadratic) {
            qubo_vars.insert(kv.first.first);
            qubo_vars.insert(kv.first.second);
        }
        std::cout << "Quadratized QUBO has "
                  << qubo_vars.size() << " variables, "
                  << qubo.linear.size() << " linear terms, "
                  << qubo.quadratic.size() << " quadratic terms\n";

        auto csr = qubo_to_csr_ising(qubo);
        save_csr_and_h(csr, N_str);

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Time taken to construct CSR J: " << secs << " seconds\n";
        std::cout << "Wrote row_ptr_" << N_str << ".csv, col_idx_"
                  << N_str << ".csv, J_values_" << N_str
                  << ".csv, h_vector_" << N_str << ".csv\n";
        return 0;
    } catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
