/*
 * qubo_factorization.cpp
 *
 * Converts an integer factorization problem (N = P*Q, both prime) into a QUBO.
 * Mirrors the Python pipeline:
 *   initialize_variables -> generate_column_clauses -> clause_simplifier
 *   -> qubo_hamiltonian_from_clauses -> quadratize -> output CSR + metadata
 *
 * Polynomial representation:
 *   A Poly is unordered_map<Monomial, int64_t>
 *   A Monomial is a sorted vector<int> of variable indices (binary: x^2=x, so no duplicates)
 *   The constant term uses an empty vector {} as its monomial key.
 *
 * Variable naming/indexing:
 *   p_i  -> "p_i"
 *   q_i  -> "q_i"
 *   s_i_j -> "s_i_j"
 *   Auxiliary quadratization vars -> "w_n", "w2_n"
 *
 * Build:
 *   g++ -O2 -std=c++17 -o qubo_factorization qubo_factorization.cpp
 *
 * Run:
 *   ./qubo_factorization <N>
 *   e.g.  ./qubo_factorization 34447
 *
 * Output files (same as Python version):
 *   row_ptr_N.csv, col_idx_N.csv, J_values_N.csv, h_vector_N.csv
 *   index_to_var_N.txt, assignment_constraints_N.txt, expression_constraints_N.txt
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// Variable registry  (string name <-> int index)
// ============================================================
struct VarRegistry {
    std::unordered_map<std::string, int> nameToIdx;
    std::vector<std::string>             idxToName;

    int get(const std::string& name) {
        auto it = nameToIdx.find(name);
        if (it != nameToIdx.end()) return it->second;
        int idx = (int)idxToName.size();
        nameToIdx[name] = idx;
        idxToName.push_back(name);
        return idx;
    }
    const std::string& name(int idx) const { return idxToName[idx]; }
    int size() const { return (int)idxToName.size(); }
};

VarRegistry G_vars;  // global registry

// ============================================================
// Monomial = sorted list of distinct variable indices
// ============================================================
using Monomial = std::vector<int>;

struct MonomialHash {
    size_t operator()(const Monomial& m) const {
        size_t h = 0;
        for (int v : m) h ^= std::hash<int>()(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================
// Polynomial = map from Monomial -> int64_t coefficient
// ============================================================
using Poly = std::unordered_map<Monomial, int64_t, MonomialHash>;

// --- helpers ---
static const Monomial CONST_MON = {};  // empty = constant term

Poly polyConst(int64_t c) {
    if (c == 0) return {};
    return {{CONST_MON, c}};
}

Poly polyVar(int idx) {
    return {{{idx}, 1}};
}

// multiply two monomials (binary: merge sorted, drop duplicates)
Monomial mulMon(const Monomial& a, const Monomial& b) {
    Monomial res;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
    return res;
}

void addTerm(Poly& p, const Monomial& m, int64_t c) {
    if (c == 0) return;
    p[m] += c;
    if (p[m] == 0) p.erase(m);
}

Poly polyAdd(const Poly& a, const Poly& b) {
    Poly res = a;
    for (auto& [m, c] : b) addTerm(res, m, c);
    return res;
}

Poly polySub(const Poly& a, const Poly& b) {
    Poly res = a;
    for (auto& [m, c] : b) addTerm(res, m, -c);
    return res;
}

Poly polyScale(const Poly& a, int64_t s) {
    if (s == 0) return {};
    Poly res;
    for (auto& [m, c] : a) res[m] = c * s;
    return res;
}

Poly polyMul(const Poly& a, const Poly& b) {
    Poly res;
    for (auto& [ma, ca] : a)
        for (auto& [mb, cb] : b)
            addTerm(res, mulMon(ma, mb), ca * cb);
    return res;
}

// substitute variable idx -> value (0 or 1) in a polynomial
Poly polySub1(const Poly& p, int idx, int val) {
    Poly res;
    for (auto& [m, c] : p) {
        bool has = std::binary_search(m.begin(), m.end(), idx);
        if (!has) {
            addTerm(res, m, c);
        } else {
            // remove idx from monomial
            Monomial nm;
            for (int v : m) if (v != idx) nm.push_back(v);
            addTerm(res, nm, c * val);
        }
    }
    return res;
}

// substitute variable idx -> another Poly expression
Poly polySubExpr(const Poly& p, int idx, const Poly& expr) {
    Poly res;
    for (auto& [m, c] : p) {
        bool has = std::binary_search(m.begin(), m.end(), idx);
        if (!has) {
            addTerm(res, m, c);
        } else {
            // monomial without idx
            Monomial nm;
            for (int v : m) if (v != idx) nm.push_back(v);
            // multiply polyVar(nm) * expr * c   (binary: nm is already product of other vars)
            // we need to form Poly({nm->1}) * expr
            Poly base;
            base[nm] = 1;
            Poly term = polyScale(polyMul(base, expr), c);
            res = polyAdd(res, term);
        }
    }
    return res;
}

// apply binary rule: x^n = x  (terms with duplicate vars collapse to lower degree)
// Actually our mulMon already handles this with set_union, so polynomials stay reduced.
// This is a no-op here but kept for clarity.
Poly applyPowerRule(const Poly& p) { return p; }

int64_t getConst(const Poly& p) {
    auto it = p.find(CONST_MON);
    return it != p.end() ? it->second : 0;
}

int64_t getCoeff(const Poly& p, int idx) {
    Monomial m = {idx};
    auto it = p.find(m);
    return it != p.end() ? it->second : 0;
}

bool isZero(const Poly& p) { return p.empty(); }

int64_t gcdAbs(int64_t a, int64_t b) {
    a = std::abs(a); b = std::abs(b);
    while (b) { a %= b; std::swap(a, b); }
    return a ? a : 1;
}

int64_t polyGCD(const Poly& p) {
    int64_t g = 0;
    for (auto& [m, c] : p) g = gcdAbs(g, c);
    return g == 0 ? 1 : g;
}

Poly polyDivideConst(const Poly& p, int64_t d) {
    Poly res;
    for (auto& [m, c] : p) res[m] = c / d;
    return res;
}

// free variables (non-constant) in poly
std::set<int> freeVars(const Poly& p) {
    std::set<int> vs;
    for (auto& [m, c] : p)
        for (int v : m) vs.insert(v);
    return vs;
}

// print a poly for debugging
std::string polyStr(const Poly& p) {
    if (p.empty()) return "0";
    std::string s;
    for (auto& [m, c] : p) {
        if (!s.empty()) s += " + ";
        s += std::to_string(c);
        for (int v : m) s += "*" + G_vars.name(v);
    }
    return s;
}

// ============================================================
// Constraint types
// ============================================================

// A "value constraint": var = integer (0 or 1)
struct ValConstraint {
    std::string varName;
    int value;  // 0 or 1
};

// An "expression constraint": var = Poly expression
struct ExprConstraint {
    std::string varName;
    Poly        expr;
};

// ============================================================
// Initialize variables
// ============================================================
struct ProblemVars {
    int n_p, n_q;
    // fixed vars: p[0]=1, p[n_p-1]=1, q[0]=1, q[n_q-1]=1
    // free vars registered in G_vars
    std::unordered_map<int, Poly> p;  // bit index -> Poly (constant 1 or single var)
    std::unordered_map<int, Poly> q;
    std::map<std::pair<int,int>, Poly> s;  // (from_col, to_col) -> Poly
};

ProblemVars initializeVariables(uint64_t N) {
    ProblemVars pv;
    int n_m = (int)std::ceil(std::log2((double)N + 1));
    pv.n_q  = (int)std::ceil(0.5 * std::log2((double)N));
    pv.n_p  = pv.n_q;

    std::cout << "Factoring N = " << N << " (" << n_m << " bits)\n";
    std::cout << "Assuming n_p = " << pv.n_p << ", n_q = " << pv.n_q << "\n";

    // p variables
    pv.p[0] = polyConst(1);
    pv.p[pv.n_p - 1] = polyConst(1);
    for (int i = 1; i < pv.n_p - 1; i++) {
        std::string nm = "p_" + std::to_string(i);
        pv.p[i] = polyVar(G_vars.get(nm));
    }
    // If n_p == 1, only index 0 exists (already set)
    // If n_p == 2, index 0 and 1 are both fixed
    if (pv.n_p == 1) { /* nothing extra */ }

    // q variables
    pv.q[0] = polyConst(1);
    pv.q[pv.n_q - 1] = polyConst(1);
    for (int i = 1; i < pv.n_q - 1; i++) {
        std::string nm = "q_" + std::to_string(i);
        pv.q[i] = polyVar(G_vars.get(nm));
    }

    // carry variables s[col_from][col_to]
    for (int i = 1; i < pv.n_p + pv.n_q; i++) {
        int num_prod = std::min(i, pv.n_q - 1) - std::max(i - pv.n_p + 1, 0) + 1;
        int max_sum  = num_prod + (i - 1);
        if (max_sum > 1) {
            int bits = (int)std::floor(std::log2((double)max_sum));
            for (int j = 1; j <= bits; j++) {
                if (i + j < pv.n_p + pv.n_q) {
                    std::string nm = "s_" + std::to_string(i) + "_" + std::to_string(i + j);
                    pv.s[{i, i + j}] = polyVar(G_vars.get(nm));
                }
            }
        }
    }
    return pv;
}
