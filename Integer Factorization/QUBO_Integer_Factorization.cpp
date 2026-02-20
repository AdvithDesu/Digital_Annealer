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
