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
