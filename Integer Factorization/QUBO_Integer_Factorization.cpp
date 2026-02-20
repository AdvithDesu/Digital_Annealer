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

// ============================================================
// Generate column clauses
// ============================================================
std::vector<Poly> generateColumnClauses(uint64_t N, const ProblemVars& pv) {
    int n_m = pv.n_p + pv.n_q;
    // get bits of N, LSB first
    std::vector<int> Nbits(n_m, 0);
    for (int i = 0; i < n_m; i++) Nbits[i] = (int)((N >> i) & 1);

    std::vector<Poly> clauses;
    for (int i = 1; i < n_m; i++) {
        Poly clause = polyConst(Nbits[i]);

        // subtract product terms
        for (int j = 0; j < pv.n_p; j++) {
            int qidx = i - j;
            if (qidx >= 0 && qidx < pv.n_q) {
                auto pit = pv.p.find(j);
                auto qit = pv.q.find(qidx);
                if (pit != pv.p.end() && qit != pv.q.end()) {
                    Poly pterm = polyMul(pit->second, qit->second);
                    clause = polySub(clause, pterm);
                }
            }
        }

        // subtract input carries
        for (int k = 1; k < i; k++) {
            auto it = pv.s.find({k, i});
            if (it != pv.s.end() && !isZero(it->second))
                clause = polySub(clause, it->second);
        }

        // add output carries
        for (int j = 1; ; j++) {
            auto it = pv.s.find({i, i + j});
            if (it == pv.s.end() || isZero(it->second)) break;
            int64_t scale = (int64_t)1 << j;
            clause = polyAdd(clause, polyScale(it->second, scale));
        }

        clauses.push_back(clause);
    }
    return clauses;
}

// ============================================================
// Constraint application rules
// ============================================================

// Helper: is this Poly a single variable (degree-1 monomial with coeff +-1)?
bool isSingleVar(const Poly& p, int& outIdx, int64_t& outCoeff) {
    if (p.size() != 1) return false;
    auto& [m, c] = *p.begin();
    if (m.size() == 1 && (c == 1 || c == -1)) {
        outIdx   = m[0];
        outCoeff = c;
        return true;
    }
    return false;
}

// Rule 1 & 2: all vars same sign and sum = const matches count -> fix all to 0 or 1
// Returns map var->value, empty if rule doesn't apply
std::unordered_map<int,int> applyRule12(const Poly& clause) {
    std::unordered_map<int,int> res;
    if (isZero(clause)) return res;

    int64_t constTerm = getConst(clause);

    // collect linear terms only (degree 1)
    std::vector<std::pair<int,int64_t>> linTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1) linTerms.push_back({m[0], c});
        else return res;  // has higher degree terms, rule doesn't apply
    }

    if (linTerms.empty()) return res;
    int64_t n = (int64_t)linTerms.size();

    // all coeff == -1 and const == n  -> all vars = 1
    bool allNeg = std::all_of(linTerms.begin(), linTerms.end(), [](auto& p){ return p.second == -1; });
    if (allNeg && constTerm == n) {
        for (auto& [v, c] : linTerms) res[v] = 1;
        return res;
    }
    // all coeff == +1 and const == -n -> all vars = 1
    bool allPos = std::all_of(linTerms.begin(), linTerms.end(), [](auto& p){ return p.second == 1; });
    if (allPos && constTerm == -n) {
        for (auto& [v, c] : linTerms) res[v] = 1;
        return res;
    }
    // all coeff == +1 and const == 0  -> all vars = 0
    if (allPos && constTerm == 0) {
        for (auto& [v, c] : linTerms) res[v] = 0;
        return res;
    }
    // all coeff == -1 and const == 0  -> all vars = 0
    if (allNeg && constTerm == 0) {
        for (auto& [v, c] : linTerms) res[v] = 0;
        return res;
    }
    return res;
}

// Rule 4: coefficient dominance
// If |c_i| > sum of all other |coefficients| (including const), var is determined
std::unordered_map<int,int> applyRule4(const Poly& clause) {
    std::unordered_map<int,int> res;
    if (isZero(clause)) return res;

    int64_t constTerm = getConst(clause);

    // collect linear terms only
    std::vector<std::pair<int,int64_t>> linTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1) linTerms.push_back({m[0], c});
        else return res;
    }

    // separate positive and negative
    int64_t posSum = 0, negSum = 0;
    for (auto& [v, c] : linTerms) {
        if (c > 0) posSum += c;
        else       negSum += (-c);
    }

    for (auto& [v, c] : linTerms) {
        if (c > 0) {
            if (c > negSum - constTerm) res[v] = 0;
            if (c > posSum + constTerm) res[v] = 1;
        } else {
            int64_t ac = -c;
            if (ac > posSum + constTerm) res[v] = 0;
            if (ac > negSum - constTerm) res[v] = 1;
        }
    }
    return res;
}

// Rule 3: clause has exactly 3 linear terms that match x1 + x2 - 2*x3 = 0
// Returns {var_to_eliminate -> expression} plus mul_ass (simplified here)
// We try to find s variable to substitute
std::unordered_map<int,Poly> applyRule3(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    int64_t constTerm = getConst(clause);
    if (constTerm != 0) return res;

    std::vector<std::pair<int,int64_t>> linTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1) linTerms.push_back({m[0], c});
        else return res;
    }
    if (linTerms.size() != 3) return res;

    // find the one with coeff +-2
    int twoIdx = -1;
    for (int i = 0; i < 3; i++)
        if (std::abs(linTerms[i].second) == 2) { twoIdx = i; break; }
    if (twoIdx < 0) return res;

    int64_t twoSign  = (linTerms[twoIdx].second > 0) ? 1 : -1;
    int64_t restSign = -twoSign;  // the other two should sum to twoSign * x3

    // check other two have coeff restSign
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        if (i == twoIdx) continue;
        if (linTerms[i].second != restSign) { ok = false; break; }
    }
    if (!ok) return res;

    // x1 + x2 = 2*x3 (up to sign)
    // Substitute x3 = (x1 + x2) / 2  -- not integer, so instead substitute
    // the s_ variable if present
    int v3 = linTerms[twoIdx].first;
    int v1 = linTerms[(twoIdx+1)%3].first;
    int v2 = linTerms[(twoIdx+2)%3].first;

    // x3 = x1  (since binary, x1=x2=x3 follows)
    // prefer to eliminate s_ variable
    std::string n3 = G_vars.name(v3), n1 = G_vars.name(v1), n2 = G_vars.name(v2);

    auto hasS = [](const std::string& s){ return s.rfind("s_",0)==0; };

    if (hasS(n3)) {
        res[v3] = polyVar(v1);  // s = x1 (and x1=x2 implied, but we only do one sub)
    } else if (hasS(n1)) {
        res[v1] = polyVar(v3);
    } else if (hasS(n2)) {
        res[v2] = polyVar(v3);
    } else {
        res[v3] = polyVar(v1);
    }
    return res;
}

// Rule 6: two large positive (or negative) terms dominate -> complementary
std::unordered_map<int,Poly> applyRule6(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    int64_t constTerm = getConst(clause);
    std::vector<std::pair<int,int64_t>> linTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1) linTerms.push_back({m[0], c});
        else return res;
    }

    std::vector<std::pair<int,int64_t>> pos, neg;
    for (auto& kv : linTerms) {
        if (kv.second > 0) pos.push_back(kv);
        else               neg.push_back({kv.first, -kv.second});
    }
    std::sort(pos.begin(), pos.end(), [](auto& a, auto& b){ return a.second > b.second; });
    std::sort(neg.begin(), neg.end(), [](auto& a, auto& b){ return a.second > b.second; });

    int64_t posSum = 0, negSum = 0;
    for (auto& kv : pos) posSum += kv.second;
    for (auto& kv : neg) negSum += kv.second;

    auto hasS = [](const std::string& s){ return s.rfind("s_",0)==0; };

    // positive dominance case
    if (posSum > negSum - constTerm && constTerm < 0 && pos.size() >= 2) {
        int64_t cy = pos[0].second, cx = pos[1].second;
        if (cy + cx > negSum - constTerm && -constTerm - posSum + cy + cx > 0) {
            int vy = pos[0].first, vx = pos[1].first;
            // y = 1 - x
            Poly oneMinusX = polyAdd(polyConst(1), polyScale(polyVar(vx), -1));
            if (hasS(G_vars.name(vy)))
                res[vy] = oneMinusX;
            else
                res[vy] = oneMinusX;
        }
    }
    // negative dominance case
    if (negSum > posSum + constTerm && constTerm > 0 && neg.size() >= 2) {
        int64_t cy = neg[0].second, cx = neg[1].second;
        if (cy + cx > posSum + constTerm && negSum - constTerm - cy - cx < 0) {
            int vy = neg[0].first, vx = neg[1].first;
            Poly oneMinusX = polyAdd(polyConst(1), polyScale(polyVar(vx), -1));
            if (hasS(G_vars.name(vy)))
                res[vy] = oneMinusX;
            else
                res[vy] = oneMinusX;
        }
    }
    return res;
}

// Parity rule: find pairs of degree-1 terms with odd coefficients
std::unordered_map<int,Poly> applyParityRule(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    std::vector<std::pair<int,int64_t>> oddTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1 && (std::abs(c) % 2 != 0))
            oddTerms.push_back({m[0], (c % 2 + 2) % 2});  // reduce to +1
    }
    int64_t oddConst = (int64_t)(getConst(clause) % 2);
    if (oddConst < 0) oddConst += 2;

    if (oddTerms.size() == 2) {
        int v1 = oddTerms[0].first, v2 = oddTerms[1].first;
        std::string n1 = G_vars.name(v1), n2 = G_vars.name(v2);
        auto hasS = [](const std::string& s){ return s.rfind("s_",0)==0; };
        if (oddConst == 0) {
            // v1 = v2
            if (hasS(n1)) res[v1] = polyVar(v2);
            else if (hasS(n2)) res[v2] = polyVar(v1);
            else res[v1] = polyVar(v2);
        } else {
            // v1 = 1 - v2
            Poly oneMinusV2 = polyAdd(polyConst(1), polyScale(polyVar(v2), -1));
            if (hasS(n1)) res[v1] = oneMinusV2;
            else if (hasS(n2)) res[v2] = polyAdd(polyConst(1), polyScale(polyVar(v1), -1));
            else res[v1] = oneMinusV2;
        }
    }
    return res;
}

// Replacement rule: find a clause where one s_ var can be isolated
std::unordered_map<int,Poly> applyReplacement(const std::vector<Poly>& clauses) {
    std::unordered_map<int,Poly> res;
    for (auto& clause : clauses) {
        if (isZero(clause)) continue;
        for (auto& [m, c] : clause) {
            if (m.size() != 1) continue;
            int idx = m[0];
            if (c != 1 && c != -1) continue;
            std::string nm = G_vars.name(idx);
            if (nm.rfind("s_", 0) != 0) continue;
            // expr = -(clause - c*var) / c
            Poly rest;
            for (auto& [m2, c2] : clause) {
                if (m2 == m) continue;
                rest[m2] = c2;
            }
            // var = -rest / c
            Poly val = polyScale(rest, -c);
            res[idx] = val;
            return res;
        }
    }
    return res;
}

// Apply a value substitution to all clauses
void applyValueSub(std::vector<Poly>& clauses, int varIdx, int value) {
    for (auto& c : clauses)
        c = polySub1(c, varIdx, value);
}

// Apply an expression substitution to all clauses
void applyExprSub(std::vector<Poly>& clauses, int varIdx, const Poly& expr) {
    for (auto& c : clauses)
        c = polySubExpr(c, varIdx, expr);
}

// ============================================================
// Clause simplifier (mirrors clause_simplifier in Python)
// ============================================================
struct SimplifierResult {
    std::vector<Poly> clauses;
    // assignment constraints: var name -> value (0 or 1)
    std::vector<std::pair<std::string,int>>   assignmentConstraints;
    // expression constraints: var name -> Poly expression
    std::vector<std::pair<std::string,Poly>>  expressionConstraints;
    // mul constraints (simplified: just expression constraints for now)
};

SimplifierResult clauseSimplifier(std::vector<Poly> clauses) {
    SimplifierResult result;
    result.clauses = std::move(clauses);

    int maxIter = 2 * (int)result.clauses.size();
    std::cout << "Total iterations possible: " << maxIter << "\n";

    for (int iter = 0; iter < maxIter; iter++) {
        bool changed = false;

        // --- Divide by GCD ---
        for (auto& c : result.clauses) {
            if (!isZero(c)) {
                int64_t g = polyGCD(c);
                if (g > 1) c = polyDivideConst(c, g);
            }
        }

        // --- Rule 1 & 2 ---
        for (auto& clause : result.clauses) {
            auto cons = applyRule12(clause);
            if (!cons.empty()) {
                for (auto& [v, val] : cons) {
                    std::string nm = G_vars.name(v);
                    result.assignmentConstraints.push_back({nm, val});
                    applyValueSub(result.clauses, v, val);
                }
                changed = true;
                break;
            }
        }

        // --- Rule 4 ---
        for (auto& clause : result.clauses) {
            auto cons = applyRule4(clause);
            if (!cons.empty()) {
                for (auto& [v, val] : cons) {
                    std::string nm = G_vars.name(v);
                    result.assignmentConstraints.push_back({nm, val});
                    applyValueSub(result.clauses, v, val);
                }
                changed = true;
                break;
            }
        }

        // --- Rule 3 ---
        for (auto& clause : result.clauses) {
            auto cons = applyRule3(clause);
            if (!cons.empty()) {
                for (auto& [v, expr] : cons) {
                    std::string nm = G_vars.name(v);
                    // check if p or q
                    if (nm[0] == 'p' || nm[0] == 'q')
                        result.expressionConstraints.push_back({nm, expr});
                    applyExprSub(result.clauses, v, expr);
                }
                changed = true;
                break;
            }
        }

        // --- Rule 6 ---
        for (auto& clause : result.clauses) {
            auto cons = applyRule6(clause);
            if (!cons.empty()) {
                for (auto& [v, expr] : cons) {
                    std::string nm = G_vars.name(v);
                    if (nm[0] == 'p' || nm[0] == 'q')
                        result.expressionConstraints.push_back({nm, expr});
                    applyExprSub(result.clauses, v, expr);
                }
                changed = true;
                break;
            }
        }

        // --- Parity rule ---
        for (auto& clause : result.clauses) {
            auto cons = applyParityRule(clause);
            if (!cons.empty()) {
                for (auto& [v, expr] : cons) {
                    std::string nm = G_vars.name(v);
                    if (nm[0] == 'p' || nm[0] == 'q')
                        result.expressionConstraints.push_back({nm, expr});
                    applyExprSub(result.clauses, v, expr);
                }
                changed = true;
                break;
            }
        }

        // --- Replacement ---
        if (!changed) {
            auto cons = applyReplacement(result.clauses);
            if (!cons.empty()) {
                for (auto& [v, expr] : cons) {
                    std::string nm = G_vars.name(v);
                    if (nm[0] == 'p' || nm[0] == 'q')
                        result.expressionConstraints.push_back({nm, expr});
                    applyExprSub(result.clauses, v, expr);
                }
                changed = true;
            } else {
                break;  // no rule fired, done
            }
        }
    }

    result.clauses = result.clauses;
    return result;
}

// ============================================================
// Build QUBO: H = sum_i clause_i^2
// qubo[(i,j)] += coeff  for i<=j  (upper triangular)
// ============================================================
using QUBODict = std::map<std::pair<int,int>, double>;

// get or create a fresh auxiliary variable index
static int g_auxCounter = 0;
int newAuxVar(const std::string& prefix) {
    std::string nm = prefix + std::to_string(g_auxCounter++);
    return G_vars.get(nm);
}

// The QUBO is built directly from squaring polynomials.
// For binary vars: x*x = x.  Our mulMon already handles this.
QUBODict buildQUBO(const std::vector<Poly>& clauses, double& offset) {
    // First, collect all monomials from H = sum clause^2
    // We expand clause^2 and accumulate into a single Poly
    Poly H;
    for (auto& clause : clauses) {
        if (isZero(clause)) continue;
        Poly sq = polyMul(clause, clause);
        for (auto& [m, c] : sq) addTerm(H, m, c);
    }

    // Now H is a polynomial in binary variables.
    // Degree > 2 terms need quadratization.
    // We apply simple substitution: for a degree-3 monomial xyz, introduce w=xy (penalty: w(w-1) = 0 for binary... but we use QUBO penalty)
    // Simple iterative approach: find any monomial of degree > 2, reduce it.

    // Quadratize: replace each degree-3+ monomial with auxiliary vars
    // We repeatedly reduce until all monomials are degree <= 2.
    bool reduced = false;
    int quadIter = 0;
    while (!reduced && quadIter < 10000) {
        reduced = true;
        quadIter++;
        Poly newH;
        for (auto& [m, c] : H) {
            if (m.size() <= 2) {
                addTerm(newH, m, c);
                continue;
            }
            reduced = false;
            // take first two vars in monomial
            int va = m[0], vb = m[1];
            // introduce w = va * vb
            // find or create auxiliary
            std::string aname = "w_" + G_vars.name(va) + "_" + G_vars.name(vb);
            int w = G_vars.get(aname);

            // replace va*vb with w in this monomial
            Monomial newM;
            newM.push_back(w);
            for (int i = 2; i < (int)m.size(); i++) newM.push_back(m[i]);
            std::sort(newM.begin(), newM.end());
            addTerm(newH, newM, c);

            // Add penalty: lambda*(w - 2*w*va - 2*w*vb + va*vb + 3*w)
            // Standard QUBO penalty for w = va*vb:
            // P = lambda * (3w + va*vb - 2*w*va - 2*w*vb)
            // We pick lambda = |c| + 1 to dominate
            int64_t lam = std::abs(c) + 1;
            addTerm(newH, {w},        3 * lam);
            addTerm(newH, {va, vb},   lam);
            addTerm(newH, {w, va},   -2 * lam);
            addTerm(newH, {w, vb},   -2 * lam);
        }
        H = newH;
    }

    // Now convert to QUBO dict (upper triangular)
    QUBODict Q;
    offset = 0.0;
    for (auto& [m, c] : H) {
        if (m.empty()) { offset += (double)c; continue; }
        if (m.size() == 1) {
            auto key = std::make_pair(m[0], m[0]);
            Q[key] += (double)c;
        } else {
            // degree 2
            int i = m[0], j = m[1];
            if (i > j) std::swap(i, j);
            auto key = std::make_pair(i, j);
            Q[key] += (double)c;
        }
    }
    return Q;
}

// ============================================================
// Convert QUBO dict to Ising (h, J) and CSR
// ============================================================
struct IsingCSR {
    std::vector<int>    row_ptr;
    std::vector<int>    col_idx;
    std::vector<double> values;
    std::vector<double> h;
    double              offset;
};

IsingCSR quboToIsingCSR(const QUBODict& Q, int numVars) {
    // symmetrize: Q_sym(i,j) = (Q(i,j) + Q(j,i)) / 2
    std::map<std::pair<int,int>, double> Qsym;
    for (auto& [key, val] : Q) {
        auto [i, j] = key;
        if (i == j) {
            Qsym[{i,i}] += val;
        } else {
            Qsym[{i,j}] += val / 2.0;
            Qsym[{j,i}] += val / 2.0;
        }
    }

    IsingCSR ising;
    ising.h.assign(numVars, 0.0);
    ising.offset = 0.0;

    // Build adjacency list
    std::vector<std::vector<std::pair<int,double>>> adj(numVars);
    for (auto& [key, val] : Qsym) {
        auto [i, j] = key;
        if (i == j) {
            ising.h[i] += 0.5 * val;
            ising.offset += 0.5 * val;
        } else {
            double Jij = val / 2.0;
            adj[i].push_back({j, Jij});
            ising.h[i] += 0.5 * val;
            ising.offset += 0.25 * val;
        }
    }

    // Build CSR
    ising.row_ptr.resize(numVars + 1, 0);
    int nnz = 0;
    for (int i = 0; i < numVars; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        ising.row_ptr[i] = nnz;
        for (auto& [j, v] : adj[i]) {
            ising.col_idx.push_back(j);
            ising.values.push_back(v);
            nnz++;
        }
    }
    ising.row_ptr[numVars] = nnz;
    return ising;
}

// ============================================================
// Output / save
// ============================================================
void saveCSV(const std::string& filename, const std::vector<int>& v) {
    std::ofstream f(filename);
    for (int x : v) f << x << "\n";
}
void saveCSV(const std::string& filename, const std::vector<double>& v) {
    std::ofstream f(filename);
    f.precision(8); f << std::fixed;
    for (double x : v) f << x << "\n";
}
void saveHVector(const std::string& filename, const std::vector<double>& h) {
    std::ofstream f(filename);
    f.precision(8); f << std::fixed;
    for (int i = 0; i < (int)h.size(); i++) {
        if (i) f << ",";
        f << h[i];
    }
    f << "\n";
}

void saveIndexToVar(const std::string& filename,
                    const std::vector<std::string>& activeVars,
                    const std::vector<int>& activeIdx) {
    std::ofstream f(filename);
    for (int i = 0; i < (int)activeVars.size(); i++)
        f << i << " " << activeVars[i] << " (global_idx=" << activeIdx[i] << ")\n";
}

void saveAssignmentConstraints(const std::string& filename,
    const std::vector<std::pair<std::string,int>>& ass) {
    std::ofstream f(filename);
    for (auto& [nm, val] : ass)
        f << nm << " = " << val << "\n";
}

void saveExpressionConstraints(const std::string& filename,
    const std::vector<std::pair<std::string,Poly>>& expr) {
    std::ofstream f(filename);
    for (auto& [nm, p] : expr)
        f << nm << " = " << polyStr(p) << "\n";
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N>\n";
        return 1;
    }

    uint64_t N = std::stoull(argv[1]);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: initialize variables
    ProblemVars pv = initializeVariables(N);

    // Step 2: generate column clauses
    std::vector<Poly> clauses = generateColumnClauses(N, pv);
    std::cout << "Generated " << clauses.size() << " column clauses\n";

    // Step 3: simplify
    SimplifierResult sr = clauseSimplifier(std::move(clauses));
    std::cout << "After simplification: " << sr.assignmentConstraints.size()
              << " assignment constraints, "
              << sr.expressionConstraints.size() << " expression constraints\n";

    // Count non-zero remaining clauses
    int nonZero = 0;
    for (auto& c : sr.clauses) if (!isZero(c)) nonZero++;
    std::cout << "Non-zero remaining clauses: " << nonZero << "\n";

    // Step 4: build QUBO
    double offset = 0.0;
    QUBODict Q = buildQUBO(sr.clauses, offset);
    std::cout << "QUBO has " << Q.size() << " entries (offset=" << offset << ")\n";

    // Collect active variables (those appearing in Q)
    std::set<int> activeSet;
    for (auto& [key, val] : Q) {
        activeSet.insert(key.first);
        activeSet.insert(key.second);
    }
    std::vector<int> activeIdx(activeSet.begin(), activeSet.end());
    std::sort(activeIdx.begin(), activeIdx.end());

    // Remap to contiguous 0..n-1
    std::unordered_map<int,int> remap;
    std::vector<std::string> activeVarNames;
    for (int i = 0; i < (int)activeIdx.size(); i++) {
        remap[activeIdx[i]] = i;
        activeVarNames.push_back(G_vars.name(activeIdx[i]));
    }
    int numVars = (int)activeIdx.size();
    std::cout << "Active QUBO variables: " << numVars << "\n";

    // Remap Q keys
    QUBODict Qremapped;
    for (auto& [key, val] : Q) {
        int ri = remap[key.first], rj = remap[key.second];
        if (ri > rj) std::swap(ri, rj);
        Qremapped[{ri, rj}] += val;
    }

    // Step 5: convert to Ising CSR
    IsingCSR ising = quboToIsingCSR(Qremapped, numVars);
    ising.offset += offset;

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Total construction time: " << elapsed << " seconds\n";

    // Step 6: save outputs
    std::string Nstr = std::to_string(N);
    saveCSV("row_ptr_" + Nstr + ".csv",    ising.row_ptr);
    saveCSV("col_idx_" + Nstr + ".csv",    ising.col_idx);
    saveCSV("J_values_" + Nstr + ".csv",   ising.values);
    saveHVector("h_vector_" + Nstr + ".csv", ising.h);
    saveIndexToVar("index_to_var_" + Nstr + ".txt", activeVarNames, activeIdx);
    saveAssignmentConstraints("assignment_constraints_" + Nstr + ".txt", sr.assignmentConstraints);
    saveExpressionConstraints("expression_constraints_" + Nstr + ".txt", sr.expressionConstraints);

    // Print QUBO dict summary (first 20 entries)
    std::cout << "\nQUBO dict (first 20 non-zero entries):\n";
    int cnt = 0;
    for (auto& [key, val] : Qremapped) {
        if (cnt++ >= 20) break;
        std::cout << "  (" << key.first << "," << key.second << ") -> " << val
                  << "  [" << activeVarNames[key.first] << " x " << activeVarNames[key.second] << "]\n";
    }

    std::cout << "\nDone. Output files written with prefix *_" << Nstr << ".*\n";
    return 0;
}
