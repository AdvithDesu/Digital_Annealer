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
