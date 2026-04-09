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
// 128-bit unsigned integer support
// ============================================================
using uint128_t = __uint128_t;
using int128_t  = __int128_t;

// Signed 128-bit helpers
std::string int128ToString(int128_t n) {
    if (n == 0) return "0";
    std::string s;
    bool neg = (n < 0);
    uint128_t u = neg ? (uint128_t)(-n) : (uint128_t)n;
    while (u > 0) { s += '0' + (char)(u % 10); u /= 10; }
    if (neg) s += '-';
    std::reverse(s.begin(), s.end());
    return s;
}

std::ostream& operator<<(std::ostream& os, int128_t n) {
    return os << int128ToString(n);
}

int128_t abs128(int128_t x) { return x < 0 ? -x : x; }

std::string uint128ToString(uint128_t n) {
    if (n == 0) return "0";
    std::string s;
    while (n > 0) {
        s += '0' + (char)(n % 10);
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

uint128_t parseUint128(const std::string& s) {
    uint128_t result = 0;
    for (char c : s) {
        if (c < '0' || c > '9') break;
        result = result * 10 + (c - '0');
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, uint128_t n) {
    return os << uint128ToString(n);
}

int bitLen128(uint128_t n) {
    if (n == 0) return 0;
    int bits = 0;
    while (n > 0) { n >>= 1; bits++; }
    return bits;
}

// Accurate log2 for 128-bit integers (uses top 53 bits for precision)
double approxLog2_128(uint128_t n) {
    if (n == 0) return -1.0;
    int b = bitLen128(n);
    if (b <= 53) {
        return std::log2((double)(uint64_t)n);
    }
    int shift = b - 53;
    double top = (double)(uint64_t)(n >> shift);
    return (double)shift + std::log2(top);
}

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
// Polynomial = map from Monomial -> int128_t coefficient
// ============================================================
using Poly = std::unordered_map<Monomial, int128_t, MonomialHash>;

// --- helpers ---
static const Monomial CONST_MON = {};  // empty = constant term

Poly polyConst(int128_t c) {
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

void addTerm(Poly& p, const Monomial& m, int128_t c) {
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

Poly polyScale(const Poly& a, int128_t s) {
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

int128_t getConst(const Poly& p) {
    auto it = p.find(CONST_MON);
    return it != p.end() ? it->second : 0;
}

int128_t getCoeff(const Poly& p, int idx) {
    Monomial m = {idx};
    auto it = p.find(m);
    return it != p.end() ? it->second : 0;
}

bool isZero(const Poly& p) { return p.empty(); }

int128_t gcdAbs(int128_t a, int128_t b) {
    a = abs128(a); b = abs128(b);
    while (b) { a %= b; std::swap(a, b); }
    return a ? a : 1;
}

int128_t polyGCD(const Poly& p) {
    int128_t g = 0;
    for (auto& [m, c] : p) g = gcdAbs(g, c);
    return g == 0 ? 1 : g;
}

Poly polyDivideConst(const Poly& p, int128_t d) {
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
        s += int128ToString(c);
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

ProblemVars initializeVariables(uint128_t N) {
    ProblemVars pv;
    int n_m = bitLen128(N);
    pv.n_q  = (n_m + 1) / 2;
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
std::vector<Poly> generateColumnClauses(uint128_t N, const ProblemVars& pv) {
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
            int128_t scale = (int128_t)1 << j;
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
bool isSingleVar(const Poly& p, int& outIdx, int128_t& outCoeff) {
    if (p.size() != 1) return false;
    auto& [m, c] = *p.begin();
    if (m.size() == 1 && (c == 1 || c == -1)) {
        outIdx   = m[0];
        outCoeff = c;
        return true;
    }
    return false;
}

// Rule 1 & 2: all terms same sign and sum = const matches count -> fix all to 0 or 1
// Handles ALL terms (linear and product) as binary-valued.
// When a product term is determined to be 1, all constituent vars are set to 1.
std::unordered_map<int,int> applyRule12(const Poly& clause) {
    std::unordered_map<int,int> res;
    if (isZero(clause)) return res;

    int128_t constTerm = getConst(clause);

    // Collect all non-constant terms (any degree)
    struct TermInfo {
        Monomial mon;
        int128_t coeff;
    };
    std::vector<TermInfo> allTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        allTerms.push_back({m, c});
    }

    if (allTerms.empty()) return res;
    int128_t n = (int128_t)allTerms.size();

    // all coeff == -1 and const == n  -> all terms = 1
    bool allNeg = std::all_of(allTerms.begin(), allTerms.end(), [](auto& t){ return t.coeff == -1; });
    if (allNeg && constTerm == n) {
        for (auto& t : allTerms) {
            if (t.mon.size() == 1) res[t.mon[0]] = 1;
            else { for (int v : t.mon) res[v] = 1; } // product=1 => all vars=1
        }
        return res;
    }
    // all coeff == +1 and const == -n -> all terms = 1
    bool allPos = std::all_of(allTerms.begin(), allTerms.end(), [](auto& t){ return t.coeff == 1; });
    if (allPos && constTerm == -n) {
        for (auto& t : allTerms) {
            if (t.mon.size() == 1) res[t.mon[0]] = 1;
            else { for (int v : t.mon) res[v] = 1; }
        }
        return res;
    }
    // all coeff == +1 and const == 0  -> all terms = 0
    // For products = 0, we can only set single vars to 0
    if (allPos && constTerm == 0) {
        for (auto& t : allTerms) {
            if (t.mon.size() == 1) res[t.mon[0]] = 0;
            // product=0 means at least one factor is 0, can't determine which
        }
        return res;
    }
    // all coeff == -1 and const == 0  -> all terms = 0
    if (allNeg && constTerm == 0) {
        for (auto& t : allTerms) {
            if (t.mon.size() == 1) res[t.mon[0]] = 0;
        }
        return res;
    }
    return res;
}

// Rule 4: coefficient dominance
// Treats ALL terms (linear and product) as binary-valued [0,1].
// If |c_i| > sum of opposing |coefficients| + const, the term is forced to 0 or 1.
// Only returns single-variable (degree-1) determinations for substitution.
// When a product term is forced to 1, all its constituent vars are set to 1.
std::unordered_map<int,int> applyRule4(const Poly& clause) {
    std::unordered_map<int,int> res;
    if (isZero(clause)) return res;

    int128_t constTerm = getConst(clause);

    // collect ALL non-constant terms (any degree), each binary-valued
    struct TermInfo {
        Monomial mon;
        int128_t coeff;
    };
    std::vector<TermInfo> allTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        allTerms.push_back({m, c});
    }
    if (allTerms.empty()) return res;

    // separate positive and negative coefficient sums
    int128_t posSum = 0, negSum = 0;
    for (auto& t : allTerms) {
        if (t.coeff > 0) posSum += t.coeff;
        else              negSum += (-t.coeff);
    }

    for (auto& t : allTerms) {
        int128_t c = t.coeff;
        int determined = -1; // -1 = undetermined
        if (c > 0) {
            if (c > negSum - constTerm) determined = 0;
            if (c > posSum + constTerm) determined = 1;
        } else {
            int128_t ac = -c;
            if (ac > posSum + constTerm) determined = 0;
            if (ac > negSum - constTerm) determined = 1;
        }
        if (determined < 0) continue;

        if (t.mon.size() == 1) {
            // single variable
            res[t.mon[0]] = determined;
        } else if (determined == 1) {
            // product = 1 means ALL constituent variables must be 1
            for (int v : t.mon) res[v] = 1;
        }
        // product = 0: can't determine which var is 0, skip
    }
    return res;
}

// Rule 3: clause matches pattern t1 + t2 - 2*t3 = 0 where t_i are binary-valued terms.
// The coefficient-2 term must be a single variable (for substitution).
// The other two terms can be single vars or products.
// Returns {var_to_eliminate -> expression}
std::unordered_map<int,Poly> applyRule3(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    int128_t constTerm = getConst(clause);
    if (constTerm != 0) return res;

    // Collect all non-constant terms
    struct TermInfo {
        Monomial mon;
        int128_t coeff;
    };
    std::vector<TermInfo> terms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        terms.push_back({m, c});
    }
    if (terms.size() != 3) return res;

    // find the one with coeff +-2 (must be degree-1 for substitution)
    int twoIdx = -1;
    for (int i = 0; i < 3; i++) {
        if (abs128(terms[i].coeff) == 2 && terms[i].mon.size() == 1) {
            twoIdx = i;
            break;
        }
    }
    if (twoIdx < 0) return res;

    int128_t twoSign  = (terms[twoIdx].coeff > 0) ? 1 : -1;
    int128_t restSign = -twoSign;

    // check other two have coeff restSign
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        if (i == twoIdx) continue;
        if (terms[i].coeff != restSign) { ok = false; break; }
    }
    if (!ok) return res;

    // Pattern: t1 + t2 = 2*x3 => x3 = t1 (= t2, all equal for binary)
    int v3 = terms[twoIdx].mon[0]; // the degree-1 variable with coeff +-2
    int other1 = (twoIdx + 1) % 3;
    int other2 = (twoIdx + 2) % 3;

    auto hasS = [](const std::string& s){ return s.rfind("s_",0)==0; };
    std::string n3 = G_vars.name(v3);

    // Build the Poly expression for the substitution target
    // Prefer to substitute s_ variables
    // The coefficient-2 variable x3 = t1 (one of the other terms)
    // Pick the simpler term (prefer degree-1 single vars over products)
    auto buildTermPoly = [](const TermInfo& t) -> Poly {
        Poly p;
        p[t.mon] = 1; // the term itself (coeff is already ±1, we use abs value)
        return p;
    };

    // Prefer degree-1 target over product target for cleaner substitution
    int preferIdx = other1;
    if (terms[other1].mon.size() > 1 && terms[other2].mon.size() == 1) {
        preferIdx = other2;
    }

    if (hasS(n3)) {
        // Substitute s_ variable with the preferred other term
        res[v3] = buildTermPoly(terms[preferIdx]);
    } else {
        // The coefficient-2 term is not s_, check if either other term is a single s_ var
        bool s1 = (terms[other1].mon.size() == 1 && hasS(G_vars.name(terms[other1].mon[0])));
        bool s2 = (terms[other2].mon.size() == 1 && hasS(G_vars.name(terms[other2].mon[0])));

        if (s1) {
            // Substitute the s_ variable with the coefficient-2 variable
            res[terms[other1].mon[0]] = polyVar(v3);
        } else if (s2) {
            res[terms[other2].mon[0]] = polyVar(v3);
        } else if (terms[other1].mon.size() == 1 && terms[other2].mon.size() == 1) {
            // All three are single variables, no s_ vars
            // x1 = x2 = x3: substitute one with another
            res[terms[other1].mon[0]] = polyVar(v3);
        } else {
            // Substitute the coefficient-2 var with the preferred term
            res[v3] = buildTermPoly(terms[preferIdx]);
        }
    }
    return res;
}

// Rule 6: two large positive (or negative) terms dominate -> complementary
// Now handles clauses with higher-degree terms (all terms are binary-valued).
// Only produces substitutions involving degree-1 (single variable) terms.
std::unordered_map<int,Poly> applyRule6(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    int128_t constTerm = getConst(clause);

    // Collect ALL non-constant terms with their coefficients
    struct TermInfo {
        Monomial mon;
        int128_t coeff; // absolute value
        int origSign;  // +1 or -1
    };

    std::vector<TermInfo> pos, neg;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (c > 0) pos.push_back({m, c, 1});
        else       neg.push_back({m, -c, -1});
    }

    std::sort(pos.begin(), pos.end(), [](auto& a, auto& b){ return a.coeff > b.coeff; });
    std::sort(neg.begin(), neg.end(), [](auto& a, auto& b){ return a.coeff > b.coeff; });

    int128_t posSum = 0, negSum = 0;
    for (auto& t : pos) posSum += t.coeff;
    for (auto& t : neg) negSum += t.coeff;

    auto hasS = [](const std::string& s){ return s.rfind("s_",0)==0; };

    // positive dominance case: two largest positive terms are complementary
    if (posSum > negSum - constTerm && constTerm < 0 && pos.size() >= 2) {
        for (size_t i = 0; i < pos.size(); i++) {
            for (size_t j = i + 1; j < pos.size(); j++) {
                int128_t cy = pos[i].coeff, cx = pos[j].coeff;
                if (cy + cx > negSum - constTerm && -constTerm - posSum + cy + cx > 0) {
                    // These two terms are complementary. Only substitute if both are degree-1.
                    if (pos[i].mon.size() == 1 && pos[j].mon.size() == 1) {
                        int vy = pos[i].mon[0], vx = pos[j].mon[0];
                        Poly oneMinusX = polyAdd(polyConst(1), polyScale(polyVar(vx), -1));
                        if (hasS(G_vars.name(vy)))
                            res[vy] = oneMinusX;
                        else if (hasS(G_vars.name(vx)))
                            res[vx] = polyAdd(polyConst(1), polyScale(polyVar(vy), -1));
                        else
                            res[vy] = oneMinusX;
                        return res;
                    }
                } else {
                    break; // smaller pairs won't satisfy either
                }
            }
        }
    }

    // negative dominance case
    if (negSum > posSum + constTerm && constTerm > 0 && neg.size() >= 2) {
        for (size_t i = 0; i < neg.size(); i++) {
            for (size_t j = i + 1; j < neg.size(); j++) {
                int128_t cy = neg[i].coeff, cx = neg[j].coeff;
                if (cy + cx > posSum + constTerm && negSum - constTerm - cy - cx < 0) {
                    if (neg[i].mon.size() == 1 && neg[j].mon.size() == 1) {
                        int vy = neg[i].mon[0], vx = neg[j].mon[0];
                        Poly oneMinusX = polyAdd(polyConst(1), polyScale(polyVar(vx), -1));
                        if (hasS(G_vars.name(vy)))
                            res[vy] = oneMinusX;
                        else if (hasS(G_vars.name(vx)))
                            res[vx] = polyAdd(polyConst(1), polyScale(polyVar(vy), -1));
                        else
                            res[vy] = oneMinusX;
                        return res;
                    }
                } else {
                    break;
                }
            }
        }
    }
    return res;
}

// Parity rule: find pairs of degree-1 terms with odd coefficients.
// IMPORTANT: If ANY higher-degree term has an odd coefficient, the parity
// analysis on degree-1 terms alone is invalid, so we must skip.
std::unordered_map<int,Poly> applyParityRule(const Poly& clause) {
    std::unordered_map<int,Poly> res;
    if (isZero(clause)) return res;

    // Check if any higher-degree term has an odd coefficient
    // If so, parity analysis on degree-1 terms alone is incorrect
    for (auto& [m, c] : clause) {
        if (m.size() > 1 && (abs128(c) % 2 != 0))
            return res; // can't apply parity rule safely
    }

    std::vector<std::pair<int,int128_t>> oddTerms;
    for (auto& [m, c] : clause) {
        if (m.empty()) continue;
        if (m.size() == 1 && (abs128(c) % 2 != 0))
            oddTerms.push_back({m[0], ((c % 2) + 2) % 2});  // reduce to +1
    }
    int128_t oddConst = getConst(clause) % 2;
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

    // Backtracking strategy: save state before Replacement starts.
    // Run Replacement unlimited. If it leads to full resolution (all
    // clauses zero), keep the result. If not, the Replacement phase
    // just built a mega-clause without benefit — restore the saved state
    // which has many small clauses and produces a much smaller QUBO.
    bool checkpointSaved = false;
    std::vector<Poly> savedClauses;
    std::vector<std::pair<std::string,int>>  savedAssign;
    std::vector<std::pair<std::string,Poly>> savedExpr;
    int assignsAtCheckpoint = 0;

    for (int iter = 0; iter < maxIter; iter++) {
        bool changed = false;

        // --- Divide by GCD ---
        for (auto& c : result.clauses) {
            if (!isZero(c)) {
                int128_t g = polyGCD(c);
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

        // --- Replacement (cost-gated) ---
        if (!changed) {
            // Save checkpoint before first Replacement
            if (!checkpointSaved) {
                savedClauses = result.clauses;
                savedAssign  = result.assignmentConstraints;
                savedExpr    = result.expressionConstraints;
                assignsAtCheckpoint = (int)savedAssign.size();
                checkpointSaved = true;
            }

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

    // Backtrack check: if Replacement ran but didn't lead to additional
    // value assignments (beyond what we had at the checkpoint), the merging
    // was unproductive — restore the pre-Replacement state.
    if (checkpointSaved) {
        int finalAssigns = (int)result.assignmentConstraints.size();
        if (finalAssigns <= assignsAtCheckpoint) {
            // No new value assignments from Replacement — restore checkpoint
            std::cout << "Replacement unproductive, restoring checkpoint ("
                      << assignsAtCheckpoint << " assigns)\n";
            result.clauses = std::move(savedClauses);
            result.assignmentConstraints = std::move(savedAssign);
            result.expressionConstraints = std::move(savedExpr);
        } else {
            std::cout << "Replacement productive: " << assignsAtCheckpoint
                      << " -> " << finalAssigns << " assigns\n";
        }
    }

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
//
// Quadratization uses exact reduction formulas from the paper:
//   Positive coeff a>0:  a*x1*x2*x3 = min_w a*(w*x3 + x1*x2 - x1*w - x2*w + w)  (eq 9)
//   Negative coeff a<0: -|a|*x1*x2*x3 = min_w -|a|*w*(x1+x2+x3-2)                (eq 11)
//   Positive 4th degree: needs 2 auxiliary vars (recursive application)
//   Negative 4th degree: -|a|*x1*x2*x3*x4 = min_w -|a|*w*(x1+x2+x3+x4-3)
QUBODict buildQUBO(const std::vector<Poly>& clauses, double& offset) {
    // First, collect all monomials from H = sum clause^2
    Poly H;
    for (auto& clause : clauses) {
        if (isZero(clause)) continue;
        Poly sq = polyMul(clause, clause);
        for (auto& [m, c] : sq) addTerm(H, m, c);
    }

    // Collect degree-3 and degree-4 terms, group by unique variable sets
    // (like the Python quadrizate function)
    std::map<Monomial, int128_t> cubicTerms;   // 3-variable monomials
    std::map<Monomial, int128_t> quarticTerms; // 4-variable monomials
    Poly quadraticH; // degree <= 2 terms kept as-is

    for (auto& [m, c] : H) {
        if (m.size() <= 2) {
            addTerm(quadraticH, m, c);
        } else if (m.size() == 3) {
            cubicTerms[m] += c;
        } else if (m.size() == 4) {
            quarticTerms[m] += c;
        }
        // degree > 4 shouldn't occur from squaring degree-2 clauses
    }

    // Diagnostic: count terms by degree
    int deg2count = (int)quadraticH.size();
    int deg3count = 0, deg4count = 0;
    for (auto& [m, c] : cubicTerms) if (c != 0) deg3count++;
    for (auto& [m, c] : quarticTerms) if (c != 0) deg4count++;
    std::cout << "H terms by degree: deg<=2=" << deg2count
              << ", deg3=" << deg3count << ", deg4=" << deg4count << "\n";

    // Apply exact quadratization for cubic terms
    for (auto& [m, coeff] : cubicTerms) {
        if (coeff == 0) continue;
        int a = m[0], b = m[1], c_var = m[2];
        std::string wname = "w_" + std::to_string(g_auxCounter++);
        int w = G_vars.get(wname);

        if (coeff > 0) {
            // Positive: a*x1*x2*x3 = min_w a*(w*x3 + x1*x2 - x1*w - x2*w + w)
            addTerm(quadraticH, {w, c_var},  coeff);   // w*x3
            addTerm(quadraticH, {a, b},      coeff);   // x1*x2
            addTerm(quadraticH, {a, w},     -coeff);   // -x1*w
            addTerm(quadraticH, {b, w},     -coeff);   // -x2*w
            addTerm(quadraticH, {w},         coeff);   // w
        } else {
            // Negative: coeff*x1*x2*x3 = min_w |coeff|*(-w*(x1+x2+x3-2))
            //         = min_w |coeff|*(-w*x1 - w*x2 - w*x3 + 2*w)
            int128_t absC = -coeff;
            addTerm(quadraticH, {w, a},     -absC);    // -w*x1
            addTerm(quadraticH, {w, b},     -absC);    // -w*x2
            addTerm(quadraticH, {w, c_var}, -absC);    // -w*x3
            addTerm(quadraticH, {w},         2*absC);  // 2*w
        }
    }

    // Apply exact quadratization for quartic terms
    for (auto& [m, coeff] : quarticTerms) {
        if (coeff == 0) continue;
        int a = m[0], b = m[1], c_var = m[2], d = m[3];

        if (coeff > 0) {
            // Positive 4th degree: use 2 aux vars (recursive from paper sec 4.2)
            // x1*x2*x3*x4 = min_{w,z} (z*x4 + w*x3 - z*w - z*x3 + z + x1*x2 - w*x1 - w*x2 + w)
            std::string wname = "w_" + std::to_string(g_auxCounter++);
            std::string zname = "w_" + std::to_string(g_auxCounter++);
            int w = G_vars.get(wname);
            int z = G_vars.get(zname);

            addTerm(quadraticH, {z, d},      coeff);   // z*x4
            addTerm(quadraticH, {w, c_var},  coeff);   // w*x3
            addTerm(quadraticH, {z, w},     -coeff);   // -z*w
            addTerm(quadraticH, {z, c_var}, -coeff);   // -z*x3
            addTerm(quadraticH, {z},         coeff);   // z
            addTerm(quadraticH, {a, b},      coeff);   // x1*x2
            addTerm(quadraticH, {w, a},     -coeff);   // -w*x1
            addTerm(quadraticH, {w, b},     -coeff);   // -w*x2
            addTerm(quadraticH, {w},         coeff);   // w
        } else {
            // Negative 4th degree: -|a|*x1*x2*x3*x4 = min_w -|a|*w*(x1+x2+x3+x4-3)
            std::string wname = "w_" + std::to_string(g_auxCounter++);
            int w = G_vars.get(wname);
            int128_t absC = -coeff;

            addTerm(quadraticH, {w, a},     -absC);    // -w*x1
            addTerm(quadraticH, {w, b},     -absC);    // -w*x2
            addTerm(quadraticH, {w, c_var}, -absC);    // -w*x3
            addTerm(quadraticH, {w, d},     -absC);    // -w*x4
            addTerm(quadraticH, {w},         3*absC);  // 3*w
        }
    }

    // Verify all terms are degree <= 2
    for (auto& [m, c] : quadraticH) {
        if (m.size() > 2) {
            // This shouldn't happen for factorization (max degree 4 from squaring)
            std::cerr << "WARNING: degree-" << m.size() << " term after quadratization: ";
            for (int v : m) std::cerr << G_vars.name(v) << "*";
            std::cerr << " coeff=" << c << "\n";
        }
    }

    // Convert to QUBO dict (upper triangular)
    QUBODict Q;
    offset = 0.0;
    for (auto& [m, c] : quadraticH) {
        if (m.empty()) { offset += (double)c; continue; }
        if (m.size() == 1) {
            auto key = std::make_pair(m[0], m[0]);
            Q[key] += (double)c;
        } else if (m.size() == 2) {
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
// Post-processing: given a QUBO solution vector (0/1 per variable),
// reconstruct the factors P and Q using stored constraints.
// ============================================================
void postProcess(const std::vector<int>& quboSolution,
                 uint128_t N, int n_p, int n_q,
                 const SimplifierResult& sr,
                 const std::vector<std::string>& activeVarNames)
{
    std::map<std::string, int64_t> fullAssign;

    // 1. Assignment constraints from preprocessing
    for (auto& [nm, val] : sr.assignmentConstraints)
        fullAssign[nm] = val;

    // 2. QUBO solution bits
    for (int i = 0; i < (int)quboSolution.size(); i++)
        fullAssign[activeVarNames[i]] = quboSolution[i];

    // 3. Expression constraint resolution (pass 1)
    //    Iteratively substitute known values into expressions until no progress.
    int maxPasses = (int)sr.expressionConstraints.size() + 5;
    for (int pass = 0; pass < maxPasses; pass++) {
        bool changed = false;
        for (auto& [nm, expr] : sr.expressionConstraints) {
            if (fullAssign.count(nm)) continue;
            Poly p = expr;
            for (auto& [k, v] : fullAssign) {
                auto it = G_vars.nameToIdx.find(k);
                if (it != G_vars.nameToIdx.end())
                    p = polySub1(p, it->second, (int)v);
            }
            if (freeVars(p).empty()) {
                fullAssign[nm] = (int64_t)getConst(p);
                changed = true;
            }
        }
        if (!changed) break;
    }

    // 4. Free variable fallback: assign 1 to any unresolved p_i / q_i.
    //    This handles variables that were substituted out during preprocessing
    //    and only appear on the RHS of expression constraints.
    for (int i = 0; i < n_p; i++) {
        auto key = "p_" + std::to_string(i);
        if (!fullAssign.count(key)) fullAssign[key] = 1;
    }
    for (int i = 0; i < n_q; i++) {
        auto key = "q_" + std::to_string(i);
        if (!fullAssign.count(key)) fullAssign[key] = 1;
    }

    // 5. Expression constraint resolution (pass 2)
    //    Re-resolve now that fallback values have been assigned.
    for (int pass = 0; pass < maxPasses; pass++) {
        bool changed = false;
        for (auto& [nm, expr] : sr.expressionConstraints) {
            Poly p = expr;
            for (auto& [k, v] : fullAssign) {
                auto it = G_vars.nameToIdx.find(k);
                if (it != G_vars.nameToIdx.end())
                    p = polySub1(p, it->second, (int)v);
            }
            if (freeVars(p).empty()) {
                int64_t resolved = (int64_t)getConst(p);
                if (!fullAssign.count(nm) || fullAssign[nm] != resolved) {
                    fullAssign[nm] = resolved;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // 6. Fixed bits guarantee
    fullAssign["p_0"] = 1;
    fullAssign["p_" + std::to_string(n_p - 1)] = 1;
    fullAssign["q_0"] = 1;
    fullAssign["q_" + std::to_string(n_q - 1)] = 1;

    // 7. Reconstruct P and Q
    uint128_t P = 0, Q = 0;
    for (int i = 0; i < n_p; i++) {
        auto key = "p_" + std::to_string(i);
        if (fullAssign.count(key))
            P += (uint128_t)fullAssign[key] * ((uint128_t)1 << i);
    }
    for (int i = 0; i < n_q; i++) {
        auto key = "q_" + std::to_string(i);
        if (fullAssign.count(key))
            Q += (uint128_t)fullAssign[key] * ((uint128_t)1 << i);
    }

    // 8. Verify
    std::cout << "\nFactors of " << N << ": P = " << P << ", Q = " << Q << "\n";
    if (P * Q == N)
        std::cout << "Verification: " << P << " * " << Q << " = " << P * Q << " (CORRECT)\n";
    else
        std::cout << "Verification: " << P << " * " << Q << " = " << P * Q << " (INCORRECT, expected " << N << ")\n";
}

// ============================================================
// Read Ising spins file (tab-separated +1/-1 on first line)
// and convert to QUBO binary (0/1).
// ============================================================
std::vector<int> readSpinsFile(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open spins file: " + filename);
    std::string line;
    std::getline(f, line);
    std::istringstream iss(line);
    std::vector<int> quboSol;
    int spin;
    while (iss >> spin)
        quboSol.push_back((spin + 1) / 2);  // +1 -> 1, -1 -> 0
    return quboSol;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N> [spins_file]\n";
        std::cerr << "  Without spins_file: generate QUBO and save CSR files.\n";
        std::cerr << "  With spins_file:    generate QUBO, then post-process the annealer solution.\n";
        return 1;
    }

    uint128_t N = parseUint128(argv[1]);
    std::string spinsFile = (argc >= 3) ? argv[2] : "";

    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: initialize variables
    ProblemVars pv = initializeVariables(N);

    // Step 2: generate column clauses
    std::vector<Poly> clauses = generateColumnClauses(N, pv);
    std::cout << "Generated " << clauses.size() << " column clauses\n";

    // Step 3: simplify
    // Squaring cost budget = n_q^3.  This keeps total QUBO vars at O(n^3).
    // For close primes (clauses shrink to near-zero), the budget is never
    // hit and everything resolves fully.  For far-apart primes, the budget
    // prevents mega-clause formation from cascading replacement.
    SimplifierResult sr = clauseSimplifier(std::move(clauses));
    std::cout << "After simplification: " << sr.assignmentConstraints.size()
              << " assignment constraints, "
              << sr.expressionConstraints.size() << " expression constraints\n";

    // Count non-zero remaining clauses
    int nonZero = 0;
    int maxTerms = 0;
    int totalTerms = 0;
    for (auto& c : sr.clauses) {
        if (!isZero(c)) {
            nonZero++;
            int t = (int)c.size();
            totalTerms += t;
            if (t > maxTerms) maxTerms = t;
        }
    }
    std::cout << "Non-zero remaining clauses: " << nonZero
              << " (max terms=" << maxTerms << ", total terms=" << totalTerms << ")\n";

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
    std::string Nstr = uint128ToString(N);
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

    // Post-processing
    if (numVars == 0) {
        std::cout << "\n=== Verification (fully solved by preprocessing) ===\n";
        postProcess({}, N, pv.n_p, pv.n_q, sr, {});
    } else if (!spinsFile.empty()) {
        std::cout << "\n=== Post-processing with spins from: " << spinsFile << " ===\n";
        std::vector<int> quboSol = readSpinsFile(spinsFile);
        if ((int)quboSol.size() != numVars) {
            std::cerr << "ERROR: spins file has " << quboSol.size()
                      << " values but QUBO has " << numVars << " variables.\n";
            return 1;
        }
        postProcess(quboSol, N, pv.n_p, pv.n_q, sr, activeVarNames);
    } else {
        std::cout << "\n[QUBO has " << numVars << " variables. "
                  << "Re-run with spins file: " << argv[0] << " " << N << " spins_" << Nstr << "]\n";
    }

    return 0;
}
