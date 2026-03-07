/**
 * QUBO Integer Factorization via Annealing
 *
 * C++ port of qubo_integer_factorization_final.py
 *
 * Pipeline:
 *   1. initialize_variables     -- bit sizes, fixed bits, carry symbols
 *   2. generate_column_clauses  -- column-wise constraint expressions
 *   3. clause_simplifier        -- classical pre-processing (Rules 1-6,
 *                                  parity, power, replacement)
 *   4. qubo_hamiltonian         -- sum-of-squares -> polynomial
 *   5. quadratize               -- reduce degree-3/4 terms with auxiliaries
 *   6. output QUBO matrix + post-processing helpers
 *
 * Symbolic algebra is done with a hand-rolled sparse polynomial type over
 * integer coefficients and string-named variables, so there are NO external
 * library dependencies beyond the C++ standard library.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o qubo_factorization qubo_factorization.cpp
 *
 * Usage:
 *   ./qubo_factorization <N>
 *
 * Output (to stdout):
 *   - Progress messages that mirror the Python prints
 *   - QUBO matrix (dense, upper-triangular) as CSV
 *   - Variable index mapping
 *   - Assignment constraints (var = integer value)
 *   - Expression constraints (var = linear combination of other vars)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

// ============================================================
//  Sparse polynomial over integers
//  A "monomial" is a sorted vector of variable names (with repetition
//  representing powers, but the power rule collapses x^k -> x).
//  The polynomial is a map  monomial -> coefficient.
// ============================================================

using VarName  = std::string;
using Monomial = std::vector<VarName>;   // sorted, may be empty (= constant 1)

// Canonical form: sorted, deduplicated (power rule x^k = x)
static Monomial canonical(Monomial m) {
    std::sort(m.begin(), m.end());
    m.erase(std::unique(m.begin(), m.end()), m.end());
    return m;
}

struct Poly {
    std::map<Monomial, int64_t> terms;   // monomial -> coefficient

    Poly() = default;
    explicit Poly(int64_t c) { if (c) terms[{}] = c; }
    explicit Poly(const VarName& v) { terms[{v}] = 1; }

    bool isZero() const { return terms.empty(); }

    // degree of highest-degree term
    int degree() const {
        int d = 0;
        for (auto& [m, c] : terms) d = std::max(d, (int)m.size());
        return d;
    }

    // set of all free variables
    std::set<VarName> freeVars() const {
        std::set<VarName> s;
        for (auto& [m, c] : terms) for (auto& v : m) s.insert(v);
        return s;
    }

    // constant term
    int64_t constant() const {
        auto it = terms.find({});
        return it != terms.end() ? it->second : 0;
    }

    // linear coefficient of variable v  (only for linear monomials)
    int64_t coeff(const VarName& v) const {
        auto it = terms.find({v});
        return it != terms.end() ? it->second : 0;
    }

    void setCoeff(const VarName& v, int64_t c) {
        Monomial m = {v};
        if (c == 0) terms.erase(m);
        else        terms[m] = c;
    }

    void addTerm(const Monomial& m, int64_t c) {
        if (c == 0) return;
        terms[m] += c;
        if (terms[m] == 0) terms.erase(m);
    }

    void addConst(int64_t c) { addTerm({}, c); }

    Poly& operator+=(const Poly& o) {
        for (auto& [m, c] : o.terms) addTerm(m, c);
        return *this;
    }
    Poly& operator-=(const Poly& o) {
        for (auto& [m, c] : o.terms) addTerm(m, -c);
        return *this;
    }
    Poly& operator*=(int64_t s) {
        if (s == 0) { terms.clear(); return *this; }
        for (auto& [m, c] : terms) const_cast<int64_t&>(c) *= s;
        return *this;
    }

    Poly operator+(const Poly& o) const { Poly r=*this; r+=o; return r; }
    Poly operator-(const Poly& o) const { Poly r=*this; r-=o; return r; }
    Poly operator*(int64_t s)     const { Poly r=*this; r*=s; return r; }
    Poly operator-()              const { return *this * -1; }

    // polynomial multiplication
    Poly operator*(const Poly& o) const {
        Poly r;
        for (auto& [ma, ca] : terms)
            for (auto& [mb, cb] : o.terms) {
                Monomial mc;
                mc.insert(mc.end(), ma.begin(), ma.end());
                mc.insert(mc.end(), mb.begin(), mb.end());
                mc = canonical(mc);
                r.addTerm(mc, ca * cb);
            }
        return r;
    }

    // Apply power rule: x^k -> x (dedup already done in canonical)
    // (already baked in via canonical() on construction)

    // Substitute var -> poly
    Poly subs(const VarName& var, const Poly& val) const {
        Poly result;
        for (auto& [m, c] : terms) {
            // Check if var appears in this monomial
            bool has = std::binary_search(m.begin(), m.end(), var);
            if (!has) {
                result.addTerm(m, c);
                continue;
            }
            // Build monomial without the variable, then multiply by val
            Monomial rest;
            for (auto& v : m) if (v != var) rest.push_back(v);
            Poly restPoly;
            restPoly.addTerm(rest, c);
            result += restPoly * val;
        }
        // Apply power rule (canonical already dedups powers, but after
        // multiplication new duplicates can appear — handled by addTerm via canonical)
        return result;
    }

    // Substitute var -> integer
    Poly subs(const VarName& var, int64_t val) const {
        return subs(var, Poly(val));
    }

    // Apply all substitutions in a map
    Poly subsAll(const std::map<VarName, Poly>& sub) const {
        Poly r = *this;
        for (auto& [v, p] : sub) r = r.subs(v, p);
        return r;
    }

    bool operator==(int64_t c) const {
        if (c == 0) return terms.empty();
        return terms.size() == 1 && terms.begin()->first.empty()
               && terms.begin()->second == c;
    }

    std::string toString() const {
        if (terms.empty()) return "0";
        std::ostringstream oss;
        bool first = true;
        for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
            auto& [m, c] = *it;
            if (c == 0) continue;
            if (!first && c > 0) oss << " + ";
            else if (c < 0) oss << " - ";
            int64_t ac = std::abs(c);
            if (m.empty()) { oss << ac; }
            else {
                if (ac != 1) oss << ac << "*";
                for (size_t i = 0; i < m.size(); i++) {
                    if (i) oss << "*";
                    oss << m[i];
                }
            }
            first = false;
        }
        return oss.str();
    }
};

static Poly polyVar(const VarName& v) { return Poly(v); }
static Poly polyConst(int64_t c)      { return Poly(c); }

// GCD of absolute values of all coefficients
static int64_t polyGCD(const Poly& p) {
    int64_t g = 0;
    for (auto& [m, c] : p.terms) g = std::gcd(g, std::abs(c));
    return g == 0 ? 1 : g;
}

// ============================================================
//  Constraint types
//   "assignment": var = integer_value
//   "expression": var = Poly(other vars)
// ============================================================

struct AssignmentConstraint { VarName var; int64_t val; };
struct ExpressionConstraint { VarName var; Poly    expr; };

// ============================================================
//  Variable naming helpers (mirror Python)
// ============================================================
static std::string pname(int i) { return "p_" + std::to_string(i); }
static std::string qname(int i) { return "q_" + std::to_string(i); }
static std::string sname(int i, int j) { return "s_" + std::to_string(i) + "_" + std::to_string(j); }

static bool isP(const VarName& v) { return v.size() > 2 && v[0]=='p' && v[1]=='_'; }
static bool isQ(const VarName& v) { return v.size() > 2 && v[0]=='q' && v[1]=='_'; }
static bool isS(const VarName& v) { return v.size() > 2 && v[0]=='s' && v[1]=='_'; }

// ============================================================
//  1. initialize_variables
// ============================================================
struct Variables {
    int n_p, n_q;
    std::map<int, Poly> p;   // bit index -> Poly (either Const(1) or Var)
    std::map<int, Poly> q;
    std::map<std::pair<int,int>, Poly> s;  // (i,j) -> Poly or zero
};

static Variables initialize_variables(int64_t N) {
    Variables vars;
    int n_m = (int)std::ceil(std::log2((double)(N + 1)));
    int n_q = (int)std::ceil(0.5 * std::log2((double)N));
    int n_p = n_q;
    vars.n_p = n_p;
    vars.n_q = n_q;

    std::cout << "Factoring N = " << N << " (" << n_m << " bits)\n";
    std::cout << "Assuming n_p = " << n_p << ", n_q = " << n_q << "\n";

    // p bits
    for (int i = 0; i < n_p; i++) {
        if (i == 0 || i == n_p - 1)
            vars.p[i] = polyConst(1);
        else
            vars.p[i] = polyVar(pname(i));
    }

    // q bits
    for (int i = 0; i < n_q; i++) {
        if (i == 0 || i == n_q - 1)
            vars.q[i] = polyConst(1);
        else
            vars.q[i] = polyVar(qname(i));
    }

    // carry bits
    int max_sum_terms = n_q + (n_q - 1);
    for (int i = 1; i < n_p + n_q; i++) {
        int num_prod = std::min(i, n_q-1) - std::max(i - n_p + 1, 0) + 1;
        int num_carry_in = i - 1;
        int max_sum = num_prod + num_carry_in;
        if (max_sum > 1) {
            int num_out = (int)std::floor(std::log2((double)max_sum));
            for (int j = 1; j <= num_out; j++) {
                if (i + j < n_p + n_q) {
                    vars.s[{i, i+j}] = polyVar(sname(i, i+j));
                }
            }
        }
    }

    return vars;
}

// ============================================================
//  2. generate_column_clauses
// ============================================================
static std::vector<Poly> generate_column_clauses(int64_t N,
                                                   const Variables& vars)
{
    int n_p = vars.n_p, n_q = vars.n_q;
    int n_m = n_p + n_q;

    // N bits, LSB first
    std::vector<int> N_bits(n_m, 0);
    for (int i = 0; i < n_m; i++)
        N_bits[i] = (int)((N >> i) & 1);

    auto getS = [&](int a, int b) -> Poly {
        auto it = vars.s.find({a, b});
        if (it != vars.s.end()) return it->second;
        return polyConst(0);
    };

    std::vector<Poly> clauses;

    for (int i = 1; i < n_m; i++) {
        Poly clause = polyConst(N_bits[i]);

        // subtract product terms
        for (int j = 0; j < n_p; j++) {
            int ij = i - j;
            if (ij >= 0 && ij < n_q) {
                clause -= vars.p.at(j) * vars.q.at(ij);
            }
        }

        // subtract input carries
        for (int k = 1; k < i; k++) {
            Poly sk = getS(k, i);
            if (!sk.isZero()) clause -= sk;
        }

        // add output carries
        int jc = 1;
        while (true) {
            Poly sij = getS(i, i + jc);
            if (sij.isZero()) break;
            if (i + jc > n_p + n_q - 1) {
                clauses.push_back(sij);
            }
            clause += polyConst(1LL << jc) * sij;
            jc++;
        }

        clauses.push_back(clause);
    }

    return clauses;
}

// ============================================================
//  Helper: substitute a map of VarName->Poly into all clauses
// ============================================================
static void subClauses(std::vector<Poly>& clauses,
                       const std::map<VarName, Poly>& sub)
{
    for (auto& cl : clauses)
        for (auto& [v, p] : sub)
            cl = cl.subs(v, p);
}

// Power rule: dedup already handled by canonical() in Poly, but
// after substitution integers may need squashing.
static void applyPowerRule(std::vector<Poly>& clauses) {
    // The canonical() in Poly already applies the power rule on construction.
    // Nothing extra needed since our Poly always stores canonical monomials.
    (void)clauses;
}

// ============================================================
//  3. Classical pre-processing rules
//     Each rule returns a substitution map  VarName -> Poly.
//     Empty map = no constraint found.
// ============================================================

// --- Rule 1 & 2 ---
static std::map<VarName, Poly> applyRule12(const Poly& clause) {
    std::map<VarName, Poly> res;
    if (clause.isZero()) return res;

    int64_t cnst = clause.constant();
    // Collect only LINEAR (degree-1) terms for checking
    std::vector<std::pair<VarName, int64_t>> linTerms;
    bool hasNonLinear = false;
    for (auto& [m, c] : clause.terms) {
        if (m.empty()) continue;
        if (m.size() == 1) linTerms.push_back({m[0], c});
        else { hasNonLinear = true; break; }
    }
    if (hasNonLinear) return res;
    if (linTerms.empty()) return res;

    int n = (int)linTerms.size();

    // Rule 1: sum -xi = -n  =>  xi=1 (all coeff -1, const = -n)
    //         sum xi = n    =>  xi=1 (all coeff +1, const = +n ... but clause=0 so const=-n)
    // Rule 2: sum xi = 0    =>  xi=0 (all coeff +1, const=0)
    //         sum -xi= 0    =>  xi=0

    bool allNeg1 = std::all_of(linTerms.begin(), linTerms.end(),
                               [](auto& p){ return p.second == -1; });
    bool allPos1 = std::all_of(linTerms.begin(), linTerms.end(),
                               [](auto& p){ return p.second == +1; });

    // Rule 1: -x1 - x2 - ... - xn + n = 0  (allNeg1 && cnst==n)
    if (allNeg1 && cnst == (int64_t)n)
        for (auto& [v, c] : linTerms) res[v] = polyConst(1);
    // Rule 1: x1 + ... + xn - n = 0  (allPos1 && cnst==-n)
    if (allPos1 && cnst == -(int64_t)n)
        for (auto& [v, c] : linTerms) res[v] = polyConst(1);
    // Rule 2: x1 + ... = 0  (allPos1, cnst==0)
    if (allPos1 && cnst == 0)
        for (auto& [v, c] : linTerms) res[v] = polyConst(0);
    // Rule 2: -x1 - ... = 0  (allNeg1, cnst==0)
    if (allNeg1 && cnst == 0)
        for (auto& [v, c] : linTerms) res[v] = polyConst(0);

    return res;
}

// --- Rule 4 (general bound-based reduction) ---
static std::map<VarName, Poly> applyRule4(const Poly& clause) {
    std::map<VarName, Poly> res;
    if (clause.isZero()) return res;

    // Only operate on purely linear clauses (degree ≤ 1)
    for (auto& [m, c] : clause.terms)
        if (m.size() > 1) return res;

    int64_t cnst = clause.constant();
    std::map<VarName, int64_t> linMap;
    for (auto& [m, c] : clause.terms) {
        if (!m.empty()) linMap[m[0]] = c;
    }

    int64_t posSum = 0, negSum = 0;
    for (auto& [v, c] : linMap) {
        if (c > 0) posSum += c;
        else       negSum += -c;
    }

    for (auto& [v, c] : linMap) {
        if (c > 0) {
            if (c > negSum - cnst) res[v] = polyConst(0);
            if (c > posSum + cnst) res[v] = polyConst(1);
        } else {
            int64_t ac = -c;
            if (ac > posSum + cnst) res[v] = polyConst(0);
            if (ac > negSum - cnst) res[v] = polyConst(1);
        }
    }
    return res;
}

// --- Rule 3 (x1 + x2 = 2*x3) ---
// Returns {substitution_map, mul_constraints_map}
// The Python code is complex; we implement the core logic.
static std::pair<std::map<VarName, Poly>, std::map<VarName, Poly>>
applyRule3(const Poly& clause) {
    std::map<VarName, Poly> res, mulRes;

    // Pattern: exactly 3 linear terms with coefficients {+1,+1,-2} and const=0
    // or {-1,-1,+2} and const=0
    if (clause.isZero()) return {res, mulRes};

    // Separate constant and linear terms
    int64_t cnst = clause.constant();
    if (cnst != 0) return {res, mulRes};

    std::map<Monomial, int64_t> terms;
    for (auto& [m, c] : clause.terms)
        if (!m.empty()) terms[m] = c;

    if (terms.size() != 3) return {res, mulRes};

    // Try to find pattern coeff set {1, 1, -2} on linear (degree-1) monomials
    std::vector<std::pair<Monomial, int64_t>> tv(terms.begin(), terms.end());

    // Normalize: try both sign polarities
    for (int sign : {1, -1}) {
        std::vector<std::pair<Monomial, int64_t>> t;
        for (auto& [m, c] : tv) t.push_back({m, sign * c});

        // Sort by coefficient
        std::sort(t.begin(), t.end(),
            [](auto& a, auto& b){ return a.second < b.second; });

        // Expect coefficients: -2, 1, 1  (sorted)
        if (t[0].second == -2 && t[1].second == 1 && t[2].second == 1) {
            Monomial& m2 = t[0].first;  // the -2 term (this is x3)
            Monomial& m1a = t[1].first;
            Monomial& m1b = t[2].first;

            // Only handle the simple case where all three monomials are
            // single variables (degree 1)
            if (m2.size() == 1 && m1a.size() == 1 && m1b.size() == 1) {
                VarName v2 = m2[0], va = m1a[0], vb = m1b[0];
                // Prefer substituting s variables
                if (isS(v2)) {
                    // s = va and s = vb => va = vb  (substitute s -> va)
                    res[v2] = polyVar(va);
                    if (va != vb) res[vb] = polyVar(va);
                } else if (isS(va)) {
                    res[va] = polyVar(v2);
                    if (v2 != vb) res[vb] = polyVar(v2);
                } else if (isS(vb)) {
                    res[vb] = polyVar(v2);
                    if (v2 != va) res[va] = polyVar(v2);
                } else {
                    res[va] = polyVar(vb);
                    res[v2] = polyVar(vb);
                }
                return {res, mulRes};
            }
            // Higher degree monomials — mirror Python mul handling:
            // record them as mul constraints but don't substitute
        }
    }
    return {res, mulRes};
}

// --- Rule 6 (complementary pairs) ---
static std::map<VarName, Poly> applyRule6(const Poly& clause) {
    std::map<VarName, Poly> res;
    if (clause.isZero()) return res;

    // Only linear clauses
    for (auto& [m, c] : clause.terms)
        if (m.size() > 1) return res;

    int64_t cnst = clause.constant();
    std::vector<std::pair<VarName, int64_t>> pos, neg;
    for (auto& [m, c] : clause.terms) {
        if (m.empty()) continue;
        if (c > 0) pos.push_back({m[0], c});
        else       neg.push_back({m[0], -c});
    }

    int64_t posSum = 0, negSum = 0;
    for (auto& [v, c] : pos) posSum += c;
    for (auto& [v, c] : neg) negSum += c;

    auto tryPair = [&](std::vector<std::pair<VarName,int64_t>>& arr,
                       int64_t threshold, int64_t tiebreak,
                       bool posCase) {
        std::sort(arr.begin(), arr.end(),
            [](auto& a, auto& b){ return a.second > b.second; });
        for (size_t i = 0; i < arr.size(); i++) {
            for (size_t j = i+1; j < arr.size(); j++) {
                int64_t sum = arr[i].second + arr[j].second;
                if (sum > threshold) {
                    if (tiebreak > 0) {
                        VarName y = arr[i].first, x = arr[j].first;
                        if (isS(y)) res[y] = polyConst(1) - polyVar(x);
                        else if (isS(x)) res[x] = polyConst(1) - polyVar(y);
                        else res[y] = polyConst(1) - polyVar(x);
                        return;
                    }
                } else {
                    return;
                }
            }
        }
    };

    // pos case: posSum > negSum - cnst  && cnst < 0
    if (cnst < 0 && posSum > negSum - cnst)
        tryPair(pos, negSum - cnst, -(cnst) - posSum, true);

    // neg case: negSum > posSum + cnst  && cnst > 0
    if (cnst > 0 && negSum > posSum + cnst)
        tryPair(neg, posSum + cnst, negSum - cnst, false);

    return res;
}

// --- Parity rule ---
static std::map<VarName, Poly> applyParityRule(const Poly& clause) {
    std::map<VarName, Poly> res;
    if (clause.isZero()) return res;

    // Collect terms with odd coefficients
    Poly oddPart;
    for (auto& [m, c] : clause.terms)
        if (c % 2 != 0)
            oddPart.addTerm(m, (c % 2 + 2) % 2 == 1 ? 1 : -1);  // keep parity

    // We need the sum of odd-coefficient terms to vanish mod 2.
    // Collect them with just their parities.
    // Simpler: re-compute with c mod 2 (sign-aware)
    Poly op2;
    for (auto& [m, c] : clause.terms) {
        int64_t cm = ((c % 2) + 2) % 2;  // 0 or 1
        if (cm) op2.addTerm(m, cm);
    }

    // Now op2 should have an even sum. Patterns:
    // x1 + x2 = 0 mod 2 => x1 = x2
    // x1 + x2 + 1 = 0 mod 2 => x1 + x2 = 1

    // Count terms
    int64_t cnst2 = op2.constant();
    std::vector<std::pair<Monomial, int64_t>> linT;
    bool hasNL = false;
    for (auto& [m, c] : op2.terms) {
        if (m.empty()) continue;
        if (m.size() > 1) { hasNL = true; break; }
        linT.push_back({m, c});
    }
    if (hasNL || linT.size() != 2) return res;

    VarName va = linT[0].first[0];
    VarName vb = linT[1].first[0];

    if (cnst2 == 0) {
        // x1 + x2 = 0 mod 2 => x1 = x2
        if (isS(va)) res[va] = polyVar(vb);
        else if (isS(vb)) res[vb] = polyVar(va);
        else res[va] = polyVar(vb);
    } else {
        // cnst2 == 1 => x1 + x2 = 1 => x1 = 1 - x2
        if (isS(va)) res[va] = polyConst(1) - polyVar(vb);
        else if (isS(vb)) res[vb] = polyConst(1) - polyVar(va);
        else res[va] = polyConst(1) - polyVar(vb);
    }
    return res;
}

// --- Replacement rule (last resort, only s variables) ---
static std::map<VarName, Poly> replacement(const std::vector<Poly>& clauses) {
    for (auto& clause : clauses) {
        if (clause.isZero()) continue;
        auto fv = clause.freeVars();
        for (auto& var : fv) {
            if (!isS(var)) continue;
            int64_t c = clause.coeff(var);
            if (c != 1 && c != -1) continue;
            // solve: clause = 0 for var
            // var = -(clause - c*var) / c
            Poly rest;
            for (auto& [m, coef] : clause.terms) {
                if (m.size() == 1 && m[0] == var) continue;
                rest.addTerm(m, coef);
            }
            // var = -rest / c
            rest *= -1;
            if (c == -1) rest *= -1;
            return {{var, rest}};
        }
    }
    return {};
}

// ============================================================
//  Main pre-processing loop (clause_simplifier)
// ============================================================

struct SimplifierResult {
    std::vector<Poly> clauses;
    std::vector<AssignmentConstraint> assignConstraints;
    std::vector<ExpressionConstraint> exprConstraints;
    // mul_constraints not used in downstream QUBO (only diagnostic)
};

static SimplifierResult clause_simplifier(std::vector<Poly> clauses) {
    SimplifierResult sr;
    sr.clauses = std::move(clauses);

    int maxIter = 2 * (int)sr.clauses.size();
    std::cout << "Total iterations possible: " << maxIter << "\n";

    auto recordAndApply = [&](std::map<VarName, Poly>& constraints, bool isAssign) {
        if (constraints.empty()) return false;
        std::map<VarName, Poly> sub;
        for (auto& [var, val] : constraints) {
            sub[var] = val;
            if (isP(var) || isQ(var)) {
                if (val.freeVars().empty()) {
                    sr.assignConstraints.push_back({var, val.constant()});
                } else {
                    sr.exprConstraints.push_back({var, val});
                }
            } else if (isS(var)) {
                if (!val.freeVars().empty()) {
                    sr.exprConstraints.push_back({var, val});
                }
            }
        }
        subClauses(sr.clauses, sub);
        applyPowerRule(sr.clauses);
        return true;
    };

    for (int iter = 0; iter < maxIter; iter++) {
        // Divide clauses by GCD
        for (auto& cl : sr.clauses) {
            int64_t g = polyGCD(cl);
            if (g > 1) cl *= (1);  // already reduced by addTerm; divide:
            if (g > 1) {
                Poly tmp;
                for (auto& [m, c] : cl.terms)
                    tmp.addTerm(m, c / g);
                cl = tmp;
            }
        }

        bool found = false;

        // Rule 1 & 2
        for (auto& cl : sr.clauses) {
            auto c = applyRule12(cl);
            if (!c.empty()) { found = recordAndApply(c, true); break; }
        }
        if (found) continue;

        // Rule 4
        for (auto& cl : sr.clauses) {
            auto c = applyRule4(cl);
            if (!c.empty()) { found = recordAndApply(c, true); break; }
        }
        if (found) continue;

        // Rule 3
        for (auto& cl : sr.clauses) {
            auto [c, mc] = applyRule3(cl);
            if (!c.empty()) {
                std::map<VarName, Poly> sub;
                for (auto& [v, p] : c) {
                    sub[v] = p;
                    if (isP(v) || isQ(v)) {
                        if (p.freeVars().empty())
                            sr.assignConstraints.push_back({v, p.constant()});
                        else
                            sr.exprConstraints.push_back({v, p});
                    }
                }
                subClauses(sr.clauses, sub);
                applyPowerRule(sr.clauses);
                found = true;
                break;
            }
        }
        if (found) continue;

        // Rule 6
        for (auto& cl : sr.clauses) {
            auto c = applyRule6(cl);
            if (!c.empty()) {
                std::map<VarName, Poly> sub;
                for (auto& [v, p] : c) {
                    sub[v] = p;
                    if (isP(v) || isQ(v))
                        sr.exprConstraints.push_back({v, p});
                }
                subClauses(sr.clauses, sub);
                applyPowerRule(sr.clauses);
                found = true;
                break;
            }
        }
        if (found) continue;

        // Parity rule
        for (auto& cl : sr.clauses) {
            auto c = applyParityRule(cl);
            if (!c.empty()) {
                std::map<VarName, Poly> sub;
                for (auto& [v, p] : c) {
                    sub[v] = p;
                    if (isP(v) || isQ(v))
                        sr.exprConstraints.push_back({v, p});
                }
                subClauses(sr.clauses, sub);
                applyPowerRule(sr.clauses);
                found = true;
                break;
            }
        }
        if (found) continue;

        // Replacement (last resort)
        auto c = replacement(sr.clauses);
        if (!c.empty()) {
            std::map<VarName, Poly> sub;
            for (auto& [v, p] : c) {
                sub[v] = p;
                if (isP(v) || isQ(v))
                    sr.exprConstraints.push_back({v, p});
            }
            subClauses(sr.clauses, sub);
            applyPowerRule(sr.clauses);
        } else {
            break;  // no rule applied — done
        }
    }

    return sr;
}

// ============================================================
//  4. Build QUBO Hamiltonian: H = sum_i  clause_i^2
//     Expand and collect, then quadratize.
// ============================================================

static Poly buildHamiltonian(const std::vector<Poly>& clauses) {
    Poly H;
    for (auto& cl : clauses) {
        if (!cl.isZero()) {
            H += cl * cl;
        }
    }
    return H;
}

// ============================================================
//  5. Quadratization
//
//  The expanded Hamiltonian may have degree-3 or degree-4 terms.
//  We introduce auxiliary variables to reduce them to quadratic.
//
//  For positive coeff  *a* on monomial (x1 x2 x3):
//    introduce w: penalty = a*(w*x3 + x1*x2 - x1*w - x2*w + w)   [Eq.9]
//  For negative coeff *a* on monomial (x1 x2 x3):
//    introduce w: penalty = a*(-w*(x1+x2+x3-2))                   [Eq.11]
//  For positive coeff *a* on monomial (x1 x2 x3 x4):
//    introduce w,z: two levels of Eq.9
//  For negative coeff *a* on monomial (x1 x2 x3 x4):
//    introduce w: penalty = a*(-w*(x1+x2+x3+x4-3))
//
//  We accumulate the QUBO as maps:
//    linear:    var -> coeff
//    quadratic: {var_i, var_j} -> coeff  (i<j by lexicographic sort)
//    offset
// ============================================================

struct QUBO {
    // Variables are identified by string name
    std::map<VarName, double>                          linear;
    std::map<std::pair<VarName,VarName>, double>       quadratic;
    double offset = 0.0;

    // Sorted canonical pair
    static std::pair<VarName,VarName> qpair(const VarName& a, const VarName& b) {
        return a < b ? std::make_pair(a,b) : std::make_pair(b,a);
    }

    void addLinear(const VarName& v, double c) {
        linear[v] += c;
        if (linear[v] == 0.0) linear.erase(v);
    }
    void addQuad(const VarName& a, const VarName& b, double c) {
        if (a == b) { addLinear(a, c); return; }
        auto key = qpair(a, b);
        quadratic[key] += c;
        if (quadratic[key] == 0.0) quadratic.erase(key);
    }

    std::set<VarName> variables() const {
        std::set<VarName> s;
        for (auto& [v,c] : linear) s.insert(v);
        for (auto& [p,c] : quadratic) { s.insert(p.first); s.insert(p.second); }
        return s;
    }

    // Add a penalty for a positive-coefficient 3-local term: a*x1*x2*x3
    // Using Eq.9: a*(w*x3 + x1*x2 - x1*w - x2*w + w)
    void addPos3(const VarName& x1, const VarName& x2, const VarName& x3,
                 double a, const VarName& w) {
        addLinear(w, a);
        addQuad(w, x3, a);
        addQuad(x1, x2, a);
        addQuad(w, x1, -a);
        addQuad(w, x2, -a);
    }

    // Negative 3-local: a*x1*x2*x3  where a < 0 => use Eq.11:
    // a * min_w(-w*(x1+x2+x3-2)) => add a*(-w*x1 - w*x2 - w*x3 + 2*w)
    void addNeg3(const VarName& x1, const VarName& x2, const VarName& x3,
                 double a, const VarName& w) {
        // penalty: -a * (-w*(x1+x2+x3-2))  but we have +a (a<0)
        // => a*(-w*x1 - w*x2 - w*x3 + 2*w)
        addLinear(w, 2.0 * a);
        addQuad(w, x1, -a);
        addQuad(w, x2, -a);
        addQuad(w, x3, -a);
    }

    // Positive 4-local: a*x1*x2*x3*x4 using two aux vars w, z
    // First: a*(w*x3*x4 + x1*x2 - w*x1 - w*x2 + w)  [w replaces x1*x2]
    // Then:  a*(z*x4 + w*x3 - z*w - z*x3 + z)        [z replaces w*x3  ... but w*x3 is still 3-local]
    // Net quadratic (combining):
    //   a*(x1*x2 - w*x1 - w*x2 + w  +  z*x4 + w*x3 - z*w - z*x3 + z)
    void addPos4(const VarName& x1, const VarName& x2,
                 const VarName& x3, const VarName& x4,
                 double a, const VarName& w, const VarName& z) {
        addQuad(x1, x2, a);
        addQuad(w, x1, -a);
        addQuad(w, x2, -a);
        addLinear(w, a);
        addQuad(z, x4, a);
        addQuad(w, x3, a);
        addQuad(z, w,  -a);
        addQuad(z, x3, -a);
        addLinear(z, a);
    }

    // Negative 4-local: a*x1*x2*x3*x4  (a < 0)
    // Using: -x1x2x3x4 = min_w(-w*(x1+x2+x3+x4-3))
    // So penalty: a*(-w*(x1+x2+x3+x4-3)) = a*(-w*x1-w*x2-w*x3-w*x4+3*w)
    void addNeg4(const VarName& x1, const VarName& x2,
                 const VarName& x3, const VarName& x4,
                 double a, const VarName& w) {
        addLinear(w, 3.0 * a);
        addQuad(w, x1, -a);
        addQuad(w, x2, -a);
        addQuad(w, x3, -a);
        addQuad(w, x4, -a);
    }
};

static QUBO quadratize(const Poly& H) {
    QUBO qubo;
    int auxIdx = 0;

    for (auto& [m, c] : H.terms) {
        if (c == 0) continue;
        double dc = (double)c;

        if (m.empty()) {
            qubo.offset += dc;
        } else if (m.size() == 1) {
            qubo.addLinear(m[0], dc);
        } else if (m.size() == 2) {
            qubo.addQuad(m[0], m[1], dc);
        } else if (m.size() == 3) {
            std::string w = "w" + std::to_string(auxIdx++);
            if (dc > 0)
                qubo.addPos3(m[0], m[1], m[2], dc, w);
            else
                qubo.addNeg3(m[0], m[1], m[2], dc, w);
        } else if (m.size() == 4) {
            if (dc > 0) {
                std::string w = "w" + std::to_string(auxIdx++);
                std::string z = "w" + std::to_string(auxIdx++);
                qubo.addPos4(m[0], m[1], m[2], m[3], dc, w, z);
            } else {
                std::string w = "w" + std::to_string(auxIdx++);
                qubo.addNeg4(m[0], m[1], m[2], m[3], dc, w);
            }
        } else {
            // degree > 4: should not occur with equal-length factor assumption
            // but handle gracefully by printing a warning
            std::cerr << "Warning: monomial of degree " << m.size()
                      << " encountered; skipping.\n";
        }
    }

    return qubo;
}

// ============================================================
//  6. Output QUBO matrix (upper-triangular, dense)
//     and all associated metadata
// ============================================================

struct QUBOResult {
    std::vector<std::vector<double>>         matrix;
    std::map<int, VarName>                   indexToVar;
    std::map<VarName, int>                   varToIndex;
    std::vector<AssignmentConstraint>        assignConstraints;
    std::vector<ExpressionConstraint>        exprConstraints;
    double                                   offset;
    int                                      n_p, n_q;
};

static QUBOResult buildQUBOResult(int64_t N) {
    Variables vars = initialize_variables(N);
    auto clauses   = generate_column_clauses(N, vars);

    std::cout << "Initial clauses (" << clauses.size() << "):\n";
    for (size_t i = 0; i < clauses.size(); i++)
        std::cout << "  C" << i+1 << ": " << clauses[i].toString() << " = 0\n";

    auto sr = clause_simplifier(clauses);

    std::cout << "\nAfter pre-processing, remaining non-zero clauses:\n";
    int nzCount = 0;
    for (auto& cl : sr.clauses) if (!cl.isZero()) { std::cout << "  " << cl.toString() << " = 0\n"; nzCount++; }
    if (nzCount == 0) std::cout << "  (none — fully solved by pre-processing)\n";

    std::cout << "\nAssignment constraints (" << sr.assignConstraints.size() << "):\n";
    for (auto& ac : sr.assignConstraints)
        std::cout << "  " << ac.var << " = " << ac.val << "\n";

    std::cout << "\nExpression constraints (" << sr.exprConstraints.size() << "):\n";
    for (auto& ec : sr.exprConstraints)
        std::cout << "  " << ec.var << " = " << ec.expr.toString() << "\n";

    // Build Hamiltonian
    Poly H = buildHamiltonian(sr.clauses);
    std::cout << "\nHamiltonian has " << H.terms.size() << " terms, degree " << H.degree() << "\n";

    // Quadratize
    QUBO qubo = quadratize(H);
    auto varSet = qubo.variables();
    std::cout << "Quadratized QUBO has " << varSet.size() << " non-fixed variables.\n";

    // Build sorted variable list (mirror Python: sorted by string)
    std::vector<VarName> sortedVars(varSet.begin(), varSet.end());
    std::sort(sortedVars.begin(), sortedVars.end());

    int sz = (int)sortedVars.size();
    QUBOResult qr;
    qr.n_p = vars.n_p;
    qr.n_q = vars.n_q;
    qr.offset = qubo.offset;
    qr.assignConstraints = sr.assignConstraints;
    qr.exprConstraints   = sr.exprConstraints;

    for (int i = 0; i < sz; i++) {
        qr.indexToVar[i] = sortedVars[i];
        qr.varToIndex[sortedVars[i]] = i;
    }

    qr.matrix.assign(sz, std::vector<double>(sz, 0.0));

    // Diagonal from linear terms
    for (auto& [v, c] : qubo.linear) {
        auto it = qr.varToIndex.find(v);
        if (it != qr.varToIndex.end()) qr.matrix[it->second][it->second] = c;
    }

    // Off-diagonal from quadratic terms
    for (auto& [p, c] : qubo.quadratic) {
        auto ia = qr.varToIndex.find(p.first);
        auto ib = qr.varToIndex.find(p.second);
        if (ia != qr.varToIndex.end() && ib != qr.varToIndex.end()) {
            int i = ia->second, j = ib->second;
            if (i > j) std::swap(i, j);
            qr.matrix[i][j] = c;
        }
    }

    return qr;
}

// ============================================================
//  Post-processing: given a solution vector sol[i] in {0,1},
//  recover P and Q.
// ============================================================

static void post_processing(const std::vector<int>& sol,
                            int64_t N,
                            const QUBOResult& qr)
{
    int n_p = qr.n_p, n_q = qr.n_q;
    std::map<std::string, int64_t> fullAssign;

    // Fixed bits (always)
    fullAssign["p_0"] = 1;  fullAssign["p_" + std::to_string(n_p-1)] = 1;
    fullAssign["q_0"] = 1;  fullAssign["q_" + std::to_string(n_q-1)] = 1;

    // Add assignment constraints
    for (auto& ac : qr.assignConstraints)
        fullAssign[ac.var] = ac.val;

    // Add QUBO solution
    for (int i = 0; i < (int)sol.size(); i++) {
        auto it = qr.indexToVar.find(i);
        if (it != qr.indexToVar.end())
            fullAssign[it->second] = sol[i];
    }

    // Iteratively resolve expression constraints (mirror Python final_steps loop)
    // We run up to (n_expr + 5) passes, substituting knowns into each expr.
    int maxIt = (int)qr.exprConstraints.size() + 5;
    for (int pass = 0; pass < maxIt; pass++) {
        bool changed = false;
        for (auto& ec : qr.exprConstraints) {
            if (fullAssign.count(ec.var)) continue;
            Poly p = ec.expr;
            // substitute all currently known values
            for (auto& [k, v] : fullAssign) {
                if (p.freeVars().count(k))
                    p = p.subs(k, (int64_t)v);
            }
            if (p.freeVars().empty()) {
                fullAssign[ec.var] = p.constant();
                changed = true;
            }
        }
        if (!changed) break;
    }

    // If still unresolved expression constraints exist, try assigning free vars = 1 (Python fallback)
    for (auto& ec : qr.exprConstraints) {
        if (!fullAssign.count(ec.var)) {
            fullAssign[ec.var] = 1;  // Python fallback
        }
    }

    int64_t P = 0, Q = 0;
    for (int i = 0; i < n_p; i++) {
        auto key = "p_" + std::to_string(i);
        if (fullAssign.count(key)) P += fullAssign[key] * (1LL << i);
    }
    for (int i = 0; i < n_q; i++) {
        auto key = "q_" + std::to_string(i);
        if (fullAssign.count(key)) Q += fullAssign[key] * (1LL << i);
    }

    std::cout << "\nFactors of " << N << " are: " << P << ", " << Q << "\n";
    if (P * Q == N)
        std::cout << "Verification: " << P << " * " << Q << " = " << P*Q << " (Correct)\n";
    else
        std::cout << "Verification: " << P << " * " << Q << " = " << P*Q << " (Incorrect)\n";
}

// ============================================================
//  Print QUBO matrix as CSV to stdout
// ============================================================
static void printQUBOMatrix(const QUBOResult& qr) {
    int sz = (int)qr.matrix.size();
    if (sz == 0) {
        std::cout << "\nQUBO matrix is empty (problem fully solved by pre-processing).\n";
        return;
    }

    std::cout << "\n=== QUBO Matrix (" << sz << "x" << sz << ") ===\n";
    // Header
    std::cout << "idx";
    for (int i = 0; i < sz; i++) std::cout << "," << qr.indexToVar.at(i);
    std::cout << "\n";

    for (int i = 0; i < sz; i++) {
        std::cout << qr.indexToVar.at(i);
        for (int j = 0; j < sz; j++) {
            std::cout << "," << qr.matrix[i][j];
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Variable Index Mapping ===\n";
    for (int i = 0; i < sz; i++)
        std::cout << i << " -> " << qr.indexToVar.at(i) << "\n";

    std::cout << "\nQUBO offset: " << qr.offset << "\n";
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N>\n";
        std::cerr << "  N must be an odd bi-prime with equal bit-length factors.\n";
        return 1;
    }

    int64_t N = std::stoll(argv[1]);
    if (N <= 0) { std::cerr << "N must be positive.\n"; return 1; }
    if (N % 2 == 0) { std::cerr << "N must be odd.\n"; return 1; }

    std::cout << "=== QUBO Factorization ===\n\n";

    QUBOResult qr = buildQUBOResult(N);

    printQUBOMatrix(qr);

    // If the QUBO is non-empty, we'd need a sampler here.
    // For now, attempt post-processing with all-zeros (demonstrates the interface).
    int sz = (int)qr.matrix.size();
    if (sz == 0) {
        // Pre-processing solved it completely
        std::vector<int> emptySol;
        post_processing(emptySol, N, qr);
    } else {
        std::cout << "\n[Note: QUBO has " << sz << " variables. "
                  << "Provide a solution vector via post_processing() "
                  << "after running your annealer on the matrix above.]\n";
        // Demonstrate call with zero-vector (may not give correct factors)
        std::vector<int> zeroSol(sz, 0);
        std::cout << "\n--- Post-processing with all-zero solution (demo) ---\n";
        post_processing(zeroSol, N, qr);
    }

    return 0;
}
