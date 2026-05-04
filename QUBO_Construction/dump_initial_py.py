"""Dump Python's initial column clauses in the same canonical format as
C++'s polyStrCanonical, for diffing against C++ output (Checkpoint 1)."""
import sys
import sympy

# Load Python module up to clause_simplifier (avoid pyqubo at bottom)
src = open("QUBO_Integer_Factorization.py").read()
cut = src.find("def final_steps(")
ns = {}
exec(src[:cut], ns)


def canonical(expr):
    """Match C++ polyStrCanonical: '<const>[ +/- <coeff>*<sorted_vars>]...'"""
    if isinstance(expr, (int, float)):
        return str(int(expr))
    expr = sympy.expand(expr)
    coeffs = expr.as_coefficients_dict()
    constant = 0
    terms = []
    for term, coeff in coeffs.items():
        c = int(coeff)
        if term == 1:
            constant += c
            continue
        # Extract symbol names from the term (Symbol or Mul of Symbols)
        if hasattr(term, "free_symbols"):
            names = sorted(str(s) for s in term.free_symbols)
        else:
            names = [str(term)]
        terms.append((names, c))
    terms.sort()
    out = str(constant)
    for names, c in terms:
        sign = " - " if c < 0 else " + "
        out += sign + str(abs(c))
        for n in names:
            out += "*" + n
    return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 51959
    out_file = sys.argv[2] if len(sys.argv) > 2 else "initial_clauses_py.txt"

    n_p, n_q, p, q, s = ns["initialize_variables"](N)
    clauses = ns["generate_column_clauses"](N, n_p, n_q, p, q, s)

    with open(out_file, "w") as f:
        for i, c in enumerate(clauses):
            f.write(f"C{i + 1}: {canonical(c)}\n")
    print(f"Dumped {len(clauses)} initial clauses to {out_file}")


if __name__ == "__main__":
    main()
