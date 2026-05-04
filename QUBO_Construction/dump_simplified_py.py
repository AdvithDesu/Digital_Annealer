"""Dump Python's simplified clauses + active variable set in the same
canonical format as C++'s --dump-simplified-clauses (Checkpoint 3)."""
import sys
import sympy

src = open("QUBO_Integer_Factorization.py").read()
cut = src.find("def final_steps(")
ns = {}
exec(src[:cut], ns)


def canonical(expr):
    """Match C++ polyStrCanonical."""
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
    out_file = sys.argv[2] if len(sys.argv) > 2 else "simplified_clauses_py.txt"

    n_p, n_q, p, q, s = ns["initialize_variables"](N)
    initial_clauses = ns["generate_column_clauses"](N, n_p, n_q, p, q, s)
    clauses, ass, expr, mul = ns["clause_simplifier"](initial_clauses)

    canon = []
    var_set = set()
    for c in clauses:
        if c == 0 or c == sympy.S.Zero:
            continue
        if not hasattr(c, "free_symbols"):
            # constant non-zero clause - shouldn't normally happen post-simplifier
            canon.append(canonical(c))
            continue
        canon.append(canonical(c))
        for sym in c.free_symbols:
            var_set.add(str(sym))

    canon.sort()
    with open(out_file, "w") as f:
        f.write(f"# non-zero clauses: {len(canon)}\n")
        f.write(f"# active vars (union of free symbols): {len(var_set)}\n")
        f.write("## CLAUSES (sorted)\n")
        for s in canon:
            f.write(s + "\n")
        f.write("## ACTIVE_VARS (sorted)\n")
        for v in sorted(var_set):
            f.write(v + "\n")
    print(f"Dumped {len(canon)} simplified clauses, {len(var_set)} active vars to {out_file}")


if __name__ == "__main__":
    main()
