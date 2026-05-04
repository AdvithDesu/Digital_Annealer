"""Dump raw squared QUBO H = sum_k clause_k^2 in canonical integer form,
matching C++ --dump-raw-qubo (Checkpoint 4a). Pre-quadratization, pre-Ising.

Usage: python dump_raw_qubo_py.py <N> <out_file> [--relabel-p1-to-q1]
  --relabel-p1-to-q1 : substitute p_1 = 1 - q_1 (use on C++ side to compare
                       against Python which kept q_1)
"""
import sys
import sympy

src = open("QUBO_Integer_Factorization.py").read()
cut = src.find("def final_steps(")
ns = {}
exec(src[:cut], ns)


def dump_qubo(expr, out_file):
    expr = sympy.expand(expr)
    coeffs = expr.as_coefficients_dict()
    H = {}  # tuple of sorted var names -> coeff
    constant = 0
    for term, coeff in coeffs.items():
        c = int(coeff)
        if c == 0:
            continue
        if term == 1:
            constant += c
            continue
        if hasattr(term, "free_symbols"):
            names = tuple(sorted(str(s) for s in term.free_symbols))
        else:
            names = (str(term),)
        H[names] = H.get(names, 0) + c
    nonconst = sorted((names, c) for names, c in H.items() if c != 0)
    with open(out_file, "w") as f:
        f.write(f"# raw squared QUBO terms (excluding constant): {len(nonconst)}\n")
        f.write(f"# constant: {constant}\n")
        for names, c in nonconst:
            f.write(str(c))
            for n in names:
                f.write("*" + n)
            f.write("\n")
    print(f"Dumped raw QUBO ({len(nonconst)} non-const terms, const={constant}) to {out_file}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    N = int(args[0]) if args else 51959
    out_file = args[1] if len(args) > 1 else "raw_qubo_py.txt"
    relabel = "--relabel-p1-to-q1" in flags

    n_p, n_q, p, q, s = ns["initialize_variables"](N)
    initial_clauses = ns["generate_column_clauses"](N, n_p, n_q, p, q, s)
    clauses, ass, expr_constr, mul = ns["clause_simplifier"](initial_clauses)

    H = sympy.S.Zero
    for c in clauses:
        if c == 0:
            continue
        H = H + c * c

    if relabel:
        p_1 = sympy.Symbol("p_1")
        q_1 = sympy.Symbol("q_1")
        H = H.subs(p_1, 1 - q_1)
        print("(applied p_1 = 1 - q_1 relabeling)")

    dump_qubo(H, out_file)


if __name__ == "__main__":
    main()
