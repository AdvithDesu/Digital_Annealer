"""Dump Python's post-quadratization QUBO in the same format as
C++ --dump-quad-qubo. Uses pyqubo's compile + quadrizate path."""
import sys

src = open("QUBO_Integer_Factorization.py").read()
cut = src.find("def final_steps(")
ns = {}
exec(src[:cut], ns)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 51959
    out_file = sys.argv[2] if len(sys.argv) > 2 else "quad_qubo_py.txt"

    n_p, n_q, p, q, s = ns["initialize_variables"](N)
    initial_clauses = ns["generate_column_clauses"](N, n_p, n_q, p, q, s)
    clauses, ass, expr_constr, mul = ns["clause_simplifier"](initial_clauses)

    H = ns["qubo_hamiltonian_from_clauses"](clauses)
    model = H.compile()
    bqm = model.to_bqm()
    bqm = ns["quadrizate"](bqm)
    qubo_dict, offset = bqm.to_qubo()

    rows = []
    for (a, b), v in qubo_dict.items():
        if a == b:
            rows.append((a, a, v))
        else:
            x, y = (a, b) if a < b else (b, a)
            rows.append((x, y, v))
    rows.sort()

    with open(out_file, "w") as f:
        f.write(f"# offset: {repr(offset)}\n")
        f.write(f"# entries: {len(rows)}\n")
        for a, b, v in rows:
            if a == b:
                f.write(f"{repr(v)} {a}\n")
            else:
                f.write(f"{repr(v)} {a} {b}\n")
    print(f"Dumped {len(rows)} entries (offset={offset}) to {out_file}")


if __name__ == "__main__":
    main()
