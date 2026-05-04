"""Read C++ and Python raw QUBO dumps, apply p_1 = 1 - q_1 to the C++ side,
re-canonicalize, and diff. If they match, the QUBO Hamiltonians are
mathematically identical — divergence is in quadratization or Ising conversion."""
import sys
import sympy
import re


def parse_dump(path):
    """Parse a dump file into a sympy expression."""
    expr = sympy.S.Zero
    constant = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                m = re.match(r"# constant:\s*(-?\d+)", line)
                if m:
                    constant = int(m.group(1))
                continue
            # Format: "<int>*<name>*<name>*..."
            parts = line.split("*")
            coeff = int(parts[0])
            term = sympy.Integer(coeff)
            for name in parts[1:]:
                term = term * sympy.Symbol(name)
            expr = expr + term
    expr = expr + sympy.Integer(constant)
    return expr


def canonical_terms(expr):
    """Return dict {sorted-var-tuple: int_coeff} including constant under ()."""
    expr = sympy.expand(expr)
    out = {}
    for term, coeff in expr.as_coefficients_dict().items():
        c = int(coeff)
        if c == 0:
            continue
        if term == 1:
            out[()] = out.get((), 0) + c
        else:
            names = tuple(sorted(str(s) for s in term.free_symbols))
            out[names] = out.get(names, 0) + c
    return out


def main():
    cpp_file = sys.argv[1] if len(sys.argv) > 1 else "raw_qubo_cpp_51959.txt"
    py_file = sys.argv[2] if len(sys.argv) > 2 else "raw_qubo_py_51959.txt"

    print(f"Loading C++  : {cpp_file}")
    H_cpp = parse_dump(cpp_file)
    print(f"Loading Py   : {py_file}")
    H_py = parse_dump(py_file)

    # Apply p_1 = 1 - q_1 to C++ to align with Python's variable choice
    p_1 = sympy.Symbol("p_1")
    q_1 = sympy.Symbol("q_1")
    H_cpp_relabeled = sympy.expand(H_cpp.subs(p_1, 1 - q_1))

    cpp_terms = canonical_terms(H_cpp_relabeled)
    py_terms = canonical_terms(H_py)

    cpp_keys = set(cpp_terms.keys())
    py_keys = set(py_terms.keys())

    only_cpp = cpp_keys - py_keys
    only_py = py_keys - cpp_keys
    common = cpp_keys & py_keys
    differing = [k for k in common if cpp_terms[k] != py_terms[k]]

    print(f"\nC++ (post-relabel) terms : {len(cpp_terms)}")
    print(f"Python terms             : {len(py_terms)}")
    print(f"Common terms             : {len(common)}")
    print(f"Only in C++              : {len(only_cpp)}")
    print(f"Only in Python           : {len(only_py)}")
    print(f"Differing on common      : {len(differing)}")

    if not only_cpp and not only_py and not differing:
        print("\n*** IDENTICAL: raw squared QUBOs match exactly under p_1 = 1 - q_1 ***")
        return

    print("\n--- Sample of disagreement ---")
    for k in list(only_cpp)[:5]:
        print(f"  only C++: {'*'.join(k) or 'CONST':30s}  coeff={cpp_terms[k]}")
    for k in list(only_py)[:5]:
        print(f"  only Py : {'*'.join(k) or 'CONST':30s}  coeff={py_terms[k]}")
    for k in differing[:10]:
        print(f"  differs : {'*'.join(k) or 'CONST':30s}  cpp={cpp_terms[k]}  py={py_terms[k]}")


if __name__ == "__main__":
    main()
