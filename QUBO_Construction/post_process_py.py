"""Python re-implementation of the post-process step that mirrors the
intent of post_processing2 (forward-only chain resolution + missing-bit-as-0
behavior). Reads C++'s saved artifacts so we run on identical inputs.

Inputs:
  - assignment_constraints_<N>.txt
  - expression_constraints_<N>.txt
  - index_to_var_<N>.txt
  - <spins_file>          (whitespace-separated +1/-1 or 0/1)
  - n_p, n_q              (auto-derived from N)

Outputs the resolved (P, Q) and prints every step so we can diff against
C++'s behavior bit-for-bit."""
import sys
import re
import sympy
from math import ceil, log2


def parse_assignment(path):
    """Returns dict {var_name -> int}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+)\s*=\s*(-?\d+)$", line)
            if not m:
                continue
            out[m.group(1)] = int(m.group(2))
    return out


def parse_poly_term(s):
    """Parse a single canonical term like '-1*p_3*q_1' or '4*s_4_6' or '1' into
    (coeff:int, list of var names)."""
    s = s.strip()
    parts = s.split("*")
    if not parts:
        return 0, []
    try:
        coeff = int(parts[0])
        names = parts[1:]
    except ValueError:
        # No leading integer: assume coeff 1
        coeff = 1
        names = parts
    return coeff, names


def parse_expression(rhs_str):
    """Parse an expression like '-1*q_1 + 1' into a sympy expression."""
    # Replace ' + -' with ' - ' and split by ' + ' / ' - ' carefully.
    # Easiest: insert explicit signs and split.
    s = rhs_str.strip()
    # Tokenize: split on '+' or '-' but keep the sign
    # Convert to a normalized form: each term prefixed with sign
    s = s.replace(" + ", " +").replace(" - ", " -")
    # If no leading sign, add '+'
    if s and s[0] not in "+-":
        s = "+" + s
    # Now split on whitespace; each token starts with + or -
    tokens = []
    i = 0
    while i < len(s):
        if s[i] in "+-":
            j = i + 1
            while j < len(s) and s[j] != " ":
                j += 1
            tokens.append(s[i:j])
            i = j + 1
        else:
            i += 1
    expr = sympy.S.Zero
    for tok in tokens:
        sign = 1
        body = tok
        if body.startswith("+"):
            body = body[1:]
        elif body.startswith("-"):
            sign = -1
            body = body[1:]
        coeff, names = parse_poly_term(body)
        coeff *= sign
        term = sympy.Integer(coeff)
        for n in names:
            term = term * sympy.Symbol(n)
        expr = expr + term
    return expr


def parse_expressions(path):
    """Returns list of (var_name:str, sympy_expr) in file order."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+)\s*=\s*(.+)$", line)
            if not m:
                continue
            var = m.group(1)
            rhs = parse_expression(m.group(2))
            out.append((var, rhs))
    return out


def parse_index_to_var(path):
    """Returns list ordered by index of var names."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            name = parts[1]
            out.append((idx, name))
    out.sort()
    return [name for _, name in out]


def parse_spins(path):
    """Returns list of 0/1 ints."""
    txt = open(path).read().strip().split()
    out = []
    for s in txt:
        v = int(s)
        out.append(1 if v == 1 else 0 if v == -1 else int(v))
    return out


def post_process_py(N, n_p, n_q, assignments, expressions, idx_to_var, spins, verbose=True):
    """Mirror of Python's post_processing2 logic."""
    full = {}
    # Step 1: load assignment constraints (Python filters to p_/q_ but we load all
    # since C++ records all and they're harmless)
    for k, v in assignments.items():
        full[k] = v
    # Step 2: SA spins
    for i, name in enumerate(idx_to_var):
        full[name] = spins[i]

    if verbose:
        unresolved_pq_pre = []
        for i in range(n_p):
            if "p_" + str(i) not in full:
                unresolved_pq_pre.append("p_" + str(i))
        for i in range(n_q):
            if "q_" + str(i) not in full:
                unresolved_pq_pre.append("q_" + str(i))
        print(f"[py] before chain: {len(unresolved_pq_pre)} p/q unresolved: {unresolved_pq_pre}")

    # Step 3: iteratively resolve expressions (forward only, like post_processing2)
    max_passes = len(expressions) + 5
    for p in range(max_passes):
        changed = False
        for var, expr in expressions:
            if var in full:
                continue
            subs_dict = {sympy.Symbol(k): v for k, v in full.items()}
            sub = expr.subs(subs_dict)
            if sub.is_number:
                full[var] = int(sub)
                changed = True
        if not changed:
            break

    if verbose:
        unresolved_pq_post = []
        for i in range(n_p):
            if "p_" + str(i) not in full:
                unresolved_pq_post.append("p_" + str(i))
        for i in range(n_q):
            if "q_" + str(i) not in full:
                unresolved_pq_post.append("q_" + str(i))
        print(f"[py] after chain : {len(unresolved_pq_post)} p/q unresolved: {unresolved_pq_post}")

    # Step 4 (Python style): default fixed bits if missing
    for k in [f"p_0", f"p_{n_p-1}", f"q_0", f"q_{n_q-1}"]:
        if k not in full:
            full[k] = 1

    # Step 5: reconstruct, missing -> 0
    P = 0
    for i in range(n_p):
        k = f"p_{i}"
        P += int(full.get(k, 0)) * (1 << i)
    Q = 0
    for i in range(n_q):
        k = f"q_{i}"
        Q += int(full.get(k, 0)) * (1 << i)
    return P, Q, full


def main():
    N = int(sys.argv[1])
    spins_path = sys.argv[2]
    n_m = N.bit_length()
    n_q = (n_m + 1) // 2
    n_p = n_q

    print(f"=== Python-style post-process for N={N} ({n_m} bits, n_p={n_p}, n_q={n_q}) ===")
    assigns = parse_assignment(f"assignment_constraints_{N}.txt")
    exprs = parse_expressions(f"expression_constraints_{N}.txt")
    idx_to_var = parse_index_to_var(f"index_to_var_{N}.txt")
    spins = parse_spins(spins_path)
    print(f"  assignments: {len(assigns)}, expressions: {len(exprs)}, "
          f"active vars: {len(idx_to_var)}, spins: {len(spins)}")
    if len(spins) != len(idx_to_var):
        print(f"ERROR: spins length {len(spins)} != active var count {len(idx_to_var)}")
        return

    P, Q, full = post_process_py(N, n_p, n_q, assigns, exprs, idx_to_var, spins)
    print(f"\n[py] P = {P}, Q = {Q}, P*Q = {P*Q}")
    if P * Q == N:
        print(f"[py] CORRECT: {P} * {Q} = {N}")
    else:
        print(f"[py] WRONG: expected {N}, got {P*Q} (diff={P*Q - N})")


if __name__ == "__main__":
    main()
