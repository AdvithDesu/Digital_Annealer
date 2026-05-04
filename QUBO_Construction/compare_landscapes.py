"""Compare two post-quadratization QUBOs at the (p, q)-projected level.
Aux-var names differ between C++ and Python (w_0 vs w0, etc.), so direct
diff is misleading. What matters for SA is the energy landscape over (p, q)
after minimizing the auxiliaries.

For several (P, Q) settings — both correct and random — compute
  E(P, Q) = min over aux vars of QUBO(p_bits, q_bits, w)
and compare across both QUBOs. If equal at every test point, the QUBOs
are equivalent for SA purposes."""
import sys
import re
import random


def parse(path):
    linear, quad = {}, {}
    offset = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.match(r"# offset:\s*(\S+)", line)
                if m:
                    offset = float(m.group(1))
                continue
            parts = line.split()
            v = float(parts[0])
            names = parts[1:]
            if len(names) == 1:
                linear[names[0]] = linear.get(names[0], 0.0) + v
            elif len(names) == 2:
                a, b = sorted(names)
                quad[(a, b)] = quad.get((a, b), 0.0) + v
    return offset, linear, quad


def vars_in(linear, quad):
    s = set(linear)
    for a, b in quad:
        s.add(a); s.add(b)
    return s


def is_pq(name):
    return bool(re.match(r"^[pq]_\d+$", name))


def pin_pq(linear, quad, allvars, P, Q):
    fixed = {}
    for v in allvars:
        m = re.match(r"^p_(\d+)$", v)
        if m:
            fixed[v] = (P >> int(m.group(1))) & 1
            continue
        m = re.match(r"^q_(\d+)$", v)
        if m:
            fixed[v] = (Q >> int(m.group(1))) & 1
    free = sorted(v for v in allvars if v not in fixed)
    new_lin = {v: 0.0 for v in free}
    new_quad = {}
    const = 0.0
    for v, c in linear.items():
        if v in fixed: const += c * fixed[v]
        else:          new_lin[v] += c
    for (a, b), c in quad.items():
        af, bf = a in fixed, b in fixed
        if af and bf:        const += c * fixed[a] * fixed[b]
        elif af and not bf:  new_lin[b] += c * fixed[a]
        elif bf and not af:  new_lin[a] += c * fixed[b]
        else:                new_quad[(a, b) if a < b else (b, a)] = new_quad.get((a, b) if a < b else (b, a), 0.0) + c
    return new_lin, new_quad, const, free


def greedy_min(lin, quad, free, restarts=10000, max_passes=500, seed=0):
    rng = random.Random(seed)
    nbrs = {v: [] for v in free}
    for (a, b), c in quad.items():
        if a in nbrs and b in nbrs:
            nbrs[a].append((b, c)); nbrs[b].append((a, c))
    best = float("inf")
    for r in range(restarts):
        if r == 0:   assign = {v: 0 for v in free}
        elif r == 1: assign = {v: 1 for v in free}
        else:        assign = {v: rng.randint(0, 1) for v in free}
        e = sum(c*assign[v] for v, c in lin.items()) + sum(c*assign[a]*assign[b] for (a,b), c in quad.items())
        for p in range(max_passes):
            any_flip = False
            order = list(free); rng.shuffle(order)
            for v in order:
                cur = assign[v]
                field = lin.get(v, 0.0)
                for n, c in nbrs[v]: field += c * assign[n]
                dE = (1 - 2 * cur) * field
                if dE < -1e-12:
                    assign[v] = 1 - cur; e += dE; any_flip = True
            if not any_flip: break
        if e < best: best = e
    return best


def projected_energy(qubo, P, Q):
    offset, linear, quad = qubo
    allvars = vars_in(linear, quad)
    lin2, quad2, const, free = pin_pq(linear, quad, allvars, P, Q)
    return offset + const + greedy_min(lin2, quad2, free)


def main():
    cpp_path = sys.argv[1]
    py_path = sys.argv[2]
    P_true = int(sys.argv[3])
    Q_true = int(sys.argv[4])
    n_p = (P_true.bit_length() if P_true.bit_length() % 2 == 0 else P_true.bit_length())
    nbits = max(P_true.bit_length(), Q_true.bit_length())

    cpp = parse(cpp_path)
    py = parse(py_path)

    print(f"C++:    offset={cpp[0]}  linear={len(cpp[1])}  quad={len(cpp[2])}  vars={len(vars_in(cpp[1], cpp[2]))}")
    print(f"Python: offset={py[0]}  linear={len(py[1])}  quad={len(py[2])}  vars={len(vars_in(py[1], py[2]))}")

    rng = random.Random(42)
    test_points = [(P_true, Q_true)]
    for _ in range(5):
        test_points.append((rng.getrandbits(nbits) | 1, rng.getrandbits(nbits) | 1))

    print(f"\n{'(P, Q)':30s} {'E_cpp':>14s} {'E_py':>14s} {'diff':>14s}")
    print("-" * 78)
    all_match = True
    for P, Q in test_points:
        E_c = projected_energy(cpp, P, Q)
        E_p = projected_energy(py, P, Q)
        d = E_c - E_p
        marker = "" if abs(d) < 1e-6 else "  *** DIFFERENT ***"
        if abs(d) >= 1e-6: all_match = False
        print(f"({P}, {Q})".ljust(30) + f" {E_c:14.4f} {E_p:14.4f} {d:14.4f}{marker}")

    print()
    if all_match:
        print("*** Energy landscapes (over (p,q)) are IDENTICAL — QUBOs equivalent for SA ***")
    else:
        print("*** Energy landscapes DIFFER — quadratization is producing different effective Hamiltonians ***")


if __name__ == "__main__":
    main()
