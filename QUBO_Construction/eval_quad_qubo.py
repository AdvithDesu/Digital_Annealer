"""Read a post-quadratization QUBO dump (--dump-quad-qubo format), pin
p_i/q_i variables to the bits of given true (P, Q), greedily minimize over
remaining (w_*) auxiliary variables, and report residual energy.

If quadratization is correct, residual = 0. A nonzero residual is direct
evidence of a quadratization bug.

Usage: python eval_quad_qubo.py <quad_qubo_dump> <P> <Q> [--label CPP|PY]
"""
import sys
import re


def parse_quad_qubo(path):
    """Returns (linear: dict[name->coeff], quadratic: dict[(a,b)->coeff], offset, vars: set)."""
    linear = {}
    quadratic = {}
    offset = 0.0
    vars_ = set()
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
            coeff = float(parts[0])
            names = parts[1:]
            for n in names:
                vars_.add(n)
            if len(names) == 1:
                linear[names[0]] = linear.get(names[0], 0.0) + coeff
            elif len(names) == 2:
                a, b = sorted(names)
                quadratic[(a, b)] = quadratic.get((a, b), 0.0) + coeff
    return linear, quadratic, offset, vars_


def pin_pq(linear, quadratic, vars_, P, Q):
    """Substitute p_i/q_i bits into the QUBO. Returns (lin, quad, const, w_vars)
    over remaining (presumably w_*) variables only."""
    fixed = {}
    for v in vars_:
        m = re.match(r"^p_(\d+)$", v)
        if m:
            fixed[v] = (P >> int(m.group(1))) & 1
            continue
        m = re.match(r"^q_(\d+)$", v)
        if m:
            fixed[v] = (Q >> int(m.group(1))) & 1
            continue
    free = sorted(v for v in vars_ if v not in fixed)
    print(f"  fixed {len(fixed)} p/q vars, {len(free)} free (w_*) vars")

    new_lin = {v: 0.0 for v in free}
    new_quad = {}
    const = 0.0

    for v, c in linear.items():
        if v in fixed:
            const += c * fixed[v]
        else:
            new_lin[v] += c

    for (a, b), c in quadratic.items():
        af = a in fixed
        bf = b in fixed
        if af and bf:
            const += c * fixed[a] * fixed[b]
        elif af and not bf:
            new_lin[b] += c * fixed[a]
        elif bf and not af:
            new_lin[a] += c * fixed[b]
        else:
            key = (a, b) if a < b else (b, a)
            new_quad[key] = new_quad.get(key, 0.0) + c

    return new_lin, new_quad, const, free


def energy(lin, quad, assign):
    e = 0.0
    for v, c in lin.items():
        e += c * assign[v]
    for (a, b), c in quad.items():
        e += c * assign[a] * assign[b]
    return e


def greedy_min(lin, quad, free, max_passes=300, restarts=2000, seed=0):
    """Greedy single-bit-flip with multiple random restarts."""
    import random
    rng = random.Random(seed)
    # Build neighbor list once
    neighbors = {v: [] for v in free}
    for (a, b), c in quad.items():
        if a in neighbors and b in neighbors:
            neighbors[a].append((b, c))
            neighbors[b].append((a, c))

    best_e = float("inf")
    best_assign = None
    best_passes = 0
    for r in range(restarts):
        if r == 0:
            assign = {v: 0 for v in free}
        elif r == 1:
            assign = {v: 1 for v in free}
        else:
            assign = {v: rng.randint(0, 1) for v in free}
        e = energy(lin, quad, assign)
        for p in range(max_passes):
            any_flip = False
            order = list(free)
            rng.shuffle(order)
            for v in order:
                cur = assign[v]
                field = lin.get(v, 0.0)
                for n, c in neighbors[v]:
                    field += c * assign[n]
                dE_flip = (1 - 2 * cur) * field
                if dE_flip < -1e-12:
                    assign[v] = 1 - cur
                    e += dE_flip
                    any_flip = True
            if not any_flip:
                break
        if e < best_e:
            best_e = e
            best_assign = dict(assign)
            best_passes = p + 1
    return best_e, best_assign, best_passes


def main():
    path = sys.argv[1]
    P = int(sys.argv[2])
    Q = int(sys.argv[3])
    label = "QUBO"
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]

    print(f"=== {label} : {path} ===")
    linear, quadratic, offset, vars_ = parse_quad_qubo(path)
    print(f"  parsed: {len(linear)} linear, {len(quadratic)} quadratic, offset={offset}")
    print(f"  total vars: {len(vars_)}")

    lin2, quad2, const2, free = pin_pq(linear, quadratic, vars_, P, Q)
    pinned_energy = offset + const2
    print(f"  energy from pinned p/q (incl offset): {pinned_energy:.6f}")

    e_min, assign, passes = greedy_min(lin2, quad2, free)
    total = pinned_energy + e_min
    print(f"  greedy min over w (passes={passes}): {e_min:.6f}")
    print(f"  TOTAL residual energy at true (P,Q): {total:.6f}")
    if abs(total) < 1e-6:
        print("  *** quadratization OK at true (P,Q): energy = 0 ***")
    else:
        print(f"  *** RESIDUAL = {total:.6f} -- inconsistent with H(true)=0 ***")


if __name__ == "__main__":
    main()
