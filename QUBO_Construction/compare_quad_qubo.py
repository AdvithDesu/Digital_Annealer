"""Numerically compare two post-quadratization QUBO dumps, ignoring float
formatting differences."""
import sys
import re


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


def main():
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    o1, l1, q1 = parse(p1)
    o2, l2, q2 = parse(p2)

    print(f"file1: {p1}\n  offset={o1}  linear={len(l1)}  quad={len(q1)}")
    print(f"file2: {p2}\n  offset={o2}  linear={len(l2)}  quad={len(q2)}")

    EPS = 1e-9
    print(f"\noffset diff: {abs(o1 - o2)}")

    keys_lin = set(l1) | set(l2)
    keys_quad = set(q1) | set(q2)
    diff_lin = [k for k in keys_lin if abs(l1.get(k, 0) - l2.get(k, 0)) > EPS]
    diff_quad = [k for k in keys_quad if abs(q1.get(k, 0) - q2.get(k, 0)) > EPS]
    print(f"linear differing : {len(diff_lin)} / {len(keys_lin)}")
    print(f"quadratic differing: {len(diff_quad)} / {len(keys_quad)}")

    if abs(o1 - o2) < EPS and not diff_lin and not diff_quad:
        print("\n*** QUBOs are NUMERICALLY IDENTICAL ***")
        return
    print("\n--- sample diffs ---")
    for k in diff_lin[:5]:
        print(f"  lin {k:30s} f1={l1.get(k,0)} f2={l2.get(k,0)}")
    for k in diff_quad[:5]:
        print(f"  quad {str(k):60s} f1={q1.get(k,0)} f2={q2.get(k,0)}")


if __name__ == "__main__":
    main()
