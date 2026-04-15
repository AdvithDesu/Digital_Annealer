import sys
sys.path.insert(0, '.')

# Import the module — but skip the bottom-of-file factorize calls by importing functions directly
import importlib.util
spec = importlib.util.spec_from_file_location("qif", "QUBO_Integer_Factorization.py")
# Patch: prevent the bottom of the file from running the heavy CSR build
import builtins
_orig_print = builtins.print
def _silent(*a, **k):
    pass

# Easier: just import the module and accept the bottom runs once. But that's expensive.
# Instead: import by reading source up to clause_simplifier only.

import sympy
from collections import defaultdict
from math import ceil, log2

# Re-import only what we need by exec'ing the file with the bottom call stripped.
src = open("QUBO_Integer_Factorization.py").read()
# Cut off after clause_simplifier definition (before final_steps) to avoid pyqubo dependencies
# Actually we need apply_power_rule etc. — keep through clause_simplifier
cut_marker = "def final_steps("
idx = src.find(cut_marker)
src_top = src[:idx]
ns = {}
exec(src_top, ns)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 159197
out = sys.argv[2] if len(sys.argv) > 2 else "trace_py.txt"

ns["TRACE"] = True
with open(out, "w") as f:
    ns["TRACE_FILE"] = f
    n_p, n_q, p, q, s = ns["initialize_variables"](N)
    initial_clauses = ns["generate_column_clauses"](N, n_p, n_q, p, q, s)
    clauses, ass, expr, mul = ns["clause_simplifier"](initial_clauses)
print(f"wrote {out}")
print(f"assignments: {len(ass)}, expressions: {len(expr)}")
