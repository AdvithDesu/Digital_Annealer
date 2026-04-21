from pyqubo import Binary
from pyqubo import Spin
from pyqubo import Array

import math
from math import *

import sympy
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import time

#Quadrize functions

def n3(a,b,c,bqm,n,interaction):
  bqm.add_variable('w'+str(n),-2*interaction)
  bqm.add_interaction('w'+str(n),a,interaction)
  bqm.add_interaction('w'+str(n),b,interaction)
  bqm.add_interaction('w'+str(n),c,interaction)
  return bqm

def n4(a,b,c,d,bqm,n,interaction):
  bqm.add_variable('w'+str(n),-3*interaction)
  bqm.add_interaction('w'+str(n),a,interaction)
  bqm.add_interaction('w'+str(n),b,interaction)
  bqm.add_interaction('w'+str(n),c,interaction)
  bqm.add_interaction('w'+str(n),d,interaction)
  return bqm

def p3(a,b,c,bqm,n,interaction):

  bqm.add_variable('w'+str(n),1*interaction)
  bqm.add_interaction('w'+str(n),c,interaction)
  bqm.add_interaction('w'+str(n),b,-1*interaction)
  bqm.add_interaction('w'+str(n),a,-1*interaction)

  for i in bqm.quadratic.keys():
    if a in i and b in i:
      bqm.quadratic[i]+=interaction
      break
  else:
    bqm.quadratic[(a,b)]=interaction

  return bqm

def p4(a,b,c,d,bqm,n,interaction):
  bqm.add_variable('w'+str(n),interaction)
  bqm.add_variable('w#'+str(n),interaction)

  bqm.add_interaction('w'+str(n),a,-1*interaction)
  bqm.add_interaction('w#'+str(n),d,interaction)
  bqm.add_interaction('w'+str(n),b,-1*interaction)
  bqm.add_interaction('w#'+str(n),c,-1*interaction)
  bqm.add_interaction('w'+str(n),c,interaction)
  bqm.add_interaction('w#'+str(n),'w'+str(n),-1*interaction)

  for i in bqm.quadratic.keys():
    if a in i and b in i:
      bqm.quadratic[i]+=interaction
      break
  else:
    bqm.quadratic[(a,b)]=interaction

  return bqm

def quadrizate(bqm):
    #Linear
    del_list=[]
    testset=[]
    intdict={}
    for i in bqm.linear.keys():
        if ' * ' in i:
            del_list.append(i)
            parts = i.split(' * ')
            if len(parts) == 3:
                a, b, c = parts
                uniqcheck = sorted({a, b, c})
                if uniqcheck in testset:
                    intdict[tuple(uniqcheck)] += bqm.linear[i]
                else:
                    testset.append(uniqcheck)
                    intdict[tuple(uniqcheck)] = bqm.linear[i]
                continue

            a,b=parts
            for j in bqm.quadratic.keys():
                if a in j and b in j:
                    bqm.quadratic[j]+=bqm.linear[i]
                    break
            else:
                bqm.quadratic[(a,b)]=bqm.linear[i]

    #Quadratic
    for i in list(bqm.quadratic.keys()):

        # --- MODIFICATION IS INSIDE THIS BLOCK ---
        if ' * ' in i[0] and ' * ' in i[1]:
            parts_a = i[0].split(' * ')
            parts_b = i[1].split(' * ')

            # Original logic for the standard ('p1*q1', 'p2*q2') case
            if len(parts_a) == 2 and len(parts_b) == 2:
                a,b=parts_a
                c,d=parts_b
                uniqcheck={a,b,c,d}
                uniqcheck=sorted(uniqcheck)
                if uniqcheck in testset and len(uniqcheck)!=2:
                    intdict[tuple(uniqcheck)]+=bqm.quadratic[i]
                    continue
                else:
                    testset.append(uniqcheck)
                    intdict[tuple(uniqcheck)]=bqm.quadratic[i]
                    continue
            # New logic for complex cases like ('p1*p2*q1', 'p2*q1')
            else:
                all_vars = parts_a + parts_b
                uniqcheck = sorted(set(all_vars)) # Use set to get unique variables
                if uniqcheck in testset:
                    intdict[tuple(uniqcheck)] += bqm.quadratic[i]
                else:
                    testset.append(uniqcheck)
                    intdict[tuple(uniqcheck)] = bqm.quadratic[i]
                continue
        # --- END OF MODIFICATION ---

        elif ' * ' in i[0]:
            parts = i[0].split(' * ')
            if len(parts) == 3:
                a, b, c = parts
                d = i[1]
                uniqcheck = sorted({a, b, c, d})
                if uniqcheck in testset:
                    intdict[tuple(uniqcheck)] += bqm.quadratic[i]
                else:
                    testset.append(uniqcheck)
                    intdict[tuple(uniqcheck)] = bqm.quadratic[i]
                continue

            a,b=parts
            c=i[1]
            uniqcheck={a,b,c}
            uniqcheck=sorted(uniqcheck)
            if len(uniqcheck)!=3:
                a1,b1=list(uniqcheck)
                for j in bqm.quadratic.keys():
                    if a1 in j and b1 in j:
                        bqm.quadratic[j]+=bqm.quadratic[i]
                        break
                else:
                    bqm.quadratic[(a1,b1)]=bqm.quadratic[i]
                continue
            if uniqcheck in testset:
                intdict[tuple(uniqcheck)]+=bqm.quadratic[i]
                continue
            else:
                testset.append(uniqcheck)
                intdict[tuple(uniqcheck)]=bqm.quadratic[i]

        elif ' * ' in i[1]:
            parts = i[1].split(' * ')
            if len(parts) == 3:
                a, b, c = parts
                d = i[0]
                uniqcheck = sorted({a, b, c, d})
                if uniqcheck in testset:
                    intdict[tuple(uniqcheck)] += bqm.quadratic[i]
                else:
                    testset.append(uniqcheck)
                    intdict[tuple(uniqcheck)] = bqm.quadratic[i]
                continue

            a,b=parts
            c=i[0]
            uniqcheck={a,b,c}
            uniqcheck=sorted(uniqcheck)
            if len(uniqcheck)!=3:
                a1,b1=list(uniqcheck)
                for j in bqm.quadratic.keys():
                    if a1 in j and b1 in j:
                        bqm.quadratic[j]+=bqm.quadratic[i]
                        break
                else:
                    bqm.quadratic[(a1,b1)]=bqm.quadratic[i]
                continue
            if uniqcheck in testset:
                intdict[tuple(uniqcheck)]+=bqm.quadratic[i]
                continue
            else:
                testset.append(uniqcheck)
                intdict[tuple(uniqcheck)]=bqm.quadratic[i]

    n=0
    for i in intdict.keys():
        l=list(i)
        if len(l)==3 and intdict[i]>0:
            bqm=p3(l[0],l[1],l[2],bqm,n,intdict[i])
        elif len(l)==3 and intdict[i]<0:
            bqm=n3(l[0],l[1],l[2],bqm,n,intdict[i])
        elif len(l)==4 and intdict[i]>0:
            bqm=p4(l[0],l[1],l[2],l[3],bqm,n,intdict[i])
        elif len(l)==4 and intdict[i]<0:
            bqm=n4(l[0],l[1],l[2],l[3],bqm,n,intdict[i])
        n+=1

    for i in del_list:
        del bqm.linear[i]
    return bqm


def initialize_variables(N):###
    # Determine the number of bits for P and Q
    n_m = ceil(log2(N + 1))
    #n_p=floor(log2(N))-1
    n_q=ceil(0.5*log2(N))
    n_p=n_q

    print(f"Factoring N = {N} ({n_m} bits)")
    print(f"Assuming n_p = {n_p}, n_q = {n_q}")

    # Create symbolic variables for the unknown bits of P and Q
    # p_0 and p_{n_p-1} are fixed to 1
    p = {i: sympy.symbols(f'p_{i}') for i in range(1, n_p)}
    p[0] = 1
    p[n_p-1]=1 ##

    # q_0 and q_{n_q-1} are fixed to 1
    q = {i: sympy.symbols(f'q_{i}') for i in range(1, n_q)}
    q[0] = 1
    q[n_q-1]=1 ##

    # Create symbolic variables for the carry bits
    # s[j, i] represents a carry from column j to column i
    s = defaultdict(lambda: 0)
    max_sum_terms = n_q + (n_q - 1) # Max terms in a central column  #Waste term
    max_carry_val = max_sum_terms   #Waste term
    for i in range(1, n_p+n_q):
        # Estimate max possible sum in column i to determine needed carries
        num_prod_terms = min(i, n_q - 1) - max(i - n_p + 1, 0) + 1
        num_carry_in = i - 1 # A rough upper bound
        max_sum_in_col = num_prod_terms + num_carry_in

        if max_sum_in_col > 1:
            num_carry_out_bits = floor(log2(max_sum_in_col))
            for j in range(1, num_carry_out_bits + 1):
                if i + j < n_p+n_q:
                    s[i, i + j] = sympy.symbols(f's_{i}_{i+j}')

    return n_p, n_q, p, q, s

def generate_column_clauses(N, n_p, n_q, p, q, s):

    """
    Generates the initial symbolic column clauses for the factorization of N.

    Args:
        N (int): The integer to be factored.
        n_p (int): Number of bits for P.
        n_q (int): Number of bits for Q.
        p (dict): Dictionary of symbolic variables for P.
        q (dict): Dictionary of symbolic variables for Q.
        s (dict): Dictionary of symbolic carry variables.
   Returns:
        list: A list of SymPy expressions, one for each column clause C_i.
    """
    n_m = n_p+n_q
    N_bits = [int(bit) for bit in bin(N)[2:].zfill(n_m)]
    N_bits.reverse() # LSB first
    clauses =[]
    # Column 0 is trivial as p_0=q_0=1 and N must be odd (N_0=1)
    # C_0 = N_0 - p_0*q_0 = 1 - 1*1 = 0
    for i in range(1, n_m):
        # 1. Start with the i-th bit of N
        clause = N_bits[i]

        # 2. Subtract product terms p_j * q_{i-j}
        for j in range(n_p):
            if 0 <= i - j < n_q:
                clause -= p[j] * q[i - j]

        # 3. Subtract input carries s_{k,i}
        for k in range(1, i):
            if s[k, i]!= 0:
                clause -= s[k, i]

        # 4. Add output carries 2^j * s_{i,i+j}
        # The value of a carry from column i to i+j is 2^j relative to column i
        j = 1
        while s[i, i + j]!= 0:
            clause += (2**j) * s[i, i + j]
            if i+j>n_p+n_q-1: ##Just in cases where initialize variables is not used and s[i,i+j] is present where i+j>np+nq-1
              clauses.append(s[i,i+j])
            j += 1
        clauses.append(clause)

    return clauses


def apply_rule_1_and_2(clause):
    """Applies Rule 1 (sum = n) and Rule 2 (sum = 0)."""
    new_constraints = {}
    if not isinstance(clause, sympy.Expr):
       print("Erroneous input")
       return new_constraints

    # Isolate variables and constant term
    terms = clause.as_coefficients_dict()
    const = terms.pop(1, 0)

    # Check for sum = n (Rule 1)
    if all(c == -1 for c in terms.values()) and const == len(terms):
        for var in terms:
            new_constraints[var] = 1
    if all(c==1 for c in terms.values()) and -const == len(terms):
            for var in terms:
              new_constraints[var] = 1
    # Check for sum = 0 (Rule 2)
    if all(c == 1 for c in terms.values()) and const == 0:
        for var in terms:
            new_constraints[var] = 0
    if all(c == -1 for c in terms.values()) and const == 0:
        for var in terms:
            new_constraints[var] = 0

    return new_constraints

def apply_rule_3(clause):
    """Applies Rule 3 (x1 + x2 = 2*x3 => x1=x2=x3). Prioritizes substituting 's' variables."""
    new_constraints = {}
    x1, x2, x3 = sympy.Wild('x1'), sympy.Wild('x2'), sympy.Wild('x3')
    pattern = x1 + x2 - 2*x3
    mul=[]
    mul_ass={}
    ind=0

    match = clause.match(pattern)
    if match:
        # Extract the variables and handle potential non-symbolic matches
        v1 = match[x1]
        v2 = match[x2]
        v3 = match[x3]
        for v in [v1, v2, v3]:
            dih=v.as_coefficients_dict()
            i=0
            for key,value in dih.items():
                if value!=1 or i==1:
                    ind+=1
                    break
                i+=1
            if ind==1:
              break
            if v.is_Mul:
                  mul.append(v)#contains terms like p1*q1 and so on have to be treated seperately

        if ind==1:
          clause=-clause
          mul=[]
          match = clause.match(pattern)
          if match:
              # Extract the variables and handle potential non-symbolic matches
              v1 = match[x1]
              v2 = match[x2]
              v3 = match[x3]
              for v in [v1, v2, v3]:
                  dih=v.as_coefficients_dict()
                  i=0
                  for key,value in dih.items():
                      if value!=1 or i==1:
                          return new_constraints,mul_ass
                      i+=1
                  if v.is_Mul:
                        mul.append(v)#contains terms like p1*q1 and so on have to be treated seperatel

        # Rule 3 implies v1 = v2 = v3.
        # Prioritize substituting 's' variables.
        # If any variable is 's', substitute it with one of the others.
        s_vars = [v for v in [v1, v2, v3] if 's_' in str(v)]
        other_vars = [v for v in [v1, v2, v3] if v not in s_vars]
        if len(mul)==0:
            if s_vars:
                # Substitute all s_vars with the first other_var, or the first s_var if no other_vars
                sub_target = other_vars[0] if other_vars else s_vars[0]
                for s_var in s_vars:
                    if s_var != sub_target:
                        new_constraints[s_var] = sub_target
                # If there are other_vars, ensure they are also equal
                if len(other_vars) > 1:
                    new_constraints[other_vars[1]] = other_vars[0] # Assuming at most 2 other_vars in this rule
            else:
                # If no 's' variables, all are p/q, set them equal
                new_constraints[v1] = v2
                new_constraints[v3] = v2

        else:
          if len(s_vars)==2:
            new_constraints[s_vars[0]] = s_vars[1]
            mul_ass[s_vars[0]]=mul[0]
            mul_ass[s_vars[1]]=mul[0]
          elif len(s_vars)==1:
            if other_vars[0].is_Symbol:
                new_constraints[s_vars[0]] = other_vars[0]
                mul_ass[other_vars[1]]=other_vars[0]
            elif other_vars[1].is_Symbol:
                new_constraints[s_vars[0]] = other_vars[1]
                mul_ass[other_vars[0]]=other_vars[1]
            else:
              mul_ass[s_vars[0]]=other_vars[0]
              mul_ass[other_vars[1]]=other_vars[0]
          else:

            if len(mul)==3:
              mul_ass[other_vars[0]]=other_vars[2]
              mul_ass[other_vars[1]]=other_vars[2]

            elif len(mul)==2:
              if other_vars[0].is_Symbol:
                mul_ass[other_vars[1]]=other_vars[0]
                mul_ass[other_vars[2]]=other_vars[0]
              elif other_vars[1].is_Symbol:
                mul_ass[other_vars[0]]=other_vars[1]
                mul_ass[other_vars[2]]=other_vars[1]
              else:
                mul_ass[other_vars[0]]=other_vars[2]
                mul_ass[other_vars[1]]=other_vars[2]

            else:
                if other_vars[0].is_Mul:
                  mul_ass[other_vars[0]]=other_vars[1]
                  #mul_ass[other_vars[2]]=other_vars[1]
                  new_constraints[other_vars[2]]=other_vars[1]
                elif other_vars[1].is_Mul:
                  mul_ass[other_vars[1]]=other_vars[0]
                  #mul_ass[other_vars[2]]=other_vars[0]
                  new_constraints[other_vars[2]]=other_vars[0]
                else:
                  #mul_ass[other_vars[0]]=other_vars[1]
                  mul_ass[other_vars[2]]=other_vars[1]
                  new_constraints[other_vars[0]]=other_vars[1]


        return new_constraints,mul_ass

    return new_constraints,mul_ass

def apply_rule_4(clause):
    """Applies Rule 4, e.g., x1+x2 = 2*x3+1 => x3=0."""
    new_constraints = {}
    # Specific case from paper: x1 + x2 - 2*x3 - 1 = 0 => x3 = 0
    """x1, x2, x3 = sympy.Wild('x1'), sympy.Wild('x2'), sympy.Wild('x3')
    pattern = x1 + x2 - 2*x3 - 1
    match = clause.match(pattern)
    if match and match[x1].is_Symbol and match[x2].is_Symbol and match[x3].is_Symbol:
            new_constraints[match[x3]] = 0
            return new_constraints
    else:
      clause=-clause
      match = clause.match(pattern)
      if match and match[x1].is_Symbol and match[x2].is_Symbol and match[x3].is_Symbol:
            new_constraints[match[x3]] = 0
            return new_constraints"""

    # General case
    terms = clause.as_coefficients_dict()
    const = terms.pop(1, 0)

    # Check for c_i > n - c_0 (from Eq. 11)
    # Reformulated for clause = 0: c_0 + sum(c_i*y_i) - sum(x_i) = 0
    # Let positive terms be y's and negative terms be x's
    pos_terms = {v: c for v, c in terms.items() if c > 0}
    neg_terms = {v: c for v, c in terms.items() if c < 0}
    pos_sum = sum(pos_terms.values())
    neg_sum = -1*sum(neg_terms.values())
    for y, c_y in pos_terms.items():
            if c_y > neg_sum-const:
              new_constraints[y] = 0
            if c_y>pos_sum+const:
              new_constraints[y] = 1

    for x, c_x in neg_terms.items():
            if -c_x > pos_sum+const:
              new_constraints[x] = 0
            if -c_x>neg_sum-const:
              new_constraints[x] = 1

    return new_constraints

def apply_rule_6(clause):###Changed
  new_constraints = {}
  # General case
  terms = clause.as_coefficients_dict()
  const = terms.pop(1, 0)

  # Check for c_i > n - c_0 (from Eq. 11)
  # Reformulated for clause = 0: c_0 + sum(c_i*y_i) - sum(x_i) = 0
  # Let positive terms be y's and negative terms be x's

  pos_terms = {v: c for v, c in terms.items() if c > 0}
  neg_terms = {v: -c for v, c in terms.items() if c < 0}
  pos_sort = sorted(pos_terms.items(), key=lambda item: item[1], reverse=True)
  neg_sort = sorted(neg_terms.items(), key=lambda item: item[1], reverse=True)
  pos_sum = sum(pos_terms.values())
  neg_sum = sum(neg_terms.values())

  if pos_sum > neg_sum-const and const<0:
    for i in range(len(pos_sort)):
      y, c_y = pos_sort[i]
      for j in range(i+1, len(pos_sort)):
        x, c_x = pos_sort[j]
        k=0
        if c_y+c_x > neg_sum-const:
          if -const-pos_sum+c_y+c_x>0:##Can optimise the equality part

            if 's' in str(y):
              new_constraints[y] = 1-x
            elif 's' in str(x):
              new_constraints[x] = 1-y
            else:
              new_constraints[y]=1-x
            break    ####Have to optimise here###
        else:
          k+=1
          break
      if k==1:
        break

  if neg_sum > pos_sum+const and const>0:
    for i in range(len(neg_sort)):
      y, c_y = neg_sort[i]


      for j in range(i+1, len(neg_sort)):

        x, c_x = neg_sort[j]

        k=0
        if c_y+c_x > pos_sum+const:
          if neg_sum-const-c_y-c_x<0:##Can optimise the equality part
            if 's' in str(y):
              new_constraints[y] = 1-x
            elif 's' in str(x):
              new_constraints[x] = 1-y
            else:
              new_constraints[y]=1-x
            break
        else:
          k+=1
          break
      if k==1:
        break

  return new_constraints

def apply_parity_rule(clause):
    """Applies the Parity Rule."""
    new_constraints = {}
    if not isinstance(clause, sympy.Expr):
        return new_constraints

    terms = clause.as_coefficients_dict()

    odd_terms_sum = sympy.sympify(0)
    for term, coeff in terms.items():
        if coeff % 2!= 0:
            odd_terms_sum += term * (coeff % 2) # Add term with coeff 1 or -1

    # The sum of odd-coefficient terms must have even parity.
    # If odd_terms_sum simplifies to x1 + x2, it implies x1=x2.
    # If it simplifies to x1 + x2 + 1, it implies x1 + x2 = 1.

    # Case 1: x1 + x2 = 0 (implies x1=x2)
    x1, x2 = sympy.Wild('x1'), sympy.Wild('x2')
    match = odd_terms_sum.match(x1 + x2)
    if match and match[x1].is_Symbol and match[x2].is_Symbol:
      if 's' in str(match[x1]):
        new_constraints[match[x1]] = match[x2]
      elif 's' in str(match[x2]):
        new_constraints[match[x2]] = match[x1]
      else:
        new_constraints[match[x1]]=match[x2]
      return new_constraints

    # Case 2: x1 - x2 = 0 (implies x1=x2)
    match = odd_terms_sum.match(x1 - x2)
    if match and match[x1].is_Symbol and match[x2].is_Symbol:
        if 's' in str(match[x1]):
          new_constraints[match[x1]] = match[x2]
        elif 's' in str(match[x2]):
          new_constraints[match[x2]] = match[x1]
        else:
         new_constraints[match[x1]]=match[x2]
        return new_constraints

    # Case 3: x1 + x2 + 1 = 0 (implies x1+x2=1)
    match = odd_terms_sum.match(x1 + x2 + 1)
    if match and match[x1].is_Symbol and match[x2].is_Symbol:
        if 's' in str(match[x1]):
          new_constraints[match[x1]] = 1-match[x2]
        elif 's' in str(match[x2]):
          new_constraints[match[x2]] =1- match[x1]
        else:
         new_constraints[match[x1]]=1-match[x2]
        return new_constraints

    #Case 4:x1+x2-1=0
    match = odd_terms_sum.match(x1 + x2 - 1)
    if match and match[x1].is_Symbol and match[x2].is_Symbol:
        if 's' in str(match[x1]):
          new_constraints[match[x1]] = 1-match[x2]
        elif 's' in str(match[x2]):
          new_constraints[match[x2]] =1- match[x1]
        else:
         new_constraints[match[x1]]=1-match[x2]
        return new_constraints

    return new_constraints


def power_rule(expr):
    """
    Reduce a SymPy polynomial assuming all variables are binary (x^n = x for n >= 1).
    """
    expr = sympy.expand(expr)
    new_terms = {}

    for term, coeff in expr.as_coefficients_dict().items():
        if term == 1:
            # constant term
            new_terms[1] = new_terms.get(1, 0) + coeff
        else:
            # collapse powers of variables: x^n -> x
            factors = []
            for symbol, power in term.as_powers_dict().items():
                factors.append(symbol)  # ignore power, since binary
            reduced_term = sympy.Mul(*factors)
            new_terms[reduced_term] = new_terms.get(reduced_term, 0) + coeff

    # reconstruct polynomial
    reduced_expr = sum(coeff * term for term, coeff in new_terms.items())
    return sympy.expand(reduced_expr)

def apply_power_rule(clauses):
    new_clauses = []
    for clause in clauses:
        reduced_clause = power_rule(clause)
        new_clauses.append(reduced_clause)
    return new_clauses


# @title
def replacement(clauses):
    constraints={}
    """Applies the replacement rule to eliminate variables."""
    for i, clause in enumerate(clauses):
        if clause == 0:
          continue

        # Find variables in the clause
        variables = list(clause.free_symbols)

        # Try to solve for one variable
        for var in variables:
            coeff = clause.as_coefficients_dict().get(var, None)
            if coeff!=1 and coeff!=-1:
              continue
            try:
                # solve() returns a list of solutions
                solutions = sympy.solve(clause, var)

                if len(solutions) == 1:
                    # Found a unique solution for var
                    sol = solutions

                    # Do not replace primary variables (p, q) (Is this needed?)
                    if 's_' in str(var):#Remove this temporarily
                        constraints[var] = sol[0]
                        #clauses[i] = sympy.sympify(0) # This clause is now resolved
                        return constraints # Indicate a change was made

            except Exception:
                continue # Cannot solve for this variable
    return {}

# @title
def get_qubo_from_clause(clauses):
    # Combine all squared clauses
    hamiltonian = clauses

    # Expand the full expression
    expanded = sympy.expand(hamiltonian)

    # Collect variables
    variables = list(expanded.free_symbols)
    binary_vars = {str(var): Binary(str(var)) for var in variables}

    # Build the Hamiltonian manually
    qubo_ham = 0
    for term, coeff in expanded.as_coefficients_dict().items():
        if term == 1:  # constant term
            qubo_ham += coeff
        else:
            # term may be x, x*y, etc.
            factor = 1
            for symbol in term.free_symbols:
                factor *= binary_vars[str(symbol)]
            qubo_ham += coeff * factor

    return qubo_ham

def qubo_hamiltonian_from_clauses(clauses):
  H=0
  for clause in clauses:
    if clause!=0:
      h=get_qubo_from_clause(clause)
      H+=h*h

  return H


def common_factor(clause):
    """
    Finds the greatest common divisor of the *integer* coefficients
    in a SymPy expression. Assumes all coefficients are integers.
    """
    if not isinstance(clause, sympy.Expr):
        return 1 # No common factor if not an expression

    terms = clause.as_coefficients_dict()

    # Include the constant term in the coefficients and convert to integers
    # Removed redundant sympify(c), as 'c' is already a SymPy object
    coeffs = [abs(int(c)) for c in terms.values()]

    if not coeffs: # Handle the case of an empty expression (e.g., 0)
        return 1

    # Calculate the GCD of all coefficients
    gcd_val = coeffs[0]
    for i in range(1, len(coeffs)):
        gcd_val = math.gcd(gcd_val, coeffs[i])
        if gcd_val == 1:
            break # Optimization
    if gcd_val==0:
      return 1
    return gcd_val

#Applying substitution to all clauses
def sub_clause(clauses,constraints):
  for i in range(len(clauses)):
    for var,val in constraints.items():
      clauses[i]=clauses[i].subs(var,val)
  return clauses


#Whole classical pre-processing
TRACE = True
TRACE_FILE = None  # set externally
def _canon(val):
    if isinstance(val, (int, float)):
        return str(int(val))
    expr = sympy.expand(val)
    parts = []
    const = 0
    for term, coeff in expr.as_coefficients_dict().items():
        if term == 1:
            const += int(coeff)
        else:
            vars_in_term = sorted(str(s) for s in term.free_symbols)
            parts.append((vars_in_term, int(coeff)))
    parts.sort()
    s = str(const)
    for vars_, c in parts:
        sign = "+" if c >= 0 else "-"
        s += f" {sign} {abs(c)}*{'*'.join(vars_)}"
    return s

def _trace(it, rule, ci, var, val):
    if not TRACE: return
    line = f"TRACE iter={it} rule={rule} clause={ci} elim={var} val={_canon(val)}"
    if TRACE_FILE is not None:
        TRACE_FILE.write(line + "\n")
    else:
        print(line)

def clause_simplifier(initial_clauses):
  clauses=initial_clauses
  assignment_constraints=[]
  expression_constraints=[]
  mul_constraints=[]
  print("Total iterations possible:",2*len(initial_clauses))
  for outer_iter in range(2*len(initial_clauses)):
      #print("Clause 1:",clauses[0])
      ind=0
      #Removing common factor
      for i in range(len(clauses)):
            gcd=common_factor(clauses[i])
            clauses[i]=clauses[i]/gcd

      for ci, clause in enumerate(clauses):
            constraints=apply_rule_1_and_2(clause)
            constraints_={}#Just for using in product=1 process without causing changes to dictionary while iterating over it
            if len(constraints)!=0:
                          for i in constraints.keys():
                            constraints_[i]=constraints[i]
                            if 'p' in str(i) or 'q' in str(i):
                              #print(constraints)
                              assignment_constraints.append({i:constraints[i]})
                              if i.is_Mul and constraints[i]==1:
                                        print("Product=1 detected:",i,constraints[i])
                                        # Extract individual factors from the multiplication term
                                        variables = list(i.free_symbols)
                                        # Assuming the constraint is that the product is 1, which for binary
                                        # variables implies each variable in the product must be 1.
                                        for var in variables:
                                            # Ensure the variable is a Symbol before adding the constraint
                                            if isinstance(var, sympy.Symbol):
                                                constraints_[var]=1
                                                assignment_constraints.append({var: 1})
                          constraints=constraints_
                          ind=1
                          for _v, _val in constraints.items():
                              _trace(outer_iter, "rule_1_2", ci, str(_v), _val)
                          clauses=sub_clause(clauses,constraints)
                          break


      for ci, clause in enumerate(clauses):
            constraints=apply_rule_4(clause)
            constraints_={}#Just for using in product=1 process without causing changes to dictionary while iterating over it
            if len(constraints)!=0:
                          for i in constraints.keys():
                            constraints_[i]=constraints[i]
                            if 'p' in str(i) or 'q' in str(i):
                              #print(constraints)
                              assignment_constraints.append({i:constraints[i]})
                              if i.is_Mul and constraints[i]==1:
                                    print("Product=1 detected:",i,constraints[i])
                                    # Extract individual factors from the multiplication term
                                    variables = list(i.free_symbols)
                                    # Assuming the constraint is that the product is 1, which for binary
                                    # variables implies each variable in the product must be 1.
                                    for var in variables:
                                        # Ensure the variable is a Symbol before adding the constraint
                                        if isinstance(var, sympy.Symbol):
                                            constraints_[var]=1
                                            assignment_constraints.append({var: 1})
                          constraints=constraints_
                          ind=1
                          for _v, _val in constraints.items():
                              _trace(outer_iter, "rule_4", ci, str(_v), _val)
                          clauses=sub_clause(clauses,constraints)
                          break

      for ci, clause in enumerate(clauses):
            constraints,mul_constraint=apply_rule_3(clause)
            if len(mul_constraint)!=0:
              #print("Mul_Contraint:",mul_constraint)
              mul_constraints.append(mul_constraint)
            if len(constraints)!=0:
              for i in constraints.keys():
                if 'p' in str(i) or 'q' in str(i):
                  #print(constraints)
                  expression_constraints.append({i:constraints[i]})
              ind=1
              for _v, _val in constraints.items():
                  _trace(outer_iter, "rule_3", ci, str(_v), _val)
              clauses=sub_clause(clauses,constraints)
              clauses=apply_power_rule(clauses)
              break

      for ci, clause in enumerate(clauses):
            constraints=apply_rule_6(clause)
            if len(constraints)!=0:
              for i in constraints.keys():
                if 'p' in str(i) or 'q' in str(i):
                  #print(constraints)
                  expression_constraints.append({i:constraints[i]})
              ind=1
              for _v, _val in constraints.items():
                  _trace(outer_iter, "rule_6", ci, str(_v), _val)
              clauses=sub_clause(clauses,constraints)
              clauses=apply_power_rule(clauses)
              break

      for ci, clause in enumerate(clauses):
            constraints=apply_parity_rule(clause)
            if len(constraints)!=0:
              for i in constraints.keys():
                if 'p' in str(i) or 'q' in str(i):
                  #print(constraints)
                  expression_constraints.append({i:constraints[i]})
              ind=1
              for _v, _val in constraints.items():
                  _trace(outer_iter, "parity", ci, str(_v), _val)
              clauses=sub_clause(clauses,constraints)
              clauses=apply_power_rule(clauses)
              break

      if ind==0:
        constraints=replacement(clauses)
        if len(constraints)!=0:
              for i in constraints.keys():
                if 'p' in str(i) or 'q' in str(i):
                  #print(constraints)
                  expression_constraints.append({i:constraints[i]})
              for _v, _val in constraints.items():
                  _trace(outer_iter, "replacement", -1, str(_v), _val)
              clauses=sub_clause(clauses,constraints)
              clauses=apply_power_rule(clauses)
              #print("replaced")
              #print("Constraint:",constraints)
              #print("Simplified clause:",clauses)
        else:
          break
      #print("Number of without replacement iterations:",i)
  return (clauses,assignment_constraints,expression_constraints,mul_constraints)


def final_steps(model,N,np,nq,addn_constr,addn_ass):
  solv_ind=0
  full_assignment={}
  # The original 'if model!=0:' check causes TypeError when 'model' is a pyqubo.Express object
  # We need to explicitly check if it's the Python integer 0, which would indicate
  # that the Hamiltonian was entirely simplified to zero by classical pre-processing.
  if not (isinstance(model, int) and model == 0):
      model=model.compile()
      bqm=model.to_bqm()
      print("Compiled BQM:", bqm)
      bqm=quadrizate(bqm)
      print("Quadratized BQM has {} non-fixed variables.".format(len(bqm.variables)))

    #   sampler = SimulatedAnnealingSampler()
    #   sampleset = sampler.sample(bqm, num_reads=1000)
    #   best_solution = sampleset.first

    #   print(f"Energy: {best_solution.energy}")
    #   print(f"Best solution from sampler:",best_solution.sample)

    #  # Create a complete assignment dictionary including sampled variables and fixed p_0, q_0
    #   full_assignment = dict(best_solution.sample)
      solv_ind=1

   #Compile additional assignments
  for ass_dict in addn_ass:
    for i,v in ass_dict.items():
        if sympy.sympify(i).is_Mul:
          continue
        else:
          full_assignment[str(i)]=v # Convert Symbol to string for consistent keys

  initial_len = len(full_assignment)
  #if full ass is null to begin with find random constraint to assign 1
  if solv_ind==0:
      #Assign other constr dependent quantities first
      for _ in range(len(addn_constr) + 5): # Iterate a few times to resolve dependencies
          changes_made = False
          for constr_dict in addn_constr: # Iterate through the list of dictionaries
              # constr_dict is a dictionary like {variable: expression}
              for var, expr in constr_dict.items(): # Get the single item from the dictionary
                  if str(var) not in full_assignment: # Ensure variable name is string for lookup
                      # Try to substitute values from full_assignment into the expression
                      try:
                          # Create a substitution dictionary for sympy
                          subs_dict = {sympy.Symbol(str(k)): v for k, v in full_assignment.items()}
                          substituted_expr = expr.subs(subs_dict)

                          # If the expression simplifies to a numerical value, add it to full_assignment
                          if substituted_expr.is_number:
                              full_assignment[str(var)] = int(substituted_expr) # Assuming binary values
                              changes_made = True
                              #print(f"Resolved {var} = {substituted_expr}")
                      except Exception as e:
                          # Substitution failed, maybe due to unresolved variables in expr
                          #print(f"Could not resolve {var} = {expr}: {e}")
                          pass # Keep trying in the next iteration

          if not changes_made and len(full_assignment) >= initial_len + len(addn_constr):
              # No new variables resolved in this iteration and we've potentially resolved all added constraints
              break


      ind=0
      for const_dict in addn_constr:
        for u,v in const_dict.items():
          if str(u) not in full_assignment.keys(): # Convert Symbol to string for consistent lookup
            full_assignment[str(u)]=1 # Convert Symbol to string for consistent keys
            #print("Assigned 1 to",u)
            ind=1
            break
        if ind==1:
          break


  initial_len = len(full_assignment)
  print(full_assignment)
  # --- START: New, more robust substitution loop ---

    # 1. Convert all constraint dicts to a list of sympy.Eq objects
  all_equations = []
  for constr_dict in addn_constr:
      for var, expr in constr_dict.items():
          # Create a true equation: var = expr
          all_equations.append(sympy.Eq(var, expr))

  # 2. Iteratively solve the system
  # We still loop to allow for multi-step dependencies (e.g., solve A, use A to solve B)
  for _ in range(len(all_equations) + 5): # Iterate enough times to resolve
      changes_made = False
      # Create the substitution dictionary from all currently known values
      subs_dict = {sympy.Symbol(str(k)): v for k, v in full_assignment.items()}

      for eq in all_equations:
          # Substitute all known values into this equation
          substituted_eq = eq.subs(subs_dict)

          # Get all remaining (unknown) variables in this equation
          unknowns = substituted_eq.free_symbols

          # Check if the equation now has exactly ONE unknown variable
          if len(unknowns) == 1:
              unknown_var = list(unknowns)[0]

              # If we haven't already solved for this variable, solve it
              if str(unknown_var) not in full_assignment:
                  try:
                      # Use sympy.solve() to find the value of the unknown
                      solution = sympy.solve(substituted_eq, unknown_var)

                      # Check if solve found a single, numerical solution
                      if solution and isinstance(solution, list) and len(solution) == 1 and solution[0].is_number:

                          solved_value = int(solution[0])
                          full_assignment[str(unknown_var)] = solved_value
                          changes_made = True

                          # IMPORTANT: Update subs_dict *immediately* # This allows the next equation in *this* pass to use this new value
                          subs_dict[unknown_var] = solved_value

                          # print(f"Solved: {unknown_var} = {solved_value}") # <-- Good for debugging

                  except Exception as e:
                      # print(f"Could not solve for {unknown_var} in {substituted_eq}: {e}")
                      pass # Skip if solving fails (e.g., non-linear, no solution)

      # If no progress was made in a full pass, we're done
      if not changes_made:
          break

    # --- END: New substitution loop ---

  print("After trying to solve:",full_assignment)
  P = 0
  Q = 0

  # Reconstruct P and Q from the resolved bits using original n_p and n_q
  # Initialize with fixed bits based on initialize_variables logic (p_0=1, p_{n_p-1}=1, q_0=1, q_{n_q-1}=1)
  # Ensure these are included in full_assignment or handled here if they aren't variables in the BQM
  # Based on initialize_variables, p_0, p_{n_p-1}, q_0, q_{n_q-1} are Symbols.
  # We need to get their resolved values from full_assignment.

  p_bits = {}
  q_bits = {}

  # Populate p_bits and q_bits from full_assignment
  for var_name, value in full_assignment.items():
      if 'p_' in var_name:
          try:
              bit_pos = int(var_name.split('_')[1])
              p_bits[bit_pos] = value
          except ValueError:
              pass # Not a standard p_i variable
      elif 'q_' in var_name:
          try:
              bit_pos = int(var_name.split('_')[1])
              q_bits[bit_pos] = value
          except ValueError:
              pass # Not a standard q_i variable

  # Ensure fixed bits are in the dictionaries, using 1 as default if not resolved
  # (This might need refinement if fixed bits can be resolved to 0 based on constraints)
  # We are assuming p_0, q_0, p_np-1, q_nq-1 are 1 as per initialize_variables.
  # If they are part of full_assignment due to being variables in the BQM,
  # their values from full_assignment will be used. Otherwise, default to 1.
  if 0 not in p_bits:
      p_bits[0] = 1
  if np-1 not in p_bits:
      p_bits[np-1] = 1
  if 0 not in q_bits:
      q_bits[0] = 1
  if nq-1 not in q_bits:
      q_bits[nq-1] = 1


  # Reconstruct P and Q by summing all bits with their power of 2
  P = 0
  for bit_pos in range(np):
      if bit_pos in p_bits:
          P += int(p_bits[bit_pos]) * (2 ** bit_pos)
      # Else: bit value is not resolved, assume 0 or handle as needed
      # For binary factoring, unresolved bits can sometimes be inferred as 0 or 1 based on other constraints
      # For now, we only sum explicitly assigned bits.

  Q = 0
  for bit_pos in range(nq):
      if bit_pos in q_bits:
          Q += int(q_bits[bit_pos]) * (2 ** bit_pos)
      # Else: bit value is not resolved

  # Convert P and Q to standard Python integers for verification
  P_int = int(P)
  Q_int = int(Q)

  #print("P_vars:",p_bits) # Use p_bits and q_bits for printing
  #print("Q_vars:",q_bits)

  print(f"Factors of {N} are: {P_int}, {Q_int}")
  if P_int * Q_int == N:
      print(f"Verification: {P_int} * {Q_int} = {P_int*Q_int} (Correct)")
  else:
      print(f"Verification: {P_int} * {Q_int} = {P_int*Q_int} (Incorrect)")


#N has to be a bi-prime with factors being equal bit length for proper factorization
def factorize2(N):
  n_p,n_q,p,q,s=initialize_variables(N)
  initial_clauses=generate_column_clauses(N, n_p, n_q, p, q, s)
  clauses,ass,expr,mul=clause_simplifier(initial_clauses)
  H=qubo_hamiltonian_from_clauses(clauses)
  final_steps(H,N,n_p,n_q,expr,ass)


def get_qubo_matrix2(N):
    n_p, nq, p, q, s = initialize_variables(N)
    initial_clauses = generate_column_clauses(N, n_p, nq, p, q, s)
    clauses, ass, expr, mul = clause_simplifier(initial_clauses)
    model = qubo_hamiltonian_from_clauses(clauses)
    model = model.compile()
    bqm = model.to_bqm()
    bqm = quadrizate(bqm)

    print("Compiled BQM:", bqm)
    print("Quadratized BQM has {} non-fixed variables.".format(len(bqm.variables)))

    qubo_dict, offset = bqm.to_qubo()

    # --- FIX: SORT VARIABLES TO ENSURE STABLE ORDERING ---
    # We sort by string representation to ensure deterministic mapping
    sorted_variables = sorted(list(bqm.variables), key=str)

    variable_to_index = {v: i for i, v in enumerate(sorted_variables)}
    index_to_variable = {i: v for v, i in variable_to_index.items()}
    # -----------------------------------------------------

    size = len(sorted_variables)
    qubo_matrix = np.zeros((size, size))

    neighbors = defaultdict(list)   # adjacency list
    h = np.zeros(N, dtype=np.float32)
    offset = 0.0

    for (u, v), value in qubo_dict.items():
        if u in variable_to_index and v in variable_to_index:
            i = variable_to_index[u]
            j = variable_to_index[v]
            qubo_matrix[i, j] = value

    print(f"qubo_dict is {qubo_dict}")

    return qubo_matrix, index_to_variable, ass, expr, mul



def post_processing2(sol,N,ass_constr,addn_constr,index_to_variable):#Assuming the solution from SPIM is given as a list of 0s and 1s and sol[i] represents the solution of the ith index of qubo matrix
      full_assignment = {}
      for ass_dict in ass_constr:
        for i,v in ass_dict.items():
          if sympy.sympify(i).is_Mul:
            continue
          else:
            full_assignment[str(i)]=v # Convert Symbol to string for consistent keys



      for i in range(len(sol)):
        full_assignment[index_to_variable[i]]=sol[i]

      initial_len = len(full_assignment)
    # Simple iterative substitution loop for addn_constr (can be improved for complex dependencies)
      for _ in range(len(addn_constr) + 5): # Iterate a few times to resolve dependencies
          changes_made = False
          for constr_dict in addn_constr: # Iterate through the list of dictionaries
              # constr_dict is a dictionary like {variable: expression}
              for var, expr in constr_dict.items(): # Get the single item from the dictionary
                  if str(var) not in full_assignment: # Ensure variable name is string for lookup
                      # Try to substitute values from full_assignment into the expression
                      try:
                          # Create a substitution dictionary for sympy
                          subs_dict = {sympy.Symbol(str(k)): v for k, v in full_assignment.items()}
                          substituted_expr = expr.subs(subs_dict)

                          # If the expression simplifies to a numerical value, add it to full_assignment
                          if substituted_expr.is_number:
                              full_assignment[str(var)] = int(substituted_expr) # Assuming binary values
                              changes_made = True
                              #print(f"Resolved {var} = {substituted_expr}")
                      except Exception as e:
                          # Substitution failed, maybe due to unresolved variables in expr
                          #print(f"Could not resolve {var} = {expr}: {e}")
                          pass # Keep trying in the next iteration

          if not changes_made and len(full_assignment) >= initial_len + len(addn_constr):
              # No new variables resolved in this iteration and we've potentially resolved all added constraints
              break


      # Manually add p_0 and q_0 if they are not present and needed for factor calculation
      # This requires knowing n_p and n_q from initialization
      # For now, let's assume they are always 1 unless determined otherwise by constraints
      # A better way might be to include them in the initial variable set and handle them in constraints
      # based on the problem definition (p_0=1, q_0=1).

      # Calculate the factors P and Q
      P = 0
      Q = 0

      # Need to know the original variable names and their bit positions (0 to n_p-1, 0 to n_q-1)
      # This information should ideally be passed from initialize_variables

      # As a workaround, iterate through resolved variables and check if they are p or q bits
      # This assumes the variable naming convention 'p_i' and 'q_i'
      p_vars = {}
      q_vars = {}

      for var_name, value in full_assignment.items():
          if 'p_' in var_name:
              try:
                  bit_pos = int(var_name.split('_')[1])
                  p_vars[bit_pos] = value
              except ValueError:
                  pass # Not a standard p_i variable
          elif 'q_' in var_name:
              try:
                  bit_pos = int(var_name.split('_')[1])
                  q_vars[bit_pos] = value
              except ValueError:
                  pass # Not a standard q_i variable

      # Add p_0=1 and q_0=1 if not already present (assuming this is the rule)
      # This is a fallback; ideally, p_0 and q_0 should be part of the variable set and constraints
      # Determine n_p and n_q from the variables found, or pass them in
      # For now, let's assume n_p and n_q can be inferred or are small enough.
      # A more robust solution would pass n_p and n_q from initialize_variables.
      # For N=15 (n_p=3, n_q=3), p_0, p_1, p_2 and q_0, q_1, q_2 are expected.
      # Let's try to infer max bit position to guess n_p, n_q
      #max_p_bit = max(p_vars.keys(), default=-1)
      #max_q_bit = max(q_vars.keys(), default=-1)
      #inferred_n_p = max_p_bit + 1
      #inferred_n_q = max_q_bit + 1

      nq=ceil(0.5*log2(N))
      np=nq

      if 0 not in p_vars:
          p_vars[0] = 1
      if 0 not in q_vars:
          q_vars[0] = 1

      if np-1 not in p_vars:
          p_vars[np-1] = 1

      if nq-1 not in q_vars:
          q_vars[nq-1] = 1

      # Ensure all expected bit positions for p and q have values (either sampled or from constraints)
      # If not, it means the constraints or sampling did not provide a full solution
      # This part might need refinement based on how missing variables should be handled
      # (e.g., assuming 0 if not determined, which is not always correct for binary)
      # For now, we'll proceed with available values.


      # Reconstruct P and Q from the resolved bits
      # Use sorted keys to ensure correct bit position weighting
      P = 0  # Initialize P as a standard Python integer
      Q = 0  # Initialize Q as a standard Python integer

      for bit_pos in sorted(p_vars.keys()):
          P += int(p_vars[bit_pos]) * (2 ** bit_pos)

      for bit_pos in sorted(q_vars.keys()):
          Q += int(q_vars[bit_pos]) * (2 ** bit_pos)

      # Convert P and Q to standard Python integers for verification
      P_int = int(P)
      Q_int = int(Q)

      print(f"Factors of {N} are: {P_int}, {Q_int}")
      if P_int * Q_int == N:
          print(f"Verification: {P_int} * {Q_int} = {P_int*Q_int} (Correct)")
      else:
          print(f"Verification: {P_int} * {Q_int} = {P_int*Q_int} (Incorrect)")


def get_J_h_matrix(N):
    n_p, nq, p, q, s = initialize_variables(N)
    initial_clauses = generate_column_clauses(N, n_p, nq, p, q, s)
    clauses, ass, expr, mul = clause_simplifier(initial_clauses)
    model = qubo_hamiltonian_from_clauses(clauses).compile()
    bqm = quadrizate(model.to_bqm())

    print("Compiled BQM:", bqm)
    print("Quadratized BQM has {} non-fixed variables.".format(len(bqm.variables)))

    qubo_dict, _ = bqm.to_qubo()

    # ---- stable variable ordering ----
    sorted_vars = sorted(list(bqm.variables), key=str)
    var_to_idx = {v: i for i, v in enumerate(sorted_vars)}
    ind_to_var = {i: v for v, i in var_to_idx.items()}
    size = len(sorted_vars)

    from collections import defaultdict

    # ---- STEP 1: accumulate Q and Q^T ----
    Q_acc = defaultdict(float)
    for (u, v), val in qubo_dict.items():
        i = var_to_idx[u]
        j = var_to_idx[v]
        Q_acc[(i, j)] += float(val)

    # ---- STEP 2: build Q_sym = (Q + Q^T) / 2 ----
    Q_sym = defaultdict(float)
    visited = set()

    for (i, j), val in Q_acc.items():
        if (i, j) in visited:
            continue

        if i == j:
            Q_sym[(i, i)] = val
            visited.add((i, i))
        else:
            v_ij = val
            v_ji = Q_acc.get((j, i), 0.0)
            sym_val = 0.5 * (v_ij + v_ji)

            Q_sym[(i, j)] = sym_val
            Q_sym[(j, i)] = sym_val

            visited.add((i, j))
            visited.add((j, i))

    # ---- STEP 3: build h and sparse J (DA1-consistent) ----
    h = np.zeros(size, dtype=np.float32)
    neighbors = defaultdict(list)
    offset = 0.0

    for (i, j), Qij in Q_sym.items():
        if i == j:
            h[i] += 0.5 * Qij
            offset += 0.5 * Qij
        else:
            Jij = Qij / 2.0   # matches main_old
            neighbors[i].append((j, Jij))
            h[i] += 0.5 * Qij
            offset += 0.25 * Qij

    # ---- STEP 4: CSR construction ----
    row_ptr = np.zeros(size + 1, dtype=np.int32)
    col_idx = []
    values = []

    nnz = 0
    for i in range(size):
        row_ptr[i] = nnz
        if i in neighbors:
            neighbors[i].sort(key=lambda x: x[0])
            for j, Jij in neighbors[i]:
                col_idx.append(j)
                values.append(Jij)
                nnz += 1
    row_ptr[size] = nnz

    return row_ptr, np.array(col_idx, dtype=np.int32), np.array(values, dtype=np.float32), h, offset, ind_to_var, ass, expr, mul


def save_csr_and_h(row_ptr, col_idx, values, h, N):
    np.savetxt(f"row_ptr_{N}.csv", row_ptr, fmt="%d", delimiter=",")

    np.savetxt(f"col_idx_{N}.csv", col_idx, fmt="%d", delimiter=",")

    np.savetxt(f"J_values_{N}.csv", values, fmt="%.8f", delimiter=",")

    pd.DataFrame(h.reshape(1, -1)).to_csv(f'h_vector_{N}.csv', index=False, header=False)


N = 159197

# Note the time taken to execute the two lines below
start_time = time.time()
row_ptr, col_idx, values, h, offset, ind_to_var, ass, expr, mul = get_J_h_matrix(N)
save_csr_and_h(row_ptr, col_idx, values, h, N)
end_time = time.time()

print(f"Time taken to construct CSR J: {end_time - start_time} seconds")

# post_processing2(solution_list,N,ass,expr,ind_to_var)