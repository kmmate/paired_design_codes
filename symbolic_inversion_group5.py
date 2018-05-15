"""
Computes symbolic inverse for the 4x4 matrix based on group-wise comparison of groups of pairs with different treatment
assigment: (treatment, treatment), (no treatment, treatment), (treatment, no treatment), (no treatment, no treatment).
Does the inversion on Page G5/3.
"""

import numpy as np
from numpy import linalg
import sympy as sp
sp.init_printing(use_unicode=True)

"""
create symbolic expressions
"""

P, p_j, p_jp = sp.symbols('P p_j p_jp')

# matrix entries
a_j = P * (1 - P) * p_j
a_jp = P * (1 - P) * p_jp

M = sp.Matrix([[1, a_j, a_jp, 0], [P, a_j, 0, 0], [P, 0, a_jp, 0], [P ** 2, 0, 0, 0]])

"""
compute determinant and inverse
"""
np.random.seed([20180205])
# checking invertibility
M_det = M.det()
print('\n -------- Determinant -----------\n')
print('det(M)= ', sp.factor(M_det))
print('\n ------------ Inverse  -----------\n')
M_inv = sp.simplify(M.inv())
for row in range(4):
    print('\n Mi_nv[%s,:] = '  % str(row))
    print(sp.simplify(M_inv[row, 0]))
    print(sp.simplify(M_inv[row, 1]))
    print(sp.simplify(M_inv[row, 2]))
    print(sp.simplify(M_inv[row, 3]))


