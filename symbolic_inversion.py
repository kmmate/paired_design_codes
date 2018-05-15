"""
Thesis.
Computes symbolic inverse of the 4x4 matrix, M. Version 4th February 2018 (corresponding to papers with circled
page numbers).
"""

import numpy as np
import sympy as sp
sp.init_printing(use_unicode=True)

"""
create symbolic expressions
"""

P, p, r, q, qbar = sp.symbols('P p r q qbar')
e1, e2, e3, e4, e5, e6, e7, e8 = sp.symbols('e1:9')


# Matrix values corresponding to top-middle-circled pagenumbers
a = P ** 2 * r + P * p
b = P ** 2 * qbar
c = P ** 2 * q
# Matrix older values corresponding to top-right-circled pagenumbers
#a = P ** 2 * q + (P - P ** 2) * p
#b = P ** 2 * qbar
#c = P ** 2 * q

# Vector values
zy1 = P ** 2 * (e1 * qbar + e2 * r + e3 * r) + P * e4 * p + P * e5 * p + e6
zy2 = P ** 2 * (e1 * qbar + e2 * r + e3 * q) + P * e4 * p + P * e6
zy3 = P ** 2 * (e1 * qbar + e2 * q + e3 * r) + P * e5 * p + P * e6
zy4 = P ** 2 * (e1 * qbar + e8 * q + e7 * q) + P ** 2 * e6

M = sp.Matrix([[1, a, a, b], [P, a, c, b], [P, c, a, b], [P ** 2, c, c, b]])
zy = sp.Matrix([[zy1], [zy2], [zy3], [zy4]])

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

combinations = dict()
for row in range(4):
    print('\n============================================== For row %s the weights in linear combination: ' % str(row))
    combinations[1] = P ** 2 * qbar * sum(sp.simplify(M_inv[row, :]))
    combinations[2] = P ** 2 * r * (sp.simplify(M_inv[row, 0] + M_inv[row, 1]))
    combinations[3] = P ** 2 * r * (sp.simplify(M_inv[row, 0] + M_inv[row, 2]))
    combinations[4] = P * p * (sp.simplify(M_inv[row, 0] + M_inv[row, 1]))
    combinations[5] = P * p * (sp.simplify(M_inv[row, 0] + M_inv[row, 2]))
    combinations[6] = sp.simplify(M_inv[row, 0]) + P * sp.simplify(M_inv[row, 1]) + P * sp.simplify(M_inv[row, 2]) +\
                      P ** 2 * sp.simplify(M_inv[row, 3])
    combinations[7] = P ** 2 * q * (sp.simplify(M_inv[row, 1] + M_inv[row, 3]))
    combinations[8] = P ** 2 * q * (sp.simplify(M_inv[row, 2] + M_inv[row, 3]))
    print('c[1] P**2*qbar * (w1+w2+w3+w4) = ', sp.simplify(combinations[1]))
    print('c[2] P**2*r * (w1+w2) =   ', sp.simplify(combinations[2]))
    print('c[3] P**2*r * (w1+w3) =   ', sp.simplify(combinations[3]))
    print('c[4] P*p * (w1+2) =   ', sp.simplify(combinations[4]))
    print('c[5] P*p * (w1+w3) =  ', sp.simplify(combinations[5]))
    print('c[6] w1+P*w2+P*w3+P**2*w4 =   ', sp.simplify(combinations[6]))
    print('c[7] P**2*q * (w2+w4) =   ', sp.simplify(combinations[7]))
    print('c[8] P**2*q * (w3+w4) =   ', sp.simplify(combinations[8]))

#beta = A_inv.dot(zy)
#sp.pretty_print(sp.simplify(sp.cancel(beta[3])).subs(a2, 999))
#for row in range(4):
 #   print('\n plim(Betahat)[%s] = ' % str(row))
  #  print(sp.simplify(beta[row]))