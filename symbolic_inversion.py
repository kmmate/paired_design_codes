"""
Compute symbolic inverse of a 4x4 matrix for MA thesis
"""

import numpy as np
from numpy import linalg
import sympy as sp
sp.init_printing(use_unicode=True)

"""
create symbolic expressions
"""
# P(Zj=z_a, Z-j=z-j)
Pz = {(i, j): sp.symbols('Pz' + str(i) + str(j)) for i in range(2) for j in range(2)}
# P(Dj(Zj=z_a,Z-j=z-j)=dj, D-j(Zj=z_a, Z-j=z-j)=d-j)
Pd = {((i1, i2, i3), (j1, j2, j3)): sp.symbols('Pd_(%s%s)%s_(%s%s)%s' % (i1, i2, i3, j1, j2, j3)) for i1 in range(2)
      for i2 in range(2) for i3 in range(2) for j1 in range(2) for j2 in range(2) for j3 in range(2)}
# E[Yj(Dj=dj, D-j=d-j) |Dj(Zj=z_a,Z-j=z-j)=dj, D-j(Zj=z_a, Z-j=z-j)=d-j]
Y = {((k1, k2), (i1, i2, i3), (j1, j2, j3)): sp.symbols('Y%s%s|(%s%s)%s_(%s%s)%s' % (k1, k2, i1, i2, i3, j1, j2, j3))
     for k1 in range(2) for k2 in range(2) for i1 in range(2) for i2 in range(2) for i3 in range(2)
     for j1 in range(2) for j2 in range(2) for j3 in range(2)}
# P(D-j(Zj=z_a, Z-j=z-j)=1)
Pmd1 = {(i, j): sp.symbols('Pmd_(%s%s)1' % (i, j)) for i in range(2) for j in range(2)}
# P(Z-j=1)
Pmz1 = sp.symbols('Pmz1')

"""
 put them into the matrix
"""
a1 = 1
a2 = sum([Pd[((z1, z2, 1), (z1, z2, 1))] * Pz[(z1, z2)] for z1 in range(2) for z2 in range(2)])
a3 = sum([Pd[((z1, z2, 1), (z1, z2, 0))] * Pz[(z1, z2)] for z1 in range(2) for z2 in range(2)])
a4 = sum([Pmd1[(z1, z2)] * Pz[(z1, z2)] for z1 in range(2) for z2 in range(2)])
b1 = Pz[(1, 1)]
b2 = Pd[((1, 1, 1), (1, 1, 1))] * Pz[(1, 1)]
b3 = Pd[((1, 1, 1), (1, 1, 0))] * Pz[(1, 1)]
b4 = Pmd1[(1, 1)] * Pz[(1, 1)]
c1 = Pz[(1, 0)]
c2 = Pd[((1, 0, 1), (1, 0, 1))] * Pz[(1, 0)]
c3 = Pd[((1, 0, 1), (1, 0, 0))] * Pz[(1, 0)]
c4 = Pmd1[(1, 0)] * Pz[(1, 1)]
d1 = Pmz1
d2 = Pd[((0, 1, 1), (0, 1, 1))] * Pz[(0, 1)] + Pd[((1, 1, 1), (1, 1, 1))] * Pz[(1, 1)]
d3 = Pd[((0, 1, 1), (0, 1, 0))] * Pz[(0, 1)] + Pd[((1, 1, 1), (1, 1, 0))] * Pz[(1, 1)]
d4 = Pmd1[(0, 1)] * Pz[(0, 1)] + Pmd1[(1, 1)] * Pz[(1, 1)]

"""
put them into the vector
"""
e1 = sum([Pz[(z1, z2)] * sum([Y[((dj, dmj), (z1, z2, dj), (z1, z2, dmj))] * Pd[((z1, z2, dj), (z1, z2, dmj))]
                             for dj in range(2) for dmj in range(2)]) for z1 in range(2) for z2 in range(2)])
e2 = Pz[(1, 1)] * sum([Y[((dj, dmj), (1, 1, dj), (1, 1, dmj))] * Pd[((1, 1, dj), (1, 1, dmj))]
                             for dj in range(2) for dmj in range(2)])
e3 = Pz[(1, 0)] * sum([Y[((dj, dmj), (1, 0, dj), (1, 0, dmj))] * Pd[((1, 0, dj), (1, 0, dmj))]
                             for dj in range(2) for dmj in range(2)])
e4 = Pz[(0, 1)] * sum([Y[((dj, dmj), (0, 1, dj), (0, 1, dmj))] * Pd[((0, 1, dj), (0, 1, dmj))]
                             for dj in range(2) for dmj in range(2)]) + e2

"""
compute determinant and inverse
"""
A = sp.Matrix([[a1, a2, a3, a4], [b1, b2, b3, b4], [c1, c2, c3, c4], [d1, d2, d3, d4]])
e = sp.Matrix([[e1], [e2], [e3], [e4]])
# checking invertibility
A_det = A.det()
print('The determinant of M is\n', sp.factor(A_det))
print('\n -------- Inverse -----------\n')
A_inv = A.inv()
print(sp.simplify(A_inv[3, 3]))
STOP
beta = A_inv.dot(e)
#sp.pretty_print(sp.simplify(sp.cancel(beta[3])).subs(a2, 999))
print(sp.simplify(beta[3]))