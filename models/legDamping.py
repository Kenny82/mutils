# -*- coding : utf8 -*-
"""

This module makes symbolic computations for the damping of the simpleModel's
leg.

"""

if __name__ != '__main__':
    raise NotImplementedError, ('THIS MODULE ONLY CONTAINS SOME SYMBOLIC' + 
        'CALCULATIONS!')

import sympy

x10, x20, x30, X1, X2, X3, k, m, k_d, d, ls0, ld0, g = sympy.symbols(
'x10, x20, x30, X1, X2, X3, k, m, k_d, d, ls0, ld0, g')

#define equations of motion in Laplace space:

# k = 4 * k_d

eq1 = sympy.Eq(s*X1 - x10, X2)
eq2 = sympy.Eq(s*X2 - x20, -k / m * (X1- X3 - ls0) + g)
eq3 = sympy.Eq(s*X3 - x30, -1 / d * (-1*k*(X1 - X3 - ls0) + k_d * (X3 - ld0)))

if not 'FLAG_eq_computed' in locals():
    pass

if True:
    eq_motion = sympy.solve([eq1, eq2, eq3], [X1, X2, X3])
    FLAG_eq_computed = True

# equations can now be accessed by: eq_motion[X1], eq_motion[X2] and so on

#den should be the (common) denominator of the system

den = 1 / eq_motion[X1].as_ordered_factors()[0]

roots = sympy.solve(den, s)

# pick some complex root
r1 = roots[1]

figure(1)
clf()

Pk = 10000.
Pm = 80.
Pkd = Pk / 8.
Pd = sqrt(Pk * Pm) / 1.5

s_k = 1
s_m = 1

for kd in [1219.9, 1239.975, 1250]:
    print 'calculating for kd = ', kd
    dvals = linspace(560.6, 581.3, 1000)
    res = vstack([complex(r1.evalf(subs={k : 10000 * s_k, k_d : kd * s_k, 
        d : x * sqrt(s_m * s_k), m : 80. * s_m})) for x in dvals])
    ares = angle(res)

    plot(dvals, ares, '.-', label=('kd = %.3f' % kd))

legend()
show()
