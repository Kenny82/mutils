# some definitions:
# foot1 is the leading leg (always on ground)
# foot1 is *NOT* the tip of leg1 - it's the tip of the leading leg
# the leading leg is: the stance leg (single stance), the leg with the most
# recent touchdown (double stance)
# P_lleg indicates if leg1 or leg2 is the leading leg

from pylab import *
import fastode
from copy import deepcopy

s = fastode.FastODE('bslipss')
d = fastode.FastODE('bslipds')

all_res = []

res = zeros((1000, s.WIDTH), dtype=float64)
res[0, :] = [0, 0, .99, 0, 1.1, 0, .01]

#define P_k1 0
#define P_k2 1
#define P_a1 2
#define P_a2 3
#define P_l01 4
#define P_l02 5
#define P_b1 6
#define P_b2 7
#define P_m 8
#define P_g 9
#define P_f1x 10
#define P_f1y 11
#define P_f1z 12
#define P_f2x 13
#define P_f2y 14
#define P_f2z 15
#define P_lleg 16

p0 = [20000., 20000., 76.5 * pi / 180., 76.5 * pi / 180., 1., 1., .05, -.05,
        80., -9.81, 0, 0, -.05, -10., 0., 0., 1.]

p_B = [ 12850, 15150,  1.26623637,   1.26449104, 1., 1., .1, -.1, 80, -9.81, 0,
        0, 0, -2, 0, 0, 1.]
IC_B = [-0.097636104761338383, 0.92667180991112486, 0, 1.1014930243212504, 0.36541309273918576, 0.35327901338274703]

pars = deepcopy(p_B)
res[0, 1:] = IC_B

N = 0

import time
t0 = time.time()

#print len(pars)
for rep in range(20):
    res[0,:] = res[N,:].copy()
    N = s.odeOnce(res, res[N,0] + 1., dt=.05, pars=pars)
    all_res.append(res[:N+1, :].copy())

    y = res[N, 1:] # shortcut
    vx, vz = y[3], y[5]
    a_v_com = -arctan2(vz, vx) # correct with our coordinate system

# touchdown detected:
# (1) foot2 = foot1
# (2) foot1 = [NEW]
# (3) leading_leg = ~leading_leg
# update leg positions; change trailing leg

    pars[13] = pars[10]
    pars[15] = pars[12]
    if pars[16] == 1.:
# stance leg is leg 1 -> update leg 2 params
        pars[10] = y[0] + cos(pars[3]) * cos(pars[7] + a_v_com) * pars[5]
        pars[12] = y[2] - cos(pars[3]) * sin(pars[7] + a_v_com) * pars[5]

        #pars[13] = res[N, 1] + cos(pars[3])*cos(pars[7])*pars[5]
        #pars[15] = res[N, 3] + cos(pars[3])*sin(pars[7])*pars[5]
        pars[16] = 2.;
    else:
        pars[10] = y[0] + cos(pars[2]) * cos(pars[6] + a_v_com) * pars[4]
        pars[12] = y[2] - cos(pars[2]) * sin(pars[6] + a_v_com) * pars[4]

        #pars[10] = res[N, 1] + cos(pars[2])*cos(pars[6])*pars[4]
        #pars[12] = res[N, 3] + cos(pars[2])*sin(pars[6])*pars[4]
        pars[16] = 1.;

    print "ss-> ds: com:", y[:3]
    print "ss-> ds: foot1:", pars[10:13]
    print "ss-> ds: foot2:", pars[13:16]

    res[0,:] = res[N,:].copy()
    N = d.odeOnce(res, res[N,0] + 1, dt=.05, pars=pars)
    all_res.append(res[:N+1, :].copy())

# update leg positions; change trailing leg
    # update foot1 position!
    y = res[N, 1:] # shortcut

    vx, vz = y[3], y[5]
    a_v_com = -arctan2(vz, vx) # correct with our coordinate system

# takeoff detected:
# no parameter adaptation required

tE = time.time()
print "elapsed time for simulation:", tE - t0


ares = vstack(all_res)
figure()
subplot(2,1,1)
plot(ares[:,0], ares[:,2], 'r.-');
plot([0,5], [sin(pars[3]), ] * 2, 'k--')
plot([0,5], [sin(pars[2]), ] * 2, 'k--')
for elem in all_res:
    plot([elem[-1,0], ] * 2, [0, 1], 'k')
subplot(2,1,2)
plot(ares[:,0], ares[:,3], 'b.-');



show()

