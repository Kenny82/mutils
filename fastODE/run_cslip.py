
import fastode
import time

alpha = 68.5 * pi / 180

fl = fastode.FastODE('mmslipflight')
st = fastode.FastODE('mmslipstance')

yout_fl = zeros((1000, fl.WIDTH), dtype=float64)
yout_st = zeros((1000, st.WIDTH), dtype=float64)

IC0 = [0, 0, 1, 5., 0] # t0, x0, y0, vx0, vy0
N_st = 0
yout_st[N_st,: ] = IC0

all_dat = []

t0 = time.time()
for rep in range(1000):
    yout_fl[0,:] = yout_st[N_st, :]
    N_fl = fl.odeOnce(yout_fl, yout_fl[0,0] + 5, pars=(sin(alpha),)) #
    all_dat.append(yout_fl[:N_fl + 1, :].copy())

    yout_st[0,:] = yout_fl[N_fl, :]
    N_st = st.odeOnce(yout_st, yout_st[0,0] + 5, pars=(yout_st[0,1] + cos(alpha),)) #
    all_dat.append(yout_st[:N_st + 1, :].copy())

tE = time.time()
print "elaped time [s]:", tE - t0

all_dat = vstack(all_dat)
figure()
plot(all_dat[:,1], all_dat[:,2], 'r.-')


