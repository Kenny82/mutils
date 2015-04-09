"""
:module: SLIP_dopri_test.py
:synopsis: SLIP, computed in parallel, implemented with dopri5 from Shai
:moduleauthor: Moritz Maus <mmaus@sport.tu-darmstadt.de>

"""


from pylab import dot, array
import numpy as np
import sys

import time
from IPython.parallel import Client
startup = False

try:
    rc = Client()
    if len(rc.ids) > 0:
        startup = True
except Exception:
    # do not throw an exception, just exit
    pass

if not startup:
    print "Error - IPython parallel import failed!"
    print "Did you launch an ipcluster? (e.g.: $>ipcluster start --n 4)"
    sys.exit(1)

import libshai.integro as ode
from models.SLIP_dopri import *

lview = rc.load_balanced_view()


def get_n_steps_a(alpha):
    """ returns how many steps can be achieved 
    this function relies on that k, P and IC are present in the global (or at
    least parent) namespace
    """
    nstep = 0
    step_success = True
    IC = IC0
    pars = (k, alpha, P[0], P[1])
    n=50
    while nstep < n and step_success:
        #print "IC", IC
        #print "pars", pars
        IC = SLIP_step(IC, pars)
        if IC is None:
            step_success = False
        else:
            nstep += 1
    return nstep



def get_line(ck):
    return [get_n_steps_ak(ck, alpha) for alpha in alphas]


if __name__ == '__main__':
    # parameter scheme: k, alpha, l0, m, xfoot
    print "calculating J-shape"
    kk = np.linspace(5000, 30000, 15)
    aa = np.linspace(60, 75, 20) * np.pi / 180.
    l0 = 1.
    m = 80.
    # to create a directview dv:
    # dv = rc[:]
    rc[:].run('models/SLIP_dopri.py') # make functionality available
    # alternatively: "rc[:].execute('import ...')
    # for imports:
    # with rc[:].sync_imports():
    #    import numpy
    #    import pylab   # NOTE: import pylab as p DOES NOT WORK!
    # note: push and pull (dict-access to namespace)
    # rc[:].push({'var_a' : 23, 'b' : 39})
    # rc[:].get('var_a')
    t0 = time.time()
    print "imported!"
    P = (1., 80.) # leg rest length, mass
    IC = [0, 1, 5]
    rc[:].push({'P' : P, 'IC0' : IC, 'alphas' : aa}, block=True)
    J0 = lview.map(get_line, kk, block=True)
    tE = time.time() - t0
    print "Time elapsed:", tE
    print "calculating in serial:"
    t0 = time.time()
    J = []
    if True:
        for k in kk: 
            rc[:].push({'k' : k})
            print "calculating for k = ", k
            J.append( lview.map(get_n_steps_a, aa, block=True))
    from pylab import vstack
    J = vstack(J)
    tE = time.time() - t0
    print "Time elapsed:", tE
    print "calculating in serial:"
    t0 = time.time()
    J2 = calc_J(kk, aa, P, IC)
    tE = time.time() - t0
    print "Time elapsed:", tE
    assert np.allclose(J, J2), "differences in serial and parallel computation"

    import pylab as pl
    fig = pl.figure(1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    c = ax.pcolor(aa * 180 / np.pi, kk / 1000., J[:-1,:-1])
    c.set_clim(0,50)
    pl.colorbar(c)
    ax.set_xlabel(r'angle of attack $\alpha_0$ [deg]')
    ax.set_ylabel('stiffness k [kN m$^{-1}$]')
    ax.set_title('steps to fall for SLIP')
    pl.draw()
    fig.show()


    
