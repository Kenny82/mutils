"""
:module: sliptest.py
:synopsis: SLIP, implemented with dopri5 from Shai
:moduleauthor: Moritz Maus <mmaus@sport.tu-darmstadt.de>

"""

import libshai.integro as ode

from pylab import dot, array
import numpy as np
import sys


def dy_stance(t, state, p):
    """
    equations of motion in stance for SLIP.

    :args:
        t (float): simulation time
        state (array): system state [x, x', y, y']
        p (tuple): system parameters: k, alpha, l0, m, xfoot

    :returns:
        dy (array): derivative of y (system state)

    """
    x, vx, y, vy = state
    k, alpha, l0, m, xfoot = p
    l = np.sqrt((x - xfoot)**2 + y**2)
    F = k * (l0 - l)
    Fx = F * (x - xfoot) / l
    Fy = F * y / l
    return array([vx, Fx/m, vy, Fy/m - 9.81]) 

def to_event(t, states, traj, p):
    """
    triggers the takeoff
    """
    x1, vx1, y1, vy1 = states[0]
    x2, vx2, y2, vy2 = states[1]
    k, alpha, l0, m, xfoot = p
    l1 = np.sqrt((x1 - xfoot)**2 + y1**2)
    l2 = np.sqrt((x2 - xfoot)**2 + y2**2)
    return l1 < l0 and l2 >= l0

def to_event_refine(t, state, p):
    """
    same event, just with a different interface. Used for the "refine"
    method.

    This function returns some value whose zero-crossing indicates the event to
    be detected.
    """
    x1, vx1, y1, vy1 = state
    k, alpha, l0, m, xfoot = p
    l1 = np.sqrt((x1 - xfoot)**2 + y1**2)
    return l1 - l0


def SLIP_step(IC, pars):
    """
    performs a SLIP step

    :args:
        IC (1x3): the SLIP initial conditions (at apex): x, y, vx
        pars (tuple): SLIP parameter: k, alpha, l0, m, 

    :returns:
        FS (1x3): the final apex state or None if unsuccessful
    """
    # first: calculate touchdown point
    k, alpha, l0, m = pars
    y_TD = l0 * np.sin(alpha)
    y_fall = IC[1] - y_TD
    if y_fall < 0:
        return None
    t_TD = np.sqrt(2. * y_fall / 9.81)
    x_TD = IC[0] + t_TD * IC[2]
    x_foot = x_TD + l0 * np.cos(alpha)
    vy_TD = - t_TD * 9.81
    p = list(pars) + [x_foot]
    # calculate stance:
    o = ode.odeDP5(dy_stance, pars=p)
    o.ODE_RTOL = 1e-6
    o.ODE_ATOL = 1e-6
    o.event = to_event
    t, y = o([x_TD, IC[2], y_TD, vy_TD], 0, 2)
    tt, yy = o.refine(lambda tf,yf: to_event_refine(tf, yf, p))
    # calcualte next apex
    if yy[3] < 0:
        # liftoff velocity negative. NOTE: technically, another step could
        # occur, but here we look only for steps with apex during flight
        return None
    t_Apex = yy[3] / 9.81
    y_Apex = yy[2] + .5 * 9.81 * t_Apex**2 
    x_Apex = yy[0] + yy[1] * t_Apex 
    vx_Apex = yy[1]
    return array([x_Apex, y_Apex, vx_Apex])


def get_n_steps(k, alpha, P, IC0, n=50):
    """ returns how many steps can be achieved """
    nstep = 0
    step_success = True
    IC = IC0
    pars = (k, alpha, P[0], P[1])
    while nstep < n and step_success:
        #print "IC", IC
        #print "pars", pars
        IC = SLIP_step(IC, pars)
        if IC is None:
            step_success = False
        else:
            nstep += 1
    return nstep

def calc_J(ks, alphas, P, IC, n=50):
    """
    returns the J-shape(?)
    :args:
        ks (array): stiffnesses to test
        alphas (array): angle of attack to test (in radiant)
        P: remaining parameters: l0, m
        IC: initial conditions [x, y, vx]
    """

    res = np.zeros((len(ks), len(alphas)))
    for nr_k, k in enumerate(ks):
        print '\nk =', k
        for nr_a, alpha in enumerate(alphas):
            res[nr_k, nr_a] = get_n_steps(k, alpha, P, IC)
    return res

def get_n_steps_ak(k, alpha):
    """ returns how many steps can be achieved 
    this function relies on that k, P and IC are present in the global (or at
    least parent) namespace

    *note* this is a test function for parallel computing
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


