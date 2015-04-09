"""
:module:vpp_anyleg.py
:synopsis: general 2D VPP model (arbitrary leg function), implemented with
    dopri5 from Shai
:moduleauthor: Moritz Maus <mmaus@sport.tu-darmstadt.de>

:edited: 25.02.2013 MM
         27.02.2013 MM - bug hunting on step transition
"""

import libshai.integro as ode
import mutils.misc as mi

from pylab import dot, array, hstack, vstack
import numpy as np
import numpy.linalg as li
import sys
from copy import deepcopy


def dy(t, state, pars, output_forces=False):
    """
    equations of motion for single- and double stance

    :args:
        t (float): absolute simulation time
        state (array): system state: x, y, phi, vx, vy, vphi
        pars (dict): system parameters. required keys:
            legfun1(function: t, l, l_dot, legpars -> float): the leg force law
            legpars1(dict): the parameters for the leg force law
            legfun2(function: t, l, l_dot, legpars -> float): the leg force law
                for the 2nd leg. Can be (None) if not present
            legpars2(dict): the parameters for the leg force law
            J (float): inertia of trunk
            m (float): mass of trunk
            r_h (float): distance hip-CoM
            r_vpp1 (float): distance CoM-vpp, for leg 1
            a_vpp1 (float): angle between trunk and line CoM-VPP, for leg  1
            r_vpp2 (float): distance CoM-vpp, for leg 2
            a_vpp2 (float): angle between trunk and line CoM-VPP, for leg  2
            x_foot1 (float): x-position of foot 1, or None
            x_foot2 (float): x-position of foot 2, or None
            g (1x2 float): acceleration due to gravity (e.g. [0, -9.81])
        output_foces (bool): if True, output the forces and torques

    """
    P = mi.Struct(pars)
    x, y, phi, vx, vy, vphi = state
    hip = (x + np.sin(phi) * P.r_hip, y - np.cos(phi) * P.r_hip)
    vhip = (vx + np.cos(phi) * vphi * P.r_hip,
            vy + np.sin(phi) * vphi * P.r_hip)
    # calculate (axial) force for first leg, if present
    Fleg1 = 0
    if (P.x_foot1 is not None) and P.legfun1:
        l1 = np.sqrt((hip[0] - P.x_foot1)**2 + hip[1]**2)
        ldot1 = .5 / np.sqrt(l1) * (2. * (hip[0] - P.x_foot1 )* vhip[0] + 2. *
                hip[1] * vhip[1])
        Fleg1 = P.legfun1(t, l1, ldot1, P.legpars1)
    
    # calculate (axial) force for second leg, if present
    Fleg2 = 0
    if (P.x_foot2 is not None) and P.legfun2:
        l2 = np.sqrt((hip[0] - P.x_foot2)**2 + hip[1]**2)
        ldot2 = .5 / np.sqrt(l2) * (2. * (hip[0] - P.x_foot2 )* vhip[0] + 2. *
                hip[1] * vhip[1])
        Fleg2 = P.legfun2(t, l2, ldot2, P.legpars2)
    
    # calculate required torques at each hip (1 & 2)
    T1 = 0
    F1 = np.array([0, 0])
    if Fleg1:
        vpp1 = (x - P.r_vpp1 * np.sin(P.a_vpp1 + phi),
                y + P.r_vpp1 * np.cos(P.a_vpp1 + phi))
        dir_f1 = np.array([vpp1[0] - P.x_foot1, vpp1[1]]) 
        dir_f1 = dir_f1 / li.norm(dir_f1)
        F1 = Fleg1 * dir_f1
        dir_l1 = np.array([P.x_foot1 - hip[0], -hip[1]])
        dir_l1 = dir_l1 / li.norm(dir_l1)
        leg1 = l1 * dir_l1
        T1_T = leg1[0] * F1[1] - leg1[1] * F1[0] 
        # don't forget the torque created by Fxr_hip
        T1_F = (hip[0] - x) * F1[1] - (hip[1] - y) * F1[0]
        T1 = T1_T + T1_F

    T2 = 0
    F2 = np.array([0, 0])
    if Fleg2:
        # trunk leg angle
        vpp2 = (x - P.r_vpp2 * np.sin(P.a_vpp2 + phi),
                y + P.r_vpp2 * np.cos(P.a_vpp2 + phi))
        dir_f2 = np.array([vpp2[0] - P.x_foot2, vpp2[1]]) 
        dir_f2 = dir_f2 / li.norm(dir_f2)
        F2 = Fleg2 * dir_f2
        dir_l2 = np.array([P.x_foot2 - hip[0], -hip[1]])
        dir_l2 = dir_l2 / li.norm(dir_l2)
        leg2 = l2 * dir_l2
        T2_T = leg2[0] * F2[1] - leg2[1] * F2[0] 
        # don't forget the torque created by Fxr_hip
        T2_F = (hip[0] - x) * F2[1] - (hip[1] - y) * F2[0]
        T2 = T2_T + T2_F

    F = F1 + F2
    if not output_forces:
        return [vx, vy, vphi, F[0]/P.m + P.g[0], F[1]/P.m + P.g[1], (T1 + T2)/P.J] 
    else:
        return [F1, F2, T1, T2]


def to_event_spring(t, states, traj, pars):
    """
    triggers the takeoff of the 2nd leg.
    *Note* it is assumed that *always* the 2nd leg is the trailing leg. This
    has to be ensured by appropriate transitions.

    Condition is: rest length is reached.
    *Note* There could be further conditions: e.g., leg *must* be behind CoM,
    behind hip, ... whatever (useful for more detailled model investigations)
    """
    P = mi.Struct(pars)
    x1, y1, phi1, vx1, vy1, vphi1 = states[0]
    x2, y2, phi2, vx2, vy2, vphi2 = states[1]
    
    hip1 = (x1 + np.sin(phi1) * P.r_hip, y1 - np.cos(phi1) * P.r_hip)
    l1 = np.sqrt((hip1[0] - P.x_foot2)**2 + hip1[1]**2)
    hip2 = (x2 + np.sin(phi2) * P.r_hip, y2 - np.cos(phi2) * P.r_hip)
    l2 = np.sqrt((hip2[0] - P.x_foot2)**2 + hip2[1]**2)
    return l1 < P.legpars2['l0'] and l2 >= P.legpars2['l0']

def to_event_spring_refine(t, state, pars):
    """
    refinement function for SLIP 2nd leg takeoff event.

    This function returns some value whose zero-crossing indicates the event to
    be detected.
    """
    P = mi.Struct(pars)
    x, y, phi, vx, vy, vphi = state
    
    hip = (x + np.sin(phi) * P.r_hip, y - np.cos(phi) * P.r_hip)
    l = np.sqrt((hip[0] - P.x_foot2)**2 + hip[1]**2)
    return l - P.legpars2['l0']

def td_event_spring(t, states, traj, pars):
    """
    triggers the touchdown event for leg 2 (which will become leg 1 at
    touchdown)

    Condition is: the leg touches ground (here, a fixed angle w.r.t. world is
    assumed. To change this, edit (a) this trigger and (b) the appropriate
    touchdown transitions.
    """

    P = mi.Struct(pars)
    x1, y1, phi1, vx1, vy1, vphi1 = states[0]
    x2, y2, phi2, vx2, vy2, vphi2 = states[1]
    
    hip1 = (x1 + np.sin(phi1) * P.r_hip, y1 - np.cos(phi1) * P.r_hip)
    hip2 = (x2 + np.sin(phi2) * P.r_hip, y2 - np.cos(phi2) * P.r_hip)

    y_foot1 = hip1[1] - P.legpars2['l0'] * np.sin(P.legpars2['alpha'])
    y_foot2 = hip2[1] - P.legpars2['l0'] * np.sin(P.legpars2['alpha'])
    return y_foot1 > 0 and y_foot2 <= 0



def td_event_spring_refine(t, state, pars):
    """
    refinement function for SLIP 2nd leg touchdown event.

    This function returns some value whose zero-crossing indicates the event to
    be detected.
    """
    x, y, phi, vx, vy, vphi = state
    P = mi.Struct(pars)

    hip = (x + np.sin(phi) * P.r_hip, y - np.cos(phi) * P.r_hip)
    y_foot = hip[1] - np.sin(P.legpars2['alpha']) * P.legpars2['l0']
    return -y_foot

def VLO_event(t, states, traj, pars):
    """
    calcualtes the vertical leg orientation instant
    """
    P = mi.Struct(pars)
    x1, y1, phi1, vx1, vy1, vphi1 = states[0]
    x2, y2, phi2, vx2, vy2, vphi2 = states[1]
    
    hip1 = (x1 + np.sin(phi1) * P.r_hip, y1 - np.cos(phi1) * P.r_hip)
    hip2 = (x2 + np.sin(phi2) * P.r_hip, y2 - np.cos(phi2) * P.r_hip)

    return hip1[0] < P.x_foot1 and hip2[0] >= P.x_foot1

def VLO_event_refine(t, state, pars):
    """
    calcualtes the vertical leg orientation instant - refinement
    """
    P = mi.Struct(pars)
    x, y, phi, vx, vy, vphi = state
    hip = (x + np.sin(phi) * P.r_hip, y - np.cos(phi) * P.r_hip)

    return hip[0] - P.x_foot1

def spring_leg(t, l, l_dot, legpars):
    """
    The force of an elastic leg.

    :args:
        t (float): simulation time
        l (float): leg length
        l_dot (float): leg length change velocity
        legpars (dict): leg properties with:
            'k' (float): spring stiffness
            'l0' (float): spring rest length
    """
    return legpars['k'] * (legpars['l0'] - l)


def VPP_step(IC, pars, return_traj=True, with_forces=False):
    """
    performs a walking step of the VPP model

    :args:
        IC : the initial conditions (at VLO): x, y, vx
        pars (dict): model parameter: k, alpha, l0, m, 
        traj (bool): return trajectory (t,y) or only final state
        with_forces (bool): also, return the forces and torques

    :returns:
        t, y, (x_foot1, x_foot2, t_td, t_to): if return_traj == True,
        t, y, ( -"-), [Forces]: if additionally with_forces == True
        FS (1x3): the final apex state or None if unsuccessful
       
        ([Forces] = [F1x, F1y, F2x, F2y, tau1, tau2])
    """
    # Forces is a vector with [Fx
    Forces = []
    debug = False
    P = deepcopy(pars)
    # first phase: single support until touchdown of 2nd leg
    o = ode.odeDP5(dy, pars=P)
    o.ODE_RTOL = 1e-9
    o.ODE_ATOL = 1e-9
    o.event = td_event_spring
    t, y = o(IC, 0, 2)
    tt, yy = o.refine(lambda tf,yf: td_event_spring_refine (tf, yf, P))
    # compute forces?
    if with_forces:
        f_pt = [hstack(dy(ttt, yyy, P, output_forces=True))
                for ttt, yyy in zip(t, y)]
        Forces.append(vstack(f_pt))

    # transition: 1st leg -> 2nd leg; initialize new 1st leg (xfoot)
    t_td = tt

    tmpf, tmpp = deepcopy((P['legfun2'], P['legpars2']))
    P['legfun2'] = P['legfun1']
    P['legpars2'] = P['legpars1']
    P['legfun1'] = tmpf
    P['legpars1'] = tmpp
    P['x_foot2'] = P['x_foot1']
    hipx = yy[0] + P['r_hip'] * np.sin(yy[2])
    P['x_foot1'] = hipx + P['legpars1']['l0'] * np.cos(P['legpars1']['alpha'])

    # second phase: double support until takeoff of 1st leg
    o = ode.odeDP5(dy, pars=P)
    o.ODE_RTOL = 1e-9
    o.ODE_ATOL = 1e-9
    o.event = to_event_spring
    tnew, ynew = o(yy, tt, tt+2)
    tt, yy = o.refine(lambda tf,yf: to_event_spring_refine (tf, yf, P))
    t, y = np.hstack([t, tnew]), np.vstack([y, ynew])
    if with_forces:
        f_pt = [hstack(dy(ttt, yyy, P, output_forces=True))
                for ttt, yyy in zip(tnew, ynew)]
        # the last indexing switches leg 1 and 2
        Forces.append(vstack(f_pt)[:,[2,3,0,1,5,4]])
    t_to = tt

    # third phase: single support of leg1
    P['x_foot2'] = None
    o = ode.odeDP5(dy, pars=P)
    o.ODE_RTOL = 1e-9
    o.ODE_ATOL = 1e-9
    o.event = VLO_event

    tnew, ynew = o(yy, tt, tt + 2)
    tt, yy = o.refine(lambda tf,yf: VLO_event_refine (tf, yf, P))

    t, y = np.hstack([t, tnew]), np.vstack([y, ynew])
    if with_forces:
        f_pt = [hstack(dy(ttt, yyy, P, output_forces=True))
                for ttt, yyy in zip(tnew, ynew)]
        # the last indexing switches leg 1 and 2
        Forces.append(vstack(f_pt)[:,[2,3,0,1,5,4]])

    if return_traj:
        if with_forces:
            return (t, y, (P['x_foot1'], P['x_foot2'], t_td, t_to),
                    vstack(Forces) )
        return t, y, (P['x_foot1'], P['x_foot2'], t_td, t_to)
    return y[-1,:]


def VPP_steps(IC, pars, n=2, count_only=False):
    """
    performs up to n steps of the VPP model

    :args:
        IC (1x6 float): initial condition
        pars (dict): parameters for VPP model
        n (int): steps to perform at most
        count_only (bool): return only the number of successful steps

    :returns:
        t, y: the time and the state of the model, or
           n: the number of steps reached before falling down

    """

    localpars = deepcopy(pars)
    niter = 0
    t, y = [0], [IC]
    last_t0 = 0
    while niter < n:
        # on update: DO NOT FORGET TO UPDATE THE FEET LOCATIONS
        tt, yy, feet = VPP_step(IC, localpars, return_traj=True)
        localpars['x_foot1'] = feet[0]
        localpars['x_foot2'] = feet[1]
        # skip first element - this is just the IC
        y.append(yy[1:])
        t.append(tt[1:] + last_t0)
        last_t0 = t[-1][-1]
        niter += 1
        # check if touchdown is possible (foot position above ground):
        P = mi.Struct(pars)
        hip_y = yy[-1,1] - np.cos(yy[-1,2]) * P.r_hip
        if hip_y - np.sin(P.legpars2['alpha']) * P.legpars2['l0'] <= 0:
            print "no touchdown possible (n=", niter, ")"
            print "hip_y: ", hip_y
            break;
        IC = yy[-1,:]
    if not count_only:
        return np.hstack(t), np.vstack(y)
    else:
        return niter
    

def get_IC(Par):
    """
    sets the IC according to the Nature Communication paper, 
    given the parameter Par
    """
    x0 = 0
    y0 = Par['legpars1']['l0'] + Par['r_hip'] - np.sqrt(6./Par['legpars1']['k'])
    phi0 = 0
    vphi0 = .1
    Erem = 900. - 9.81 * y0 * Par['m'] - .5 * vphi0**2 * Par['J']
    vx0 = np.sqrt(2. * Erem / Par['m'])
    vy0 = 0
    return [x0, y0, phi0, vx0, vy0, vphi0]

def P0(k, alpha):
    """
    returns a special parameter set, where only the leg parameters k and alpha
    are open parameters
    """

    P0 = { 'legfun1': spring_leg,
        'legfun2': spring_leg,
        'legpars1': {'k': k,
                'l0': 1.,
                'alpha': alpha, },
        'legpars2': {'k': k,
                'l0': 1.,
                'alpha': alpha, },
        'J': 5.,
        'm': 80.,
        'r_hip': .1,
        'r_vpp1': .25,
        'a_vpp1': 0.,
        'r_vpp2': .25,
        'a_vpp2': 0.,
        'x_foot1': 0,
        'x_foot2': None,
        'g' : [0, -9.81],
        }
    return P0


if __name__ == '__main__':
    # provide an initial (testing) set of model parameters (SLIP leg function)
    _P0 = { 'legfun1': spring_leg,
        'legfun2': spring_leg,
        'legpars1': {'k': 13000.,
                'l0': 1.,
                'alpha': 69. * np.pi / 180., },
        'legpars2': {'k': 13000.,
                'l0': 1.,
                'alpha': 69. * np.pi / 180., },
        'J': 5.,
        'm': 80.,
        'r_hip': .1,
        'r_vpp1': .25,
        'a_vpp1': 0.,
        'r_vpp2': .25,
        'a_vpp2': 0.,
        'x_foot1': 0,
        'x_foot2': None,
        'g' : [0, -9.81],
        }

    _IC = [0, 1.078517, 0, 1.157, 0, .1]

    res = VPP_step(_IC, _P0)

