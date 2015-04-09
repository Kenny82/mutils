"""
:module: IP_dopri.py
:synopsis: inverted pendulum, implemented with dopri5 from Shai
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
        state (array): system state [phi, phi']
        p (tuple): system parameters: alpha, l0, gamma

    :returns:
        dy (array): derivative of y (system state)

    """

    phi, vphi = state
    alpha, l0, gamma = p
    return array([vphi, -9.81 / l0 * np.cos(phi)])

def td_event(t, states, traj, p):
    """
    triggers the touchdown (either other leg or fall (backwards)
    """
    phi1, vphi1 = states[0]
    phi2, vphi2 = states[1]
    alpha, l0, gamma = p
    y_f1 = l0 * np.sin(phi1 + gamma) - l0 * sin(alpha + phi1 + gamma)
    y_f2 = l0 * np.sin(phi2 + gamma) - l0 * sin(alpha + phi2 + gamma)
    foot_td = y_f1 > 0 and y_f2 < 0
    fall = (phi2 > np.pi and phi1 <= np.pi) or (phi2 <= 0 and phi1 > 0)
    return foot_td or fall

def td_event_refine(t, state, p):
    """
    same event, just with a different interface. Used for the "refine"
    method.

    This function returns some value whose zero-crossing indicates the event to
    be detected.
    """
    phi1, vphi1 = state
    alpha, l0, gamma = p
    y_f1 = l0 * np.sin(phi1 + gamma) - l0 * sin(alpha + phi1 + gamma)
    return y_f1

def IP_step(IC, pars, t0 = 0):
    """
    performs a SLIP step

    :args:
        IC (1x2): [phi, phi_dot]
        pars (tuple): SLIP parameter: alpha, l0, gamma, 

    :returns:
        t, y, FS: step time, step state, and final state (after impact in new
        coordinate system)
    """
    # calculate stance:
    o = ode.odeDP5(dy_stance, pars=pars)
    o.ODE_RTOL = 1e-6
    o.ODE_ATOL = 1e-6
    o.event = td_event
    t, y = o(IC, t0, t0 + 3)
    tt, yy = o.refine(lambda tf,yf: td_event_refine(tf, yf, pars))
    # calculate impact! -> angular momentum (="tangential velocity") is
    # conserved. This "tangential velocity" is the cosine of the between-leg
    # angle alpha
    vphi_plus = np.cos(pars[0]) * yy[1]
    phi_plus = yy[0] + pars[0]
    return (t,y,array([phi_plus, vphi_plus]))

if __name__ == '__main__':
    IC = [np.pi / 2, -1.] # leg angle, leg rotational velocity
    pars = [.4, 1., .13]  # system parameters: inter-leg angle, leg length, slope
    all_y = []
    all_t = []
    t = [0]
    for n in range(40):
        t, y, IC = IP_step(IC, pars, t[-1])
        all_t.append(t)
        all_y.append(y)
    y = np.vstack(all_y)
    t = np.hstack(all_t)
    from pylab import figure, plot, show
    fig = figure(2)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, y[:, 1], 'r.-', label='angular velocity')
    ax.legend()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('leg angular velocity [rad / s]')
    ax.set_title('inverted pendulum walk')
    show()



    

