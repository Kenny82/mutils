"""
:module: ode5demo.py
:synopsis: demonstration how to use integro.py
:moduleauthor: Moritz Maus <mmaus@sport.tu-darmstadt.de>

"""

import libshai.integro as ode

from pylab import dot, array

def ydot(t,y,p):
    """
    a general example function to use with the ode5

    :args:
        t (float): simulation time
        y (array or float): state of the system
            here: 1x2 matrix
        p (tuple or float): system parameter(s)
            here: float

    :returns:
        y' (array of float): the derivative of the system state y at time t
    """

    sysmat = array([[0, 1.],
                    [-2., p]])
    return dot(sysmat, y)

def event_raw(energy):
    """
    This is a parametrized event function that returns a function E
    This function E has this signature:
        E(t,y,p) -> system energy
    """
    pass

def efun(t, y, traj, p):
    """
    This is an event function that maps the input (*to be identified!*) to
    bool.

    :args:
        t (tuple of floats): absolute time before and after step
        y (2xd): array, containing the "before-step" and "after-step" system state
        traj (?): the trajectory event. Not sure this is really passed through
    
    :returns:
        True if an event has been detected, False otherwise
    """
    
    #print pars
    return t[1] > 1.935
    

if __name__=='__main__':
    # TODO: implement event!
    # define integrator object
    o = ode.odeDP5(ydot, pars=-1.)
    # define a stop event
    o.event = efun
    # run test integration
    # just call the object: (y0, t0, t_end)
    t, y = o([0, 1],0,5)
    
    from pylab import figure, show, plot
    fig = figure(1)
    fig.clf()
    plot(t,y,'.-')
    show()
    



