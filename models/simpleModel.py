# -*- coding : utf-8 -*-
"""
.. module:: simpleModel
    :synopsis: Module that models and analyzes the 'simpleModel' for the
        BALANCE project

.. moduleauthor:: Moritz Maus <mmaus@sport.tu-darmstadt.de>


"""


import mutils.misc as mi
from pylab import array


def legODE1D(state, t, params):
    """
    This function defines the simpleModels axial leg as an ODE. It can be used
    to simulate the behavior.

    Parameters
    ----------

        state : *array* (1x3)
            State of the system: [x1, x2, x3] alias [l, \dot{l} and l_d]

        t : *float*
            the time. This is ignored since the system is time-independent.

        params : *dict*
            a dictionary

    Returns
    -------

        dx/dt : *array* (1x3)
            The derivative of the state as a function of the state: dx/dt = f(x) 


    """
    
    P = mi.Struct(params) # allow struct-style access to keys
    x1, x2, x3 = state

    x1_dot = x2
#    F_S = -1. * P.k * (x1 - x3 - P.ls0)
    x2_dot = -1. * P.k / P.m * (x1 - x3 - P.ls0) + P.g 
    x3_dot = -1. / P.d * (-1. * P.k * (x1 - x3 - P.ls0) + P.kd * (x3 - P.ld0))

    return array([x1_dot, x2_dot, x3_dot])


P0 = {'m' : 80,
        'k' : 10000.,
        'kd' : 10000,
        'g' : -9.81,
        'ls0' : .5,
        'ld0' : .5,
        'd' : 10.,
        }


def remTerm(d, Params):
    """
    test function to visualize if there are any roots possible

    """
    
    P = mi.Struct(Params)
    
    kd = P.k / 3.

    res = (((P.k/(3.*P.m) - (P.k + kd)**2/(9.*d**2))**3 + (P.k*kd/(d*P.m) -
        P.k*(P.k + kd)/(3.*d*P.m) + 2.*(P.k + kd)**3/(27.*d**3))**2/4.)**(1./2.) +
        P.k*kd/(2.*d*P.m) - P.k*(P.k + kd)/(6.*d*P.m) + (P.k +
            kd)**3/(27.*d**3))**(1./3.) - (P.k + kd)/(3.*d)

    return res

def roi(d, Params):
    """
    a root for this expression has to be found!

    THIS IS A SIMPLIFICATION OF THE ABOVE EXPRESSION WITH k = 3*kd

    """
    
    P = mi.Struct(Params)


    res =  (((P.kd/P.m - 16*P.kd**2/(9*d**2))**3 + (-P.kd**2/(d*P.m) +
     128*P.kd**3/(27*d**3))**2/4.)**(1/2) - P.kd**2/(2*d*P.m) +
     64*P.kd**3/(27.*d**3))**(1./3.)
    
    return res


def denom(d, kd=10000., m = 50.):
    """
    no docstring
    """

    return ((kd/m- 16./9.*kd**2/d**2)**3 + .25*(-kd**2/(d*m) +
            128./27.*kd**3/d**3)**2)



def config_phaseC(r, ld, P):
        """
        This function returns the configuration of the simplemodel's leg in the
        plane, given the mass point's position relative to the toe.

        Parameters:
        -----------

        r : *array* (1x2)
            The position of the mass, relative to the foot point

        ld : *float*
            (Current) length of the damping unit

        P : *dict*
            A dictionary containing the model parameters *k* (leg stiffness),
            *c* (rotational stiffness), *phi0* (rest angle of ankle joint),
            *ls0* (rest length of the leg spring), *rtoe* (length of the foot
            segment)


        Returns:
        --------

        config : *list*
            The leg configuration, as follows: [Fx, Fy, Tau, phi, rToeX, rToeY,
            lSpring]

        """

        # use a root finding method to determine the model's state
        # -> formulate equations as root finding problem

        # first: write down the equations. See if sympy finds an analytic
        # solution :)

        pass

