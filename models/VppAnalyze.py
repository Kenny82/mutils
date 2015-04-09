"""
:module: VppAnalyze.py
:synopsis: This script performs some basic test for the VPP model.
:moduleauthor: Moritz Maus <mmaus@sport.tu-darmstadt.de>

Several utilities are provided:
    * Model parameter computation to match a desired final state in a final
      time

"""

import models.VPP_anyleg as vpp
from pylab import figure
from copy import deepcopy
import numpy as np
import scipy.optimize as opt


class DimensionError(Exception):
    """  This Exception is raised when pfun gets an invalid number
    """
    def __init__(self, value=''):
        """
        inits the DimensionError
        """
        self.value = value
        self.message = value
    def __str__(self):
        return repr(self.value)

def pfun_SLIP_1(P_orig, dim, mag):
    """ utility function (part of model sensitivity calculation)
    varies the parameter structure P (for SLIP-VPP), the nth element by mag.
    
    :args:
        P (dict): Parameter dictionary for VPP_step
        dim (int): The number of the parameter to vary. Here:
            0: k (leg 1)
            1: k (leg 2)
            2: l0 (leg 1)
            3: l0 (leg 2)
            4: alpha (leg 2)
            5: r_vpp1
            6: r_vpp2
            7: a_vpp1
            8: a_vpp2
        mag: vary by this value
    """
    P = deepcopy(P_orig)
    known_dims = range(9)
    if not dim in known_dims:
        raise DimensionError, "only 9 dimensions (0-8) present"
    if dim == 0:   
        # arbitrary scaling factors just mean the choice of another unit. Here,
        # scalings are used for better numerical accuracy
        # *NOTE*: numerical accuracy is assessed by the magnitude of the values
        # of the Jacobian (dX_dP)!
        P['legpars1']['k'] += mag * 10000.
    elif dim == 1:
        P['legpars2']['k'] += mag * 10000.
    elif dim == 2:
        P['legpars1']['l0'] += mag * .1
    elif dim == 3:
        P['legpars2']['l0'] += mag * .1
    elif dim == 4:
        P['legpars2']['alpha'] += mag * .25
    elif dim == 5:
        P['r_vpp1'] += mag 
    elif dim == 6:
        P['r_vpp2'] += mag
    elif dim == 7:
        P['a_vpp1'] += mag
    elif dim == 8:
        P['a_vpp2'] += mag
   
    return P

def goalfun_SLIP_1(IC, P):
    """ utilify function (part of parameter sensitivity calculation)
    returns critical points of a given model motion for a specific set of
    initial conditions and parameters.
    
    :args:
        IC (6x float): initial conditions of VPP model
        P (dict): parameter for VPP model
    
    :returns:
        (nx float): a set of critical values that quantify the model step.
        here: final state, step duration, touchdown time, takeoff time, horiz.
        foot position
    
    """
    t, y, foot_info = vpp.VPP_step(IC, P)
    # TODO: modify VPP_step such that additional information is returned:
    x_foot1, x_foot2, t_td, t_to = foot_info[:4]
    # scale values such that expected variations are in the same order of magnitude
    fs_scaling = np.array([1., 3., .5, .5, .5, .2, 3., 1., 1., 1.])
    
    return np.hstack([y[-1, :], t[-1], t_td, t_to, x_foot1 ]) * fs_scaling
    

def dX_dP(IC, P, pfun, goalfun):
    """ calculates the model sensitivity 
    calculates the derivative of the goalfun with respect to the model
    parameters. The idea is to find square submatrices of full rank, which can
    then be used to define a mapping "P->state", i.e. a model control scheme.
    
    :args:
        IC (6x float): initial conditions
        P (dict): (base) parameter set
        pfun (function): the parameter variation function.
            callSignature:
                pfun(P, int dim, int mag) -> P' or DimensionError
        goalfun (function): VPP_traj -> (n float) (should include final
        state)
    
    """
    J = []
    dim = 0
    try:
        while True:
            h = .0001
            P_curr = pfun(P, dim, h) 
            Xp = goalfun(IC, P_curr)
            P_curr = pfun(P, dim, -h)
            Xm = goalfun(IC, P_curr)
            col_n = (Xp - Xm) / (2. * h)
            J.append(col_n)
            dim += 1
    except DimensionError:
        J = np.vstack(J).T
    
    return J

def dk_dL(L0, k, L, dE):
    """ how to change a spring energy without changing its force
    computes the required stiffness change and rest length change
    to inject energy without changing the spring force
    
    :args:
        L0 (float): spring rest length
        k (float): spring stiffness
        L (float): current spring length
        dE (float): amount of energy to inject
    
    :returns:
        dk (float), dL (float): required changes of leg stiffness and rest
            length
    """
    dL = 2. * dE / ( k *(L0 - L))
    dk = k * ((L - L0) / (L - (L0 + dL)) - 1.)
    return dk, dL

def vary_P_SLIP(P_orig, IC, deltaP):
    """ utility function for fitting SLIP to (exp. ) data
    varies the parameter structure P (for SLIP-VPP) by deltaP
    
    :args:
        P0 (dict): Parameter dictionary for VPP_step
        IC (1x6 float): initial conditions (required to adapt leg spring energy)
        deltaP (1x6 float): parameter to vary. Dimensions are:
            0: dE_leg1
            1: k (leg 2)
            2: l0 (leg 2)
            3: alpha (leg 2)
            4: r_vpp (1+2)
            (optional) 5: a_vpp (1+2)
    :returns: P (dict) new model parmeter
    
    *NOTE*:
        for better numerical accuracy, there are scalings involved here, i.e.
            the values supplied by deltaP are scaled before they are added to
            any dimension!
    """
    P = deepcopy(P_orig)
    # stored energy in leg spring:
    hip_y = IC[1] - np.cos(IC[2]) * P['r_hip']
    dk, dL = dk_dL(P['legpars1']['l0'], P['legpars1']['k'], hip_y, deltaP[0])
    
    P['legpars1']['k'] += dk
    P['legpars1']['l0'] += dL
    P['legpars2']['k'] += deltaP[1] * 1000.
    P['legpars2']['l0'] += deltaP[2] * .1
    P['legpars2']['alpha'] += deltaP[3]
    P['a_vpp1'] += deltaP[4]
    P['a_vpp2'] += deltaP[5]
    #   P['r_vpp1'] += deltaP[5] * .5
    #    P['r_vpp2'] += deltaP[5] * .5
    
    return P

def fit_VPP_SLIP(IC, FS, T, P0):
    """ calculates the required model parameters to match a desied final state
    returns parameters that map the initial condition IC to the final state FS.
    :args:
        IC (1x6 float): the initial state of the model
        FS (1x6 float): the final state of the model
            (note: the first coordinate will be ignored!
        P0 (dict): initial guess for the parameters
    
    :returns:
        P (dict) set of parameters that map IC to FS

    *NOTE* In order to continue the force from the last step, the first guess
    (parameter P0) must have the same leg rest length and stiffness as in the
    last step.
    """
    # difference function: VPP_step(IC, P)
    def delta_fun(delta_p):
        """ function to be 0. uses a closure """
        res = vpp.VPP_step(IC, vary_P_SLIP(P0, IC, delta_p))
        return np.hstack([res[1][-1,1:] - FS[1:], res[0][-1] - T])
    res, idict, state, msg = opt.fsolve(delta_fun, x0 = [0,0,0,0,0,0], factor=.2,
            full_output=True)
    if state == 1:
        # success!
        Pn = vary_P_SLIP(P0, IC, res)
        return Pn
    else:
        print "No solution found :( reason:", msg
        return None



