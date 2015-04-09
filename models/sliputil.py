# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:31:32 2012

This file includes a collection of utility functions for use in combination
with the SLIP simulation

@author: moritz
"""

from pylab import (mean, hstack, vstack, squeeze, newaxis, array, dot,
                   zeros_like, pinv)

import fitSlip as fl
import slip as sl
import copy
import scipy.optimize as opt

# another self-written, yet required lib:
import mutils.FDatAn as fda
import mutils.io as mio 

def getPeriodicOrbit2(ICr, Tr, yminr, ICl, Tl, yminl, m, startParams=[14000.,
    1.16, 1., 0., 0.]):
    """
    Returns parameters that yield a 2-step periodic SLIP orbit starting at ICr,
    going trough ICl after time Tr and returning to ICr after time Tr + Tl.
    The nadir heights are yminr and yminl for first and second step,
    respectively.

    :args:
        ICr/l (3x float): apex states at right/left apex [y, vx, vz]
        yimr/l (float): nadir heights
        Tr/l (float): step durations
        m (float): mass
        startParams (5x float): initial guess [k, alpha, L0, beta, dE]
            (dE is ignored and can be omitted)

    :returns:
        pr, pl: parameters that yield the corresponding solution, given the
            initial condition ICr
    """
    if len(startParams) == 4:
        k, alpha, L0, beta = startParams
        dE = 0
    else:
        k, alpha, L0, beta, dE = startParams

    
    dER = (ICl[0]-ICr[0])*m*9.81 + .5*m*(ICl[1]**2 + ICl[2]**2 
                                       - ICr[1]**2 - ICr[2]**2)
    dEL = -dER
    
    P0r = [k, alpha, L0, beta, dER]
    P0l = [k, alpha, L0, beta, dEL]

    pr = fl.calcSlipParams3D2(array(ICr), m, array(ICl), yminr, Tr, P0r) 
    pl = fl.calcSlipParams3D2(array(ICl), m, array(ICr), yminl, Tl, P0l) 
    
    return pr, pl
  


def getPeriodicOrbit(statesL, T_L, ymin_L,
                     statesR, T_R, ymin_R,
                     baseParams ,
                     startParams=[14000, 1.16, 1, 0.] ):
    """
    returns a tuple of SLIP parameters, that result in the two-step periodic
    solution defined by <statesL> -> <statesR> -> >statesL>,
    with step time left (right) = <T_L> (<T_R>)
    minimal vertical position left (right) = <ymin_L> (<ymin_R>)
    statesL/R: a list of (left/right) apex states y, vx, vz
    baseParams: dict of base SLIP parameters: g, m (gravity acceleration, mass)
    
    returns: [SL, paramsL, dEL], [SR, paramsR, dER] 
             two tuples of initial apex states and corresponding SLIP
             parameters that yield the two-step periodic solution
             (dE: energy fluctuation)
        
    """    
    SL = mean(vstack(statesL), axis=0) if len(statesL) > 1 else statesL
    SR = mean(vstack(statesR), axis=0) if len(statesR) > 1 else statesR
    tr = mean(hstack(T_R))
    tl = mean(hstack(T_L))
    yminl = mean(hstack(ymin_L))
    yminr = mean(hstack(ymin_R))
    m = baseParams['m']
    g = baseParams['g']
    # energy input right (left) step
    dER = (SL[0]-SR[0])*m*abs(g) + .5*m*(SL[1]**2 + SL[2]**2 
                                       - SR[1]**2 - SR[2]**2)
    dEL = -dER

    # initialize parameters
    PR = copy.deepcopy( baseParams )
    PL = copy.deepcopy( baseParams )
    PL['IC'] = SL    
    PL['dE'] = dEL
    PR['IC'] = SR
    PR['dE'] = dER
    
    # define step params: (y_apex2, T, y_min, vz_apex2)
    spL = (SR[0], tl, yminl, SR[2])
    spR = (SL[0], tr, yminr, SL[2])
    
    # compute necessary model parameters
    paramsL = fl.calcSlipParams3D(spL, PL, startParams)
    paramsR = fl.calcSlipParams3D(spR, PR, startParams)
    
    
    return ([SL, paramsL, dEL],[SR, paramsR, dER])


def twoStepSlip(IC, pl, pr):
    """
    runs two steps with the SLIP model, starting from x0, 
    yielding (x1, x2) apex states
    """
    try:
        res = sl.SLIP_step3D(IC, pl)
        if res['sim_fail']:
            x1 = [x*1.3 for x in IC]
        else:
            x1 = [res['y'][-1], res['vx'][-1], res['vz'][-1]]
        res = sl.SLIP_step3D(x1, pr)
        if res['sim_fail']:
            x2 = [x*1.3 for x in IC]
        else:
            x2 = [res['y'][-1], res['vx'][-1], res['vz'][-1]]
    except ValueError:
        x1 = [x*1.3 for x in IC]
        x2 = [x*1.3 for x in IC]
    
    return (array(x1), array(x2))

def getPeriodicOrbit_p( params_l, params_r, aug_dict = {}, 
                       startState = [1., 3., 0 ]):
    """
    returns a tuple of (stateL, stateR) which - together with params_l, 
    params_r yield a periodic orbit.
    
    params_l, params_r : either array of parameters in the form:
        [k, alpha, L0, beta, dE] in SI-units, or a dictionary with SLIP 
        parameters that can be passed directly to SLIP_step3D
    aug_dict: a dictionary containing {'m' : mass, 'g' : gravitational acc.},
        which is required only if params_l, params_r are given in array form.
    startState: initial guess for periodic orbit starting state (left),
        [y, vx, vz]
    """
    
    if not isinstance(params_l,dict):
        pl = sp_a2d(params_l)
        pl.update(aug_dict)
    else:
        pl = params_l
    
    if not isinstance(params_r,dict):
        pr = sp_a2d(params_r)
        pr.update(aug_dict)
    else:
        pr = params_r

    keys = ['m','g','k','alpha','beta','L0','dE']
    check_pl = [pl.has_key(key) for key in keys]
    check_pr = [pr.has_key(key) for key in keys]    
    
    if (not all(check_pl)) and (not all(check_pr)):
        raise KeyError, 'missing keys in dictionaries (invalid parameters)'
    
    # the deviation from the final apex state from the initial apex state
    delta_fcn = lambda x, p_left, p_right:( twoStepSlip(x,  p_left, p_right)[1]
                                            - array(x) )
    x_fp = opt.fsolve(delta_fcn, startState, args=(pl, pr), xtol=1e-6  )
    return twoStepSlip(x_fp, pl, pr)[::-1]
    
    

def sp_d2a(params):
    """
    transforms a given SLIP parameter set (dict) into an array
    """
    return hstack([ params['k'], params['alpha'], params['L0'], params['beta'],
           params['dE']])[:,newaxis]

def sp_a2d(parameter):
    """
    transforms a given SLIP parameter set (array) into a dict
    *NOTE*: here, the array must have format k|alpha|L0|beta|dE
    """
    params = squeeze(parameter)
    return { 'k' : params[0],
             'alpha' : params[1],
             'L0' : params[2],
             'beta' : params[3],
             'dE' : params[4],
           }


def simCSLIP(x0, x0R, x0L, p0R, p0L, AR, AL, SLIP_param0, n=50):
    """
    simulates the controlled 2step-SLIP.
    input:
        x0 - initial (augmented) state, e.g. [x0L, x0R].T
        x0R - reference right apex (y, vx, vz)
        x0L - reference left apex     -"-
        p0R - reference right parameters
        p0L - reference left parameters
        AR - parameter control right leg
        AL - parameter control left leg
        SLIP_param0: dict, containing {'m': ..., 'g': ... }
        n - number of strides to simulate at most
    """
    res = []
    refStateL = hstack([x0L, x0R])[:,newaxis]
    refStateR = hstack([x0R, x0L])[:,newaxis]
    currState = array(x0)
    slip_params = copy.deepcopy(SLIP_param0)
    if currState.ndim == 1:
        currState = currState[:,newaxis]
    elif currState.shape[0] == 1:
        currState = currState.T
    for step in range(n):        
        pL = sp_d2a(p0L) + dot(AL, currState - refStateL)
        #print 'pL changed:', not allclose(pL,sp_d2a(p0L))
        slip_params.update(sp_a2d(pL))
        try:
            resL = sl.SLIP_step3D(currState[:3,0], slip_params)
        except ValueError:
            print 'simulation aborted (l1)\n'
            break
        if resL['sim_fail']:
            print 'simulation aborted (l2)\n'
            break
        res.append(resL)
        currState = hstack([resL['y'][-1],
                            resL['vx'][-1],
                            resL['vz'][-1],
                            currState[:3,0]])[:,newaxis]
        pR = sp_d2a(p0R) + dot(AR, currState - refStateR)
        #print 'pR changed:', not allclose(pR,sp_d2a(p0R))
        slip_params.update(sp_a2d(pR))
        try:
            resR = sl.SLIP_step3D(currState[:3,0], slip_params)
        except ValueError:
            print 'simulation aborted (r1)\n'
            break
        if resR['sim_fail']:
            print 'simulation aborted (r2)\n'
            break
        res.append(resR)
        currState = hstack([resR['y'][-1],
                            resR['vx'][-1],
                            resR['vz'][-1],
                            currState[:3,0]])[:,newaxis]
    return res

def simCSLIP_xp(x0, x0R, x0L, p0R, p0L, AR, AL, SLIP_param0, n=50):
    """
    simulates the controlled 2step-SLIP, using [x,p]-referenced control
    input:
        x0 - initial (augmented) state, e.g. [x0L, p0R].T
        x0R - reference right apex (y, vx, vz)
        x0L - reference left apex     -"-
        p0R - reference right parameters
        p0L - reference left parameters
        AR - parameter control right leg
        AL - parameter control left leg
        SLIP_param0: dict, containing {'m': ..., 'g': ... }
        n - number of strides to simulate at most
    """
    res = []
    refStateL = hstack([x0L, squeeze(sp_d2a(p0R))])[:,newaxis]
    refStateR = hstack([x0R, squeeze(sp_d2a(p0L))])[:,newaxis]
    currState = array(x0)
    slip_params = copy.deepcopy(SLIP_param0)
    if currState.ndim == 1:
        currState = currState[:,newaxis]
    elif currState.shape[0] == 1:
        currState = currState.T
    for step in range(n):
        #print 'AL: ', AL.shape, 'p0L: ', sp_d2a(p0L).shape
        pL = sp_d2a(p0L) + dot(AL, currState - refStateL)
        #print 'pL changed:', not allclose(pL,sp_d2a(p0L))
        slip_params.update(sp_a2d(pL))
        try:
            resL = sl.SLIP_step3D(currState[:3,0], slip_params)
        except ValueError:
            print 'simulation aborted (l1)\n'
            break
        if resL['sim_fail']:
            print 'simulation aborted (l2)\n'
            break
        res.append(resL)
        currState = hstack([resL['y'][-1],
                            resL['vx'][-1],
                            resL['vz'][-1],
                            squeeze(pL)])[:,newaxis]
        pR = sp_d2a(p0R) + dot(AR, currState - refStateR)
        #print 'pR changed:', not allclose(pR,sp_d2a(p0R))
        slip_params.update(sp_a2d(pR))
        try:
            resR = sl.SLIP_step3D(currState[:3,0], slip_params)
        except ValueError:
            print 'simulation aborted (r1)\n'
            break
        if resR['sim_fail']:
            print 'simulation aborted (r2)\n'
            break
        res.append(resR)
        currState = hstack([resR['y'][-1],
                            resR['vx'][-1],
                            resR['vz'][-1],
                            squeeze(pR)])[:,newaxis]
    return res

def stackSimRes(simRes):
    """
    input: a *list* of single steps
    returns: an array that contains the complete gait (consecutive time & way)
    """
    resDat = []
    res_t = []
    for part in simRes:
        if len(resDat) == 0:
            res_t.append(part['t'])
            resDat.append(vstack( [ part['x'],
                                    part['y'],
                                    part['z'],
                                    part['vx'],
                                    part['vy'],
                                    part['vz'],
                                    ]).T)
        else:
            res_t.append(part['t'][1:] + res_t[-1][-1])
            # compensate x and z translation
            resDat.append(vstack( [ part['x'][1:] + resDat[-1][-1,0],
                                    part['y'][1:],
                                    part['z'][1:] + resDat[-1][-1,2],
                                    part['vx'][1:],
                                    part['vy'][1:],
                                    part['vz'][1:],
                                    ]).T)
    return hstack(res_t), vstack(resDat)

def dS_dP(x0, PR, keys = [('k',750.),('alpha',0.05),('L0',0.05),
                                ('beta',0.05), ('dE', 7.5) ], r_mag = .005):
    """
    calculates the SLIP derivative with respect to 'keys'
    keys is a list of tuples with the keys of PR that should be changed,
    and the order of magnitude of deviation (i.e. something like std(x))
    
    -- only for a single step --
    """
    df = []
    # r_mag = .005 # here: relative magnitude of disturbance in standrad dev's
    
    for elem,mag in keys:
        h = r_mag*mag
        # positive direction
        PRp = copy.deepcopy(PR)
        PRp[elem] += h
        resR = sl.SLIP_step3D(x0, PRp)
        SRp = array([resR['y'][-1], resR['vx'][-1], resR['vz'][-1]])
        #fhp = array(SR2 - x0)
        # positive direction
        PRn = copy.deepcopy(PR)            
        PRn[elem] -= h
        resR = sl.SLIP_step3D(x0, PRn)
        SRn = array([resR['y'][-1], resR['vx'][-1], resR['vz'][-1]])
        #fhn = array(SR2 - x0)
        # derivative: difference quotient
        df.append( (SRp - SRn)/(2.*h) )
    
    return vstack(df).T

def dS_dX(x0, PR, h_mag = .0005):
    """
    calculates the Jacobian of the SLIP at the given point x0,
    with PR beeing the parameters for that step
    coordinates under consideration are:
        y
        vx
        vz
    only for a single step!
    """
    df = []
    for dim in range(len(x0)):
        delta = zeros_like(x0)
        delta[dim] = 1.            
        h = h_mag * delta      
        # in positive direction           
        resRp = sl.SLIP_step3D(x0 + h, PR)
        SRp = array([resRp['y'][-1], resRp['vx'][-1], resRp['vz'][-1]])
        #fhp = array(SR2 - x0)
        # in negative direction
        resRn = sl.SLIP_step3D(x0 - h, PR)
        SRn = array([resRn['y'][-1], resRn['vx'][-1], resRn['vz'][-1]])
        #fhn = array(SR2 - x0)
        # derivative: difference quotient
        df.append( (SRp - SRn)/(2.*h_mag) )
    
    return vstack(df).T

def augStates(allStates, nAug, start=0, nStride=2):
    """
    returns the augmented states
    allStates: list of subsequent apex states
    nAug: numbers of consecutive states to augment
    start: if start=0 -> return even states, start=1: odd states
    nStride: number of consecutive states that build a stride
    
    the first (start + nAug) indices of allStates are skipped.
    (internal comment: you might want to use paramsL1[nAug:,:], or
     paramsR1[nAug:,:])
    """
    aug_states = []
    startState = start + nStride * nAug
    for rep in range(startState, len(allStates), nStride):
        lastIdx = rep - nAug - 1 if rep > nAug else None
        aug_states.append(hstack(allStates[rep:lastIdx:-1]))
    
    return vstack(aug_states)

def finalState(IC, params, addDict=None):
    """
    simulates SLIP with given IC [y, vx, vz] and params.
    returns the next apex state

    *This is a convenience function. It calls SLIP_step3D internally*

    '''''''''''
    Parameter:
    '''''''''''
    IC : *array* (1-by-3)
        the initial conditions of SLIP: apex height, velocity in running and
        lateral direction
    params : *dict* or *array*(1-by-5)
        the parameters of SLIP. These can either be an array, consisting of the
        individual step parameters in the format [k, alpha, L0, beta, dE], or a
        dictionary according to the requirements of SLIP. 
        **NOTE** if params is given as array, addDict must be present and
        contain the entries 'm' (mass) and 'g' (gravity, e.g. -9.81)
    addDict : *dict*
        (only required if params is given as array)

   '''''''''' 
   Returns:
   '''''''''' 
   state : *array* (1-by-3)
        the final apex state after simulating SLIP given the initial condition
        and parameters. Format is [y, vx, vz], same as IC

    """
    simparams = {}
    if type(addDict) is dict:
        simparams.update(addDict)
    if type(params) is not dict:
        simparams.update(sp_a2d(params))
    else:
        simparams.update(params)
    
    res = sl.SLIP_step3D(IC, simparams)
    return array([res['y'][-1], res['vx'][-1], res['vz'][-1]])


def getControlMaps(state_r, state_l, dataset, conf=None, indices=None):
    """
    This function creates the linear mappings that are required for creating a controlled SLIP:
        * parameter prediction schemes
          These are mappings from [CoM state, additional states] -> [SLIP parameters],
          which tell the system how to update the SLIP parameters each step
        * Non-SLIP state propagators
          These are mappings from [CoM state, additional states] -> [additional states],
          which tell the system how the additional states (not part of original SLIP)
          evolve from apex to apex (discrete dynamics).

    As these mappings are affine mappings, the corresponding lifts (constants) are also given.
    This implies that a periodic SLIP solution is computed.
    This function uses bootstrap. 
    
    *NOTE* Do *not* detrend the input - use "original" data!
    *NOTE* It is assumed that the first 3 dimensions of the data contain the CoM state at apex!
        
    :args:
        state_r (n-by-d array): the full state of the system at right apices. 
            First three dimensions are considered as state of the CoM 
            [height, velocity in horizontal plane (2x)]
        state_l (n-by-d array): the full state of the system at left apices. 
            *NOTE* it is assumed that a left apex follows a right apex, i.e. state_r[0, :] is
            before state_l[0, :]
        dataset: a dataset obtained from build_dataset(). Essentially, this is an object with
            .all_param_r  [n-by-5 array]
            .all_param_l  [n-by-5 array]
            .yminL   [list, n elements]
            .yminR   [list, n elements]
            .TR     [list, n elements]
            .TL     [list, n elements]
            .masses [list, n elements]
        conf: a mutils.io.saveable object, containing detrending info  (int
            .dt_window and bool .dt_medfilter fields)
        indices (list or 1d array of int): indices to use for regression. If
            NONE, use bootstrap and average over matrices.

    :returns:
        (ctrl_r, ctrl_l), (prop_r, prop_l), (ref_state_r, ref_state_l, ref_param_r, ref_param_l, 
          ref_addDict)
            Tuples containing the (1) parameter update maps, (2) state propagator maps, 
            (3) reference states and parameters and additional SLIP
    """
    if conf == None:
        conf = mio.saveable()
        conf.dt_window=30
        conf.dt_medfilter=False

    if indices is not None:
        indices = array(indices).squeeze()

    d = dataset # shortcut
    addDict = { 'm' : mean(d.masses), 'g' : -9.81 }
    #def getPeriodicOrbit2(ICr, Tr, yminr, ICl, Tl, yminl, m, startParams=[14000.,
    #[ICp_r, Pp_r, dE_r], [ICp_l, Pp_l, dE_l] =
    ICp_r = mean(d.all_IC_r, axis=0)
    ICp_l = mean(d.all_IC_l, axis=0)
    TR = mean(vstack(d.TR))
    TL = mean(vstack(d.TL))
    yminr = mean(vstack(d.yminR))
    yminl = mean(vstack(d.yminL))
    Pp_r, Pp_l = getPeriodicOrbit2(ICp_r, TR, yminr, ICp_l, TL, yminl,
            mean(d.masses), startParams=mean(vstack(d.all_param_r), axis=0)[:5])
    
    # change last element to be energy input - this is consistent with the format used in the rest of the code
    if False: # obsolete code
        Pp_r = array(Pp_r)
        Pp_l = array(Pp_l)
        Pp_r[4] = dE_r
        Pp_r = Pp_r[:5]
        Pp_l[4] = dE_l
        Pp_l = Pp_l[:5]
    
    # now: detrend data
    # non-SLIP state
    dt_nss_r = fda.dt_movingavg(state_r[:, 3:], conf.dt_window,
            conf.dt_medfilter)
    dt_nss_l = fda.dt_movingavg(state_l[:, 3:],conf.dt_window,
            conf.dt_medfilter)

    # full state
    dt_fs_r = fda.dt_movingavg(state_r, conf.dt_window, conf.dt_medfilter)
    dt_fs_l = fda.dt_movingavg(state_l, conf.dt_window, conf.dt_medfilter)

    # SLIP parameters
    dt_pl = fda.dt_movingavg(d.all_param_l, conf.dt_window, conf.dt_medfilter)
    dt_pr = fda.dt_movingavg(d.all_param_r, conf.dt_window, conf.dt_medfilter)
        
    # compute non-SLIP state prediction maps (propagators)
    if indices is None:
        _, all_prop_r, _ = fda.fitData(dt_fs_r, dt_nss_l, nps=1, nrep=100,
                sections=[0,], rcond=1e-8)
        _, all_prop_l, _ = fda.fitData(dt_fs_l[:-1, :], dt_nss_r[1:, :], nps=1,
                nrep=100, sections=[0,], rcond=1e-8)
        prop_r = fda.meanMat(all_prop_r)
        prop_l = fda.meanMat(all_prop_l)
    else:
        prop_r = dot(dt_nss_l[indices, :].T, pinv(dt_fs_r[indices, :].T, rcond=1e-8))
        prop_l = dot(dt_nss_r[indices[:-1] + 1, :].T, pinv(dt_fs_l[indices[:-1], :].T, rcond=1e-8))

    # compute parameter prediction maps
    if indices is None:
        _, all_ctrl_r, _ = fda.fitData(dt_fs_r, dt_pr, nps=1, nrep=100, sections=[0,], rcond=1e-8)
        _, all_ctrl_l, _ = fda.fitData(dt_fs_l, dt_pl, nps=1, nrep=100, sections=[0,], rcond=1e-8)
        ctrl_r = fda.meanMat(all_ctrl_r)
        ctrl_l = fda.meanMat(all_ctrl_l)
    else:
        ctrl_r = dot(dt_pr[indices, :].T, pinv(dt_fs_r[indices, :].T, rcond=1e-8))
        ctrl_l = dot(dt_pl[indices, :].T, pinv(dt_fs_l[indices, :].T, rcond=1e-8))

    return (ctrl_r, ctrl_l), (prop_r, prop_l), (ICp_r, ICp_l, Pp_r, Pp_l, addDict)

    
def controlled_stride(fullstate, param0_r, param0_l, refstate_r, refstate_l, ctrl_r,
         ctrl_l, facprop_r, facprop_l, addDict, full_info = False):
    """
    This function performs a stride of the controlled SLIP. Control maps and reference
    motion / values must be given.

    :args:
        fullstate ([k+3]x float): the initial full state of the system: [CoM state; factors]
        param0_r (5x float): the reference leg parameters for right leg in
            the periodic motion: [k, alpha, L0, beta, dE]
        param0_l (5x float): the reference leg parameters for left leg
        refstate_r (3x float): the right apex state for periodic motions:
            [height, horiz. speed, lat. speed]. *NOTE* The reference value for factors is always zero.
        refstate_l (3x float): the left apex state for periodic motions
        ctrl_r (2D array): a map that predicts the off-reference values for SLIP parameters as linear function 
            of the full state (which is [CoM state; factors].T). The "right leg controller"
        ctrl_l (2D array): the corresponding "left leg controller"
        facprop_r (2D array): The propagator that maps the right full (apex) state to the factors at left apex.
        facprop_l (2D array): The propagator that maps the left full (apex) state to the factors at right apex.
        addDict (dict): contains 'm' (model mass in kg) and 'g', the gravity (typically -9.81 [ms^-2])
        full_info (bool): if additionally full information (e.g. trajectory) should be returned
    :returns:
        if full_info == False:
            final state ([k+3]x float)): the final full state of the system [CoM state; factors] after one stride.
        if full_info == True:
            (final state ([k+3]x float), [t (array), states (array)] ): final state and [time, trajectory]
    
    *NOTE* it is assumed that the reference values for the "factors" are zero.

    """
    # == first ("right") step ==
    IC = fullstate.squeeze()[:3]    
    d_state_r = fullstate.squeeze() - refstate_r.squeeze()
    # compute SLIP control input
    d_param_r = dot(ctrl_r, d_state_r)
    p_r = param0_r.squeeze() + d_param_r.squeeze()
    # simulate SLIP
    # simulate SLIP
    if full_info:
        # obtain full trajectory etc.
        # create parameter dictionary
        pdict = {} 
        pdict.update(addDict)
        pdict.update(sp_a2d(p_r))
        simres_r = sl.SLIP_step3D(IC, pdict)
        com_state_l = hstack([simres_r['y'][-1], simres_r['vx'][-1], simres_r['vz'][-1]])        
    else:        
        com_state_l = finalState(IC, p_r, addDict).squeeze()    
    # propagate factors state
    facs_state_l = dot(facprop_r, d_state_r).squeeze()
    fullstate_l = hstack([com_state_l, facs_state_l])

    # == second ("left") step ==
    IC_l = fullstate_l[:3]
    d_state_l = fullstate_l - refstate_l.squeeze()
    # compute SLIP control input
    d_param_l = dot(ctrl_l, d_state_l)
    p_l = param0_l.squeeze() + d_param_l.squeeze()
    # simulate SLIP
    if full_info:
        # obtain full trajectory etc.
        # create parameter dictionary
        pdict = {} 
        pdict.update(addDict)
        pdict.update(sp_a2d(p_l))
        simres_l = sl.SLIP_step3D(IC_l, pdict)
        com_state_final = hstack([simres_l['y'][-1], simres_l['vx'][-1], simres_l['vz'][-1]])
        # obtain trajectory etc.
    else:
        com_state_final = finalState(IC_l, p_l, addDict).squeeze()
    # propagate factors state
    facs_state_final = dot(facprop_l, d_state_l).squeeze()
    fullstate_final = hstack([com_state_final, facs_state_final])
    
    if full_info:
        return fullstate_final, stackSimRes([simres_r, simres_l])
    else: 
        return fullstate_final


def get_auto_sys(param0_r, param0_l, refstate_r, refstate_l, ctrl_r, ctrl_l, facprop_r, facprop_l, addDict, full_info=False):
    """
    Returns a function f that expresses the autonomous system: x_(n+1) = f (x_n).
    This is a convenience function for e.g. calculating the jacobian.

    :args:
        param0_r (5x float): the reference leg parameters for right leg in
            the periodic motion: [k, alpha, L0, beta, dE]
        param0_l (5x float): the reference leg parameters for left leg
        refstate_r (3x float): the right apex state for periodic motions:
            [height, horiz. speed, lat. speed]. *NOTE* The reference value for factors is always zero.
        refstate_l (3x float): the left apex state for periodic motions
        ctrl_r (2D array): a map that predicts the off-reference values for SLIP parameters as linear function 
            of the full state (which is [CoM state; factors].T). The "right leg controller"
        ctrl_l (2D array): the corresponding "left leg controller"
        facprop_r (2D array): The propagator that maps the right full (apex) state to the factors at left apex.
        facprop_l (2D array): The propagator that maps the left full (apex) state to the factors at right apex.
        addDict (dict): contains 'm' (model mass in kg) and 'g', the gravity (typically -9.81 [ms^-2])
        full_info (bool): if additionally the trajectory should be returned
    :returns:
        a lambda function f(fullstate) that calls the controlled_stride function using the given parameters.        
    """
    return lambda fullstate: controlled_stride(fullstate, param0_r, param0_l, refstate_r, refstate_l, ctrl_r,
         ctrl_l, facprop_r, facprop_l, addDict, full_info)
 
