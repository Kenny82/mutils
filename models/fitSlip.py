# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:09:27 2011

@author: moritz
"""

# obsolete version
#from SLIP.SLIP2 import sim
import scipy.optimize as opt
from models.slip import SLIP_step, SLIP_step3D
import models.slip as sl
from pylab import hstack, vstack, norm, array, dot, pinv


# fits the SLIP model to a given set of parameters


def delta(slip_params, data, IC_and_fixparams,returnOnlyDkDL = False):
    """
    returns the differences of a step data
    slip_params: k0,alpha,l0
    data: y_apex2, T, y_min: from experiment
    IC_and_fixparams: a dictionary containing  vx0, y0, dE, m, ygrd, 
       fs (hint: 5000) and steps (hint: 1)
    returnOnlyDkDl: if true, ONLY k2 and L2 are returned
    """
    
    params = { 'k' : slip_params[0],
               'alpha' : slip_params[1],
               'L0' : slip_params[2],
               'm' : IC_and_fixparams ['m'],
               'dE': IC_and_fixparams['dE'],
               'g':  IC_and_fixparams['g'],
                }
    
    IC = IC_and_fixparams['IC']
    try:
        sim_res = SLIP_step(IC,params)
    except ValueError:
        return [10, 10, 10]
    if sim_res['sim_fail']:
        return [11, 11, 11]
     
    if returnOnlyDkDL:
        return sim_res['dK'], sim_res['dL']
    else:
        return ( sim_res['y'][-1] - data[0], sim_res['t'][-1] - data[1],
                 min(sim_res['y']) - data[2] )


def calcSlipParams(step_params, model_params, x0=[18000.,1.16,.99]):
    """
    just a wrapper...
    step_params: a triple y_apex2,T,y_min
    model_params: a dictionary with IC [y0, vx0], m, dE
    (hint: 1)
    """
    IC_and_params = {'IC': model_params['IC'],
                        'g': -9.81,
                        'm': model_params['m'],
                        'dE': model_params['dE']}
    y_apex2, T, y_min = step_params
    k,alpha0,l0 = opt.fsolve(delta, x0 = x0,
                             args=([y_apex2,T,y_min],IC_and_params),
                            maxfev = 100,factor=10, xtol=1e-6 )
    
    sim_params = { 'k' : k, 
                  'alpha': alpha0,
                  'L0': l0,
                  'm' : model_params['m'],
                  'dE': model_params['dE'],
                  'g' : -9.81  }
    #k2,l02 = delta((k,alpha0,l0),[y_apex2,T,y_min],IC,True)
    sim_res = SLIP_step(model_params['IC'],sim_params)
    return k,alpha0,l0 , sim_res['dk'], sim_res['dL']


def delta3D(slip_params, data, IC_and_fixparams,returnOnlyDkDL = False):
    """
    returns the differences of a step data - 3D version
    slip_params: k0,alpha,l0,beta
    data: y_apex2, T, y_min, vz_apex2: from experiment
    IC_and_fixparams: a dictionary containing  vx0, vz0, y0, dE, m, ygrd, 
       
    returnOnlyDkDl: if true, ONLY k2 and L2 are returned
    """
    
    params = { 'k' : slip_params[0],
               'alpha' : slip_params[1],
               'L0' : slip_params[2],
               'beta' : slip_params[3],
               'm' : IC_and_fixparams ['m'],
               'dE': IC_and_fixparams['dE'],
               'g':  IC_and_fixparams['g'],
                }
    
    IC = IC_and_fixparams['IC']
    try:
        sim_res = SLIP_step3D(IC,params)
    except ValueError:
        return [10, 10, 10, 10]
    if sim_res['sim_fail']:
        return [11, 11, 11, 11]
     
    if returnOnlyDkDL:
        return sim_res['dK'], sim_res['dL']
    else:
        return ( sim_res['y'][-1] - data[0], sim_res['t'][-1] - data[1],
                 min(sim_res['y']) - data[2], sim_res['vz'][-1] - data[3] )


def calcSlipParams3D(step_params, model_params, x0=[18000.,1.16,.99,0.],factor=10):
    """
    just a wrapper...
    step_params: a tuple (y_apex2, T, y_min, vz_apex2) to be matched
    model_params: a dictionary with IC [y0, vx0, vz0], m, dE    
    factor: parameter to determine the solver's initial step size;
    """
    IC_and_params = {'IC': model_params['IC'],
                     'g': -9.81,
                     'm': model_params['m'],
                     'dE': model_params['dE']}
    #y_apex2, T, y_min, vz_apex2 = step_params
    k, alpha0, l0, beta0 = opt.fsolve(delta3D, x0 = x0,
                             args=(step_params,IC_and_params),
                            maxfev = 100,factor=factor, xtol=1e-6 )       
    
    sim_params = { 'k' : k, 
                  'alpha': alpha0,
                  'L0': l0,
                  'beta' : beta0,
                  'm' : model_params['m'],
                  'dE': model_params['dE'],
                  'g' : -9.81  }
    #k2,l02 = delta((k,alpha0,l0),[y_apex2,T,y_min],IC,True)
    sim_res = SLIP_step(model_params['IC'],sim_params)
    return k,alpha0,l0,beta0, sim_res['dk'], sim_res['dL']


def calcSlipParams3D2(IC, m, FS, ymin, T, P0 = [14000., 1.16, 1., 0., 0.]):
    """
    calculates a set of SLIP parameters that result in the desired motion.
    
    :args:
        IC (3x float) : initial condition y, vx, vz
        FS (3x float) : final state y, vx, vz
        ymin : minimal vertical excursion
        T : total step duration
        P0 (5x float) : initial guess for parameters [k, alpha, L0, beta, dE]
            (dE is ignored. It is calculated from IC and FS)
        
    :returns:
        [k, alpha, L0, beta, dE] parameter vector that results in the desired state
        
       
    """
    
    dE = (FS[0]-IC[0])*m*9.81 + .5*m*(FS[1]**2 + FS[2]**2 
                                       - IC[1]**2 - IC[2]**2)    
    k, alpha, L0, beta, _ = P0
    
    def getDiff(t, y):
        """ returns the difference in desired params """
        delta = [T - t[-1],
                 ymin - min(y[:,1]),
                 #FS - y[-1,[1,3,5]],
                 FS[[0,2]] - y[-1,[1,5]]
                 ]
        return hstack(delta)
    
    rcond = 1e-3 # for pinverting the jacobian. this will be adapted during the process
    init_success = False
    while not init_success:
        try:
            pars = [k, L0, m, alpha, beta, dE]
            t, y = sl.qSLIP_step3D(IC, pars)
            init_success = True
        except ValueError:
            L0 -= .02

    d0 = getDiff(t, y)
    nd0 = norm(d0)
    #print \"difference: \", nd0, d0
    cancel = False
    niter = 0
    while nd0 > 1e-6 and not cancel:
        niter += 1
        if niter > 20:
            print "too many iterations"
            cancel = True
            break
        # calculate jacobian dDelta / dP
        #print \"rep:\", niter
        
        
        hs = array([10, .0001, .0001, .0001])
        pdims = [0, 1, 3, 4]
        J = []
        for dim in range(4):
            parsp = [k, L0, m, alpha, beta, dE][:]
            parsm = [k, L0, m, alpha, beta, dE][:]
            
            parsp[pdims[dim]] += hs[dim]
            parsm[pdims[dim]] -= hs[dim]
            
            t, y = sl.qSLIP_step3D(IC, parsm)
            dm = getDiff(t, y)
            
            # circumvent "too low starting condition":
            try:
                t, y = sl.qSLIP_step3D(IC, parsp)
                dp = getDiff(t, y)
                J.append((dp - dm)/(2*hs[dim]))
            except ValueError:
                # run unilateral derivative instead
                parsp = [k, L0, m, alpha, beta, dE]
                t, y = sl.qSLIP_step3D(IC, parsm)
                dp = getDiff(t, y)
                J.append((dp - dm)/(hs[dim]))
                print "|1>"

            
            
        J = vstack(J).T
        pred = [k, L0, alpha, beta]
        update = dot(pinv(J, rcond=rcond), d0)
        nrm = norm(update * array([.001, 10, 10, 10]))
        success = False
        cancel = False
        rep = 0
        while not (success or cancel):
            rep += 1
            #print \"update:\", update
            if rep > 3:
                # reset
                if rcond < 1e-10:
                    #print \"rcond too small!\"
                    cancel = True
                else:
                    rcond /= 100
                pars = [k, L0, m, alpha, beta, dE]
                break           
            dk, dL0, dalpha, dbeta = update            
            pars = [k - dk, L0 - dL0, m, alpha - dalpha, beta - dbeta, dE]
            try:
                t, y = sl.qSLIP_step3D(IC, pars)
                d0 = getDiff(t, y)                
                #print "d0 = ", norm(d0)
            except (ValueError, sl.SimFailError, TypeError):
                #print "escaping ..."
                update /= 2
                continue
            if norm(d0) < nd0:
                success = True
                nd0 = norm(d0)
            else:
                update /= 2
            
        #print \"difference: \", norm(d0), d0        
        k, L0, alpha, beta = array(pars)[[0,1,3,4]]
    
    #if not cancel:
    #    print \"converged!\"
    return array([k, alpha, L0, beta, dE])


def getPeriodicOrbit2(ICr, Tr, yminr, ICl, Tl, yminl, m,
                     startParams=[14000., 1., 1.16, 0.] ):
    """
    returns a tuple of SLIP parameters, that result in the two-step periodic
    solution defined by <ICr> -> <ICl> -> <ICr>,
    with step time left (right) = <Tl> (<Tr>)
    minimal vertical position left (right) = <yminl> (<yminr>)
    
    :args:
        ICr/l (3-by-1 array): apex states: y, vx, vz
        m : mass
        startParams (list(4)) :[k, L0, alpha, beta] initial guess
    
    returns: [ppr], [ppl]: lists of periodic SLIP parameters
                (format: [k, alpha, L0, beta, dE] )

    """
   
    # energy input right (left) step
    dER = (ICl[0]-ICr[0])*m*9.81 + .5*m*(ICl[1]**2 + ICl[2]**2 
                                       - ICr[1]**2 - ICr[2]**2)
    dEL = -dER
    
    paramsR = calcSlipParams3D2(ICr, m, ICl, yminr, Tr, startParams)
    paramsL = calcSlipParams3D2(ICl, m, ICr, yminl, Tl, paramsR[[0,2,1,3]])

    
    return paramsR, paramsL

