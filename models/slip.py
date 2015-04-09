# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:46:14 2011

@author: moritz
"""

# This file implements the SLIP model

from scipy.integrate.vode import dvode, zvode
from scipy.integrate import odeint
from pylab import (zeros, sin, cos, sqrt, array, linspace,
                   arange, ones_like, hstack, vstack, argmin,
                   find, interp, allclose,
                   sign)
import numpy as np
from copy import deepcopy

import sys

import mutils.makeODE as mo





# --- begin C-code for the SLIP model
_model_code_base = r"""
    
/* some declarations for the solver */
/* note: math.h is already included */
#define ODE_DIM 6 /* excluding time */
#define ODE_NPAR 5 /* k, l0, m, x_foot, z_foot  NOTE: touchdown is computed in python */
#define ODE_DT 1e-3 /* max step / initial */

/* solver accuracy (optional, defaults are less accurate) */
#define ODE_RTOL 1e-11
#define ODE_ATOL 1e-11

void rhs(unsigned n, double t, double *Y, double *p, double *f)
{
    /* function signature
    n: iteration
    t: simulation time
    Y: state
    p: parameter
    f: return value (the "output")
    */

    float l, force_m;
    l = sqrt((Y[0] - p[3]) * (Y[0] - p[3]) + Y[1] * Y[1] + (Y[2] - p[4]) *
         (Y[2]- p[4]));
    force_m = -p[0] * (l - p[1]) / p[2];
    f[0] = Y[3];
    f[1] = Y[4];
    f[2] = Y[5];
    f[3] = (Y[0] - p[3]) / l * force_m;
    f[4] = Y[1] / l * force_m  - 9.81;
    f[5] = (Y[2] - p[4]) / l * force_m;

}


"""

_model_km_code_base = r"""
    
/* some declarations for the solver */
/* note: math.h is already included */
#define ODE_DIM 6 /* excluding time */
#define ODE_NPAR 5 /* k/m, l0, [not used], x_foot, z_foot  NOTE: touchdown is computed in python */
#define ODE_DT 1e-3 /* max step / initial */

/* solver accuracy (optional, defaults are less accurate) */
#define ODE_RTOL 1e-12
#define ODE_ATOL 1e-12

void rhs(unsigned n, double t, double *Y, double *p, double *f)
{
    /* function signature
    n: iteration
    t: simulation time
    Y: state
    p: parameter
    f: return value (the "output")
    */

    float l, force_m;
    l = sqrt((Y[0] - p[3]) * (Y[0] - p[3]) + Y[1] * Y[1] + (Y[2] - p[4]) *
         (Y[2]- p[4]));
    force_m = -p[0] * (l - p[1]);
    f[0] = Y[3];
    f[1] = Y[4];
    f[2] = Y[5];
    f[3] = (Y[0] - p[3]) / l * force_m;
    f[4] = Y[1] / l * force_m  - 9.81;
    f[5] = (Y[2] - p[4]) / l * force_m;

}


"""


_model1_evt_code = r"""


double event(double t, double *Y, int n, double *p, double *res)
{
    /* Trigger: vy > 0  */
    res[0] = Y[4];
    return res[0];
}

"""

_model2_evt_code = r"""

double event(double t, double *Y, int n, double *p, double *res)
{
    /* Trigger: l > l0  */
    float l;
    l = sqrt((Y[0] - p[3]) * (Y[0] - p[3]) + Y[1] * Y[1] + (Y[2] - p[4]) *
         (Y[2]- p[4]));
    res[0] = l - p[1];
}

"""

model1_code = ''.join([_model_code_base, _model1_evt_code])
model2_code = ''.join([_model_code_base, _model2_evt_code])

modelkm1_code = ''.join([_model_km_code_base, _model1_evt_code])
modelkm2_code = ''.join([_model_km_code_base, _model2_evt_code])

_slipmdl1 = mo.makeODE(model1_code, '_slip3D_1_a')
_slipmdl2 = mo.makeODE(model2_code, '_slip3D_2_a')

_slipmdl1_km = mo.makeODE(modelkm1_code, '_slip3D_1_km')
_slipmdl2_km = mo.makeODE(modelkm2_code, '_slip3D_2_km')

# --- end c-code for SLIP model



def dk_dL(L0,k,L,dE):
    """
    computes the required stiffness change and rest length change
    to inject energy without changing the spring force
    """
    dL = 2.*dE/(k*(L0 - L))
    dk = k*((L-L0)/(L-(L0+dL)) - 1.)
    return dk,dL

class SimFailError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

def SLIP_step(IC, SLIP_params): 
    """
    simulates the SLIP

    Parameters:
    -----------
    IC : *array* (1x3)
        initial state vector, containing y0, vx0
        (x0 is assumed to be 0; vy0 = 0 (apex);
        also ground level = 0)
    SLIP_params : *dict*
        A dictionary containing the following keys: *k*, *L0*, *m*, *alpha*,
        *dE* (energy input at midstance), *g* (gravity, typically -9.81)

    Returns:
    --------
    sim_data, sim_state
    sim_data : *dict*
        the results of the simulation
    sim_states: *dict*
        information about the simulation (e.g. errors)

    .. note::
        This docstring might be out of date!
        
    """

    alpha = SLIP_params['alpha']
    k = SLIP_params['k']
    L0 = SLIP_params['L0']
    dE = SLIP_params['dE']
    g = SLIP_params['g']
    m = SLIP_params['m']
    
    y0 = IC[0]
    vx0 = IC[1]
    
    if g >= 0:
        raise ValueError, "gravity points into wrong direction!"
    # concatenate state vector of four elements:
    # (1) time to touchdown
    # (2) time to vy = 0
    # (3) time to takeoff
    # (4) time to apex
    # (1) and (4) are analytically described
    
    y_land = L0*sin(alpha)
    if y0 < y_land:
        raise ValueError, "invalid starting condition"
    
    # before starting, define the model:
    def SLIP_ode(y,t,params):
        """
        defines the ODE of the SLIP, under stance condition
        state: 
            [x
             y
             vx
             vy]
        params:
            {'L0' : leg rest length
             'x0' : leg touchdown position
             'k'  : spring stiffness
             'm'  : mass}
        """

        dy0 = y[2]
        dy1 = y[3]
        L = sqrt((y[0]-params['xF'])**2 + y[1]**2)
        F = params['k']*(params['L0']-L)
        Fx = F*(y[0]-params['xF'])/L
        Fy = F*y[1]/L
        dy2 = Fx/m
        dy3 = Fy/m + params['g']
        return hstack([dy0,dy1,dy2,dy3])
    
    
    def sim_until(IC, params, stop_fcn, tmax = 2.):
        """
        simulated the SLIP_ode until stop_fcn has a zero-crossing
        includes a refinement of the time at this instant
        stop_fcn must be a function of the system state, e.g.
        stop_fcn(IC) must exist
        
        this function is especially adapted to the SLIP state,
        so it uses dot(x1) = x3, dot(x2) = x4
        tmax: maximal simulation time [s]
        """
        init_sign = sign(stop_fcn(IC))
        #1st: evaluate a certain fraction
        tvec_0 = .0001*arange(50)
        sim_results = []
        sim_tvecs = []
        newIC = IC
        sim_results.append (odeint(SLIP_ode,newIC,tvec_0,
                     args=(params,),rtol=1e-12))
        sim_tvecs.append(tvec_0)
        check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
        t_tot = 0.
        while min(check_vec) > 0:
            newIC = sim_results[-1][-1,:]
            sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
                     args=(params,),rtol=1e-12))
            sim_tvecs.append(tvec_0)
            check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
            t_tot += tvec_0[-1]
            # time exceeded or ground hit
            if t_tot > tmax or min(sim_results[-1][:,1] < 0):
                raise SimFailError, "simulation failed"
            
    
        # now: zero-crossing detected
        # -> refine!
        minidx = find(array(check_vec) < 0)[0]
        if minidx == 0:
            # this should not happen because the first value in
            # check_vec should be BEFORE the zero_crossing by
            # construction
            raise ValueError, "ERROR: this should not happen!"
        # refine simulation by factor 50, but only for two
        # adjacent original time frames
        newIC = sim_results[-1][minidx-1,:]
        sim_results[-1] = sim_results[-1][:minidx,:]
        sim_tvecs[-1] = sim_tvecs[-1][:minidx]
        # avoid that last position can be the zero-crossing
        n_refine = 10000
        tvec_0 = linspace(tvec_0[0], tvec_0[1] + 2./n_refine, n_refine+2) 
        sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
                    args=(params,),rtol=1e-12))
        sim_tvecs.append(tvec_0)
        
        # linearly interpolate to zero
        check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
        minidx = find(array(check_vec) < 0)[0]
        if minidx == 0:
            # this should not happen because the first value in
            # check_vec should be BEFORE the zero_crossing by
            # construction
            raise ValueError, "ERROR: this should not happen! (2)"
        
        # compute location of zero-crossing
        y0 = sim_results[-1][minidx-1,:]
        y1 = sim_results[-1][minidx,:]
        fcn0 = stop_fcn(y0)
        fcn1 = stop_fcn(y1)        
        t0 = tvec_0[minidx-1]
        t1 = tvec_0[minidx]
        t_zero = t0 - (t1-t0)*fcn0/(fcn1 - fcn0)
        # cut last simulation result and replace last values
        # by interpolated values
        sim_results[-1] = sim_results[-1][:minidx+1,:]
        sim_tvecs[-1] = sim_tvecs[-1][:minidx+1]
        
        for coord in arange(sim_results[-1].shape[1]):
            sim_results[-1][-1,coord] = interp(
                t_zero, [t0,t1],
                [sim_results[-1][-2,coord], sim_results[-1][-1,coord]] )
        sim_tvecs[-1][-1] = t_zero
        #newIC = sim_results[-1][minidx-1,:]
        #sim_results[-1] = sim_results[-1][:minidx,:]
        #sim_tvecs[-1] = sim_tvecs[-1][:minidx]
        #tvec_0 = linspace(tvec_0[0],tvec_0[1],100)
        #sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
        #            args=(params,),rtol=1e-9))
        #sim_tvecs.append(tvec_0)          
        
     
        
        
        # concatenate lists
        sim_data = vstack( [x[:-1,:] for x in sim_results[:-1] if x.shape[0] > 1]
                           + [sim_results[-1],])
        sim_time = [sim_tvecs[0],]
        for idx in arange(1,len(sim_tvecs)):
            sim_time.append(sim_tvecs[idx] + sim_time[-1][-1])
        sim_time = hstack([x[:-1] for x in sim_time[:-1]] + [sim_time[-1],])
        
        return sim_data, sim_time
        
      
        
    # Section 1: time to touchdown  
    # TODO: make sampling frequency regular
    t_flight1 = sqrt(-2.*(y0 - y_land)/g) 
    #t_flight = sqrt()
    tvec_flight1 = .001*arange(t_flight1*1000.)
    vy_flight1 = tvec_flight1*g
    y_flight1 = y0 + .5*g*(tvec_flight1**2)
    x_flight1 = vx0*tvec_flight1
    vx_flight1 = vx0*ones_like(tvec_flight1)
    
    # Section 2: time to vy = 0
    # approach: calculate forward -> estimate interval of 
    # zero position of vy -> refine simulation in that interval
    # until a point with vy sufficiently close to zero is in the 
    # resulting vector
    params = {'L0' : L0,
              'xF' : t_flight1*vx0 + L0*cos(alpha),
              'k'  : k,
              'm'  : m,
              'g'  : g}
    
    IC = array([t_flight1*vx0, y_land, vx0, t_flight1*g])

    # initial guess: L0*cos(alpha)/vx0    
    #t_sim1 = L0*cos(alpha)/vx0    
    # TODO: implement sim_fail check!
    sim_fail = False
    try:
        sim_phase2, t_phase2 = sim_until(IC,params,lambda x: x[3]) 
        t_phase2 += t_flight1
    except SimFailError:
        print 'simulation aborted (phase 2)\n'
        sim_fail = True
    
    
    # Phase 3:
    if not sim_fail:
        L = sqrt(sim_phase2[-1,1]**2 + (sim_phase2[-1,0]-params['xF'])**2 )
        dk, dL = dk_dL(L0,k,L,dE)
        params2 = deepcopy(params)
        params2['k'] += dk
        params2['L0'] += dL
        IC = sim_phase2[-1,:]
        compression = (lambda x: sqrt( 
                                     (x[0]-params2['xF'])**2 + x[1]**2) 
                                      - params2['L0'] )
        #print ('L:', L, 'dk', dk, 'dL', dL, 'dE', dE, '\ncompression:', compression(IC),
        #      'IC', IC)
        try:                    
            sim_phase3, t_phase3 = sim_until(IC, params2,compression) 
            sim_phase3 = sim_phase3[1:,:]
            t_phase3 = t_phase3[1:] + t_phase2[-1]
        except SimFailError:
            print 'simulation aborted (phase 3)\n'
            sim_fail = True
        
        
    # Phase 4:
    if not sim_fail:
        # time to apex
        # TODO: make sampling frequency regular
        vy_liftoff = sim_phase3[-1,3] 
        t_flight2 = -1.*vy_liftoff/g
        #t_flight = sqrt()
        tvec_flight2 = arange(t_flight2,0,-.001)[::-1]
        vy_flight2 = tvec_flight2*g + vy_liftoff        
        y_flight2 = (sim_phase3[-1,1] + vy_liftoff*tvec_flight2 
                    + .5*g*(tvec_flight2**2) )
        x_flight2 = sim_phase3[-1,0] + sim_phase3[-1,2]*tvec_flight2
        vx_flight2 = sim_phase3[-1,2]*ones_like(tvec_flight2)
        tvec_flight2 += t_phase3[-1]
    
    # todo: return data until error
    if sim_fail:
        return { 't': None, 
                'x': None,
                'y': None,
                'vx': None,
                'vy': None,
                'sim_fail': sim_fail,
                'dk': None,
                'dL': None
                }
        
        
        
    # finally: concatenate phases    
    x_final = hstack([x_flight1, sim_phase2[:,0], sim_phase3[:,0], x_flight2 ])
    y_final = hstack([y_flight1, sim_phase2[:,1], sim_phase3[:,1], y_flight2 ])
    vx_final= hstack([vx_flight1, sim_phase2[:,2], sim_phase3[:,2], vx_flight2])
    vy_final= hstack([vy_flight1, sim_phase2[:,3], sim_phase3[:,3], vy_flight2])    
    tvec_final = hstack([tvec_flight1, t_phase2, t_phase3, tvec_flight2 ])
    
    
    return {'t': tvec_final, 
            'x': x_final,
            'y': y_final,
            'vx': vx_final,
            'vy': vy_final,
            'sim_fail': sim_fail,
            'dk': dk,
            'dL': dL,
            #'sim_res':sim_res,
            #'sim_phase2': sim_phase2_cut,
            #'t_phase2': t_phase2_cut
            }



def SLIP_step3D(IC, SLIP_params, use_legacy=False):
    """
    simulates the SLIP in 3D

    :args:
        IC: initial state vector, containing y0, vx0, vz0
            (x0 is assumed to be 0;
             z0  is assumed to be 0; 
             vy0 = 0 (apex);
            also ground level = 0)
        SLIP_params(dict):
            k
            L0
            m
            alpha : "original" angle of attack
            beta  : lateral leg turn
                    foot position relative to CoM in flight:
                        xF = vx0*t_flight + L0*cos(alpha)*cos(beta)
                        yF = -L0*sin(alpha)
                        zF = vz0*t_flight - L0*cos(alpha)*sin(beta)
            dE: energy change in "midstance" by changing k and L0
            g: gravity (negative! should be ~ -9.81 for SI)
        use_legacy (bool): Use legacy code or SLIP model in C

    :returns:
        sim_data, sim_state
        sim_data : *dict*
            the results of the simulation
        sim_states: *dict*
            information about the simulation (e.g. errors)

    """


    alpha = SLIP_params['alpha']
    beta = SLIP_params['beta']
    k = SLIP_params['k']
    L0 = SLIP_params['L0']
    dE = SLIP_params['dE']
    g = SLIP_params['g']
    m = SLIP_params['m']
    

    y0 = IC[0]
    vx0 = IC[1]
    vz0 = IC[2]
    
    # model code in C

    #if models == None:
    #    models = []
    #    models.append(mo.makeODE(model1_code, '_slip3D_1',skiphash=skiphash))
    #    models.append(mo.makeODE(model2_code, '_slip3D_2',skiphash=skiphash))


    if not allclose(g, -9.81):
        raise ValueError, "currently gravity is hard-coded to -9.81"
    # concatenate state vector of four elements:
    # (1) time to touchdown
    # (2) time to vy = 0
    # (3) time to takeoff
    # (4) time to apex
    # (1) and (4) are analytically described
    
    y_land = L0*sin(alpha)
    if y0 < y_land:
        raise ValueError, "invalid starting condition"
    
    # before starting, define the model:
    def SLIP_ode(y,t,params):
        """
        defines the ODE of the SLIP, under stance condition
        state: 
            [x
             y
             z
             vx
             vy
             vz]
        params:
            {'L0' : leg rest length
             'x0' : leg touchdown position
             'k'  : spring stiffness
             'm'  : mass
             'xF' : anterior foot position
             'zF' : lateral foot position }
        """

        dy0 = y[3]
        dy1 = y[4]
        dy2 = y[5]
        L = sqrt((y[0]-params['xF'])**2 + y[1]**2 + (y[2]-params['zF'])**2)
        F = params['k']*(params['L0']-L)
        Fx = F*(y[0]-params['xF'])/L
        Fy = F*y[1]/L
        Fz = F*(y[2]-params['zF'])/L
        dy3 = Fx/m
        dy4 = Fy/m + params['g']
        dy5 = Fz/m
        return hstack([dy0,dy1,dy2,dy3,dy4,dy5])
    
    
    def sim_until(IC, params, stop_fcn, tmax = 2.):
        """
        simulated the SLIP_ode until stop_fcn has a zero-crossing
        includes a refinement of the time at this instant
        stop_fcn must be a function of the system state, e.g.
        stop_fcn(IC) must exist
        
        this function is especially adapted to the SLIP state,
        so it uses dot(x1) = x3, dot(x2) = x4
        tmax: maximal simulation time [s]
        """
        init_sign = sign(stop_fcn(IC))
        #1st: evaluate a certain fraction
        tvec_0 = .001*arange(50)
        sim_results = []
        sim_tvecs = []
        newIC = IC
        sim_results.append (odeint(SLIP_ode,newIC,tvec_0,
                     args=(params,),rtol=1e-9))
        sim_tvecs.append(tvec_0)
        check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
        t_tot = 0.
        while min(check_vec) > 0:
            newIC = sim_results[-1][-1,:]
            sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
                     args=(params,),rtol=1e-9))
            sim_tvecs.append(tvec_0)
            check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
            t_tot += tvec_0[-1]
            # time exceeded or ground hit
            if t_tot > tmax or min(sim_results[-1][:,1] < 0):
                raise SimFailError, "simulation failed"
            
    
        # now: zero-crossing detected
        # -> refine!
        minidx = find(array(check_vec) < 0)[0]
        if minidx == 0:
            # this should not happen because the first value in
            # check_vec should be BEFORE the zero_crossing by
            # construction
            raise ValueError, "ERROR: this should not happen!"
        # refine simulation by factor 50, but only for two
        # adjacent original time frames
        newIC = sim_results[-1][minidx-1,:]
        sim_results[-1] = sim_results[-1][:minidx,:]
        sim_tvecs[-1] = sim_tvecs[-1][:minidx]
        # avoid that last position can be the zero-crossing
        n_refine = 100
        tvec_0 = linspace(tvec_0[0], tvec_0[1] + 2./n_refine, n_refine+2) 
        sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
                    args=(params,),rtol=1e-9))
        sim_tvecs.append(tvec_0)
        
        # linearly interpolate to zero
        check_vec = [init_sign*stop_fcn(x) for x in sim_results[-1]]
        minidx = find(array(check_vec) < 0)[0]
        if minidx == 0:
            # this should not happen because the first value in
            # check_vec should be BEFORE the zero_crossing by
            # construction
            raise ValueError, "ERROR: this should not happen! (2)"
        
        # compute location of zero-crossing
        y0 = sim_results[-1][minidx-1,:]
        y1 = sim_results[-1][minidx,:]
        fcn0 = stop_fcn(y0)
        fcn1 = stop_fcn(y1)        
        t0 = tvec_0[minidx-1]
        t1 = tvec_0[minidx]
        t_zero = t0 - (t1-t0)*fcn0/(fcn1 - fcn0)
        # cut last simulation result and replace last values
        # by interpolated values
        sim_results[-1] = sim_results[-1][:minidx+1,:]
        sim_tvecs[-1] = sim_tvecs[-1][:minidx+1]
        
        for coord in arange(sim_results[-1].shape[1]):
            sim_results[-1][-1,coord] = interp(
                t_zero, [t0,t1],
                [sim_results[-1][-2,coord], sim_results[-1][-1,coord]] )
        sim_tvecs[-1][-1] = t_zero
        #newIC = sim_results[-1][minidx-1,:]
        #sim_results[-1] = sim_results[-1][:minidx,:]
        #sim_tvecs[-1] = sim_tvecs[-1][:minidx]
        #tvec_0 = linspace(tvec_0[0],tvec_0[1],100)
        #sim_results.append ( odeint(SLIP_ode, newIC, tvec_0,
        #            args=(params,),rtol=1e-9))
        #sim_tvecs.append(tvec_0)          
        
     
        
        
        # concatenate lists
        sim_data = vstack( [x[:-1,:] for x in sim_results[:-1] if x.shape[0] > 1]
                           + [sim_results[-1],])
        sim_time = [sim_tvecs[0],]
        for idx in arange(1,len(sim_tvecs)):
            sim_time.append(sim_tvecs[idx] + sim_time[-1][-1])
        sim_time = hstack([x[:-1] for x in sim_time[:-1]] + [sim_time[-1],])
        
        return sim_data, sim_time
        
      
        
    # --- Section 1: time to touchdown  
    # TODO: make sampling frequency regular
    t_flight1 = sqrt(-2.*(y0 - y_land)/g) 
    #t_flight = sqrt()
    tvec_flight1 = .01*arange(t_flight1*100.)
    vy_flight1 = tvec_flight1*g
    y_flight1 = y0 + .5*g*(tvec_flight1**2)
    x_flight1 = vx0*tvec_flight1
    vx_flight1 = vx0*ones_like(tvec_flight1)
    z_flight1 = vz0*tvec_flight1
    vz_flight1 = vz0*ones_like(tvec_flight1)
    x_TD = vx0*t_flight1
    z_TD = vz0*t_flight1

    # --- Section 2: time to vy = 0
    # approach: calculate forward -> estimate interval of 
    # zero position of vy -> refine simulation in that interval
    # until a point with vy sufficiently close to zero is in the 
    # resulting vector
    params = {'L0' : L0,
              'xF' : t_flight1*vx0 + L0*cos(alpha)*cos(beta),
              'zF' : t_flight1*vz0 - L0*cos(alpha)*sin(beta),
              'k'  : k,
              'm'  : m,
              'g'  : g}

    buffsize = 8000
    buf = zeros((buffsize, _slipmdl1.WIDTH), dtype=np.float64)
    IC = array([x_TD, y_land, z_TD, vx0, t_flight1*g, vz0])
    buf[0, 1:] = IC
    buf[0,0] = t_flight1
    
    # for the model in C
    xF = t_flight1*vx0 + L0*cos(alpha)*cos(beta)
    zF = t_flight1*vz0 - L0*cos(alpha)*sin(beta)
    #print k, L0, m, xF, zF
    
    #print "IC=", IC
    #print "t0:", t_flight1
    if use_legacy: # skip original code

        # initial guess: L0*cos(alpha)/vx0    
        #t_sim1 = L0*cos(alpha)/vx0    
        # TODO: implement sim_fail check!
        sim_fail = False
        try:
            sim_phase2, t_phase2 = sim_until(IC,params,lambda x: x[4]) 
            t_phase2 += t_flight1
        except SimFailError:
            print 'simulation aborted (phase 2)\n'
            sim_fail = True
    
        #print "t_min:", t_phase2[-1]
        #print "y_min:", sim_phase2[-1,1]

    else:
        pars = array([k, L0, m, xF, zF], dtype=np.float64)
        #print "pars=", pars

        sim_fail=False
        N = _slipmdl1.odeOnce(buf, 2., dt=5e-3, pars=pars)
        #print "FS (landing): ", buf[N, [0,2,5]]
        #print "N=", N
        if N >= buffsize-1 or buf[N, 0] >= 2.:
            sim_fail = True
            print "simulation aborted (phase 2)"

        if not allclose(buf[N,5], 0):
            print "WARNING: nadir not found"

        sim_phase2 = buf[:N, 1:].copy()
        t_phase2 = buf[:N, 0].copy()

    # Phase 3:

    #return t_phase2, sim_phase2
    if not sim_fail:
        L = sqrt(sim_phase2[-1,1]**2 
                 + (sim_phase2[-1,0]-params['xF'])**2 
                 + (sim_phase2[-1,2]-params['zF'])**2 )
        
        
        if use_legacy: # skip original code
            dk, dL = dk_dL(L0,k,L,dE)
            params2 = deepcopy(params)
            params2['k'] += dk
            params2['L0'] += dL
            IC = sim_phase2[-1,:]
            compression = (lambda x: sqrt( 
                                          (x[0]-params2['xF'])**2 + x[1]**2
                                         +(x[2]-params['zF'])**2) 
                                          - params2['L0'] )
            #print ('L:', L, 'dk', dk, 'dL', dL, 'dE', dE, '\ncompression:', compression(IC),
            #      'IC', IC)
            try:                    
                sim_phase3, t_phase3 = sim_until(IC, params2,compression) 
                sim_phase3 = sim_phase3[1:,:]
                t_phase3 = t_phase3[1:] + t_phase2[-1]
            except SimFailError:
                print 'simulation aborted (phase 3)\n'
                sim_fail = True
        else:

            dk, dL = dk_dL(L0,k,L,dE)
            pars[0] += dk
            pars[1] += dL

            # new IC and start time is final state and time from previous
            # simulation step

            buf[0, :] = buf[N, :]
            buf[0, 5] = 0. # hard-code vertical velocity set to 0.

            N = _slipmdl2.odeOnce(buf, 2., dt=5e-3, pars=pars)
            if N >= buffsize-1 or buf[N, 0] >= 2.:
                sim_fail = True
                print "simulation aborted (phase 3)"
            else:
                sim_phase3 = buf[1:N, 1:]
                t_phase3 = buf[1:N, 0]
        
    # Phase 4:
    if not sim_fail:
        # time to apex
        # TODO: make sampling frequency regular
        vy_liftoff = sim_phase3[-1,4] 
        #vz_liftoff = sim_phase3[-1,5]
        t_flight2 = -1.*vy_liftoff/g
        #t_flight = sqrt()
        tvec_flight2 = arange(t_flight2,0,-.01)[::-1]
        vy_flight2 = tvec_flight2*g + vy_liftoff        
        y_flight2 = (sim_phase3[-1,1] + vy_liftoff*tvec_flight2 
                    + .5*g*(tvec_flight2**2) )
        x_flight2 = sim_phase3[-1,0] + sim_phase3[-1,3]*tvec_flight2
        vx_flight2 = sim_phase3[-1,3]*ones_like(tvec_flight2)
        z_flight2 = sim_phase3[-1,2] + sim_phase3[-1,5]*tvec_flight2
        vz_flight2 = sim_phase3[-1,5]*ones_like(tvec_flight2)
        #print tvec_flight2
        tvec_flight2 += t_phase3[-1]
        
    
    # todo: return data until error
    if sim_fail:
        return { 't': None, 
                'x': None,
                'y': None,
                'z': None,
                'vx': None,
                'vy': None,
                'vz': None,
                'sim_fail': sim_fail,
                'dk': None,
                'dL': None
                }
        
        
        
    # finally: concatenate phases    
    x_final = hstack([x_flight1, sim_phase2[:,0], sim_phase3[:,0], x_flight2 ])
    y_final = hstack([y_flight1, sim_phase2[:,1], sim_phase3[:,1], y_flight2 ])
    z_final = hstack([z_flight1, sim_phase2[:,2], sim_phase3[:,2], z_flight2 ])
    vx_final= hstack([vx_flight1, sim_phase2[:,3], sim_phase3[:,3], vx_flight2])
    vy_final= hstack([vy_flight1, sim_phase2[:,4], sim_phase3[:,4], vy_flight2])    
    vz_final= hstack([vz_flight1, sim_phase2[:,5], sim_phase3[:,5], vz_flight2])    
    tvec_final = hstack([tvec_flight1, t_phase2, t_phase3, tvec_flight2 ])
    
    
    return {'t': tvec_final, 
            'x': x_final,
            'y': y_final,
            'z': z_final,
            'vx': vx_final,
            'vy': vy_final,
            'vz': vz_final,
            'sim_fail': sim_fail,
            'dk': dk,
            'dL': dL,
            #'sim_res':sim_res,
            #'sim_phase2': sim_phase2_cut,
            #'t_phase2': t_phase2_cut
            }


def qSLIP_step3D(IC, SLIP_params):
    """
    similar to SLIP_step3D, except for interface changes

    :args:
        IC: initial state vector, containing y0, vx0, vz0
            (x0 is assumed to be 0;
             z0  is assumed to be 0; 
             vy0 = 0 (apex);
            also ground level = 0)
        SLIP_params(array or list):
            k
            L0
            m
            alpha : "original" angle of attack
            beta  : lateral leg turn
                    foot position relative to CoM in flight:
                        xF = vx0*t_flight + L0*cos(alpha)*cos(beta)
                        yF = -L0*sin(alpha)
                        zF = vz0*t_flight - L0*cos(alpha)*sin(beta)
            dE: energy change in "midstance" by changing k and L0

    :returns:
        sim_t, sim_states .  If simulation failed, returns None, None

    """

    buffsize = 8000 #size of simulation buffer

    k = SLIP_params[0]
    L0 = SLIP_params[1]
    m = SLIP_params[2]
    alpha = SLIP_params[3]
    beta = SLIP_params[4]
    dE = SLIP_params[5]
    g = -9.81
    

    y0 = IC[0]
    vx0 = IC[1]
    vz0 = IC[2]
    
    y_land = L0*sin(alpha)
    if y0 < y_land:
        raise ValueError, "invalid starting condition"
      
        
    # --- Section 1: time to touchdown  
    t_flight1 = sqrt(-2.*(y0 - y_land)/g) 

    tvec_flight1 = .01*arange(t_flight1*100.)
    vy_flight1 = tvec_flight1*g
    y_flight1 = y0 + .5*g*(tvec_flight1**2)
    x_flight1 = vx0*tvec_flight1
    vx_flight1 = vx0*ones_like(tvec_flight1)
    z_flight1 = vz0*tvec_flight1
    vz_flight1 = vz0*ones_like(tvec_flight1)
    x_TD = vx0*t_flight1
    z_TD = vz0*t_flight1

    # --- Section 2: time to vy = 0

    buf = zeros((buffsize, _slipmdl1.WIDTH), dtype=np.float64)
    IC = array([x_TD, y_land, z_TD, vx0, t_flight1*g, vz0])
    buf[0, 1:] = IC
    buf[0,0] = t_flight1
    
    # for the model in C
    xF = t_flight1*vx0 + L0*cos(alpha)*cos(beta)
    zF = t_flight1*vz0 - L0*cos(alpha)*sin(beta)
    #pars = array([k, L0, m, xF, zF], dtype=np.float64)
    pars = [k, L0, m, xF, zF]

    sim_fail=False
    N = _slipmdl1.odeOnce(buf, 2., dt=5e-3, pars=pars)
    if N >= buffsize or buf[N, 0] >= 2.:
        sim_fail = True
        print "simulation aborted (phase 2)"
    
    sim_phase2 = buf[:N, 1:].copy()
    t_phase2 = buf[:N, 0].copy()
    
    # --- Phase 3: simulation until liftoff

    if not sim_fail:
        L = sqrt(sim_phase2[-1,1]**2 
                 + (sim_phase2[-1,0]-xF)**2 
                 + (sim_phase2[-1,2]-zF)**2 )
        
        dk, dL = dk_dL(L0,k,L,dE)
        pars[0] += dk
        pars[1] += dL

        # new IC and start time is final state and time from previous
        # simulation step
        buf[0, :] = buf[N, :]
        buf[0, 5] = 0. # hard-code vertical velocity set to 0.

        N = _slipmdl2.odeOnce(buf, 2., dt=5e-3, pars=pars)
        if N >= buffsize or buf[N, 0] >= 2.:
            sim_fail = True
            print "simulation aborted (phase 3)"
        else:
            sim_phase3 = buf[1:N, 1:]
            t_phase3 = buf[1:N, 0]
        
        
        
    # --- Phase 4: compute until apex
    if not sim_fail:
        # time to apex
        # TODO: make sampling frequency regular
        vy_liftoff = sim_phase3[-1,4] 
        #vz_liftoff = sim_phase3[-1,5]
        t_flight2 = -1.*vy_liftoff/g
        #t_flight = sqrt()
        tvec_flight2 = arange(t_flight2,0,-.01)[::-1]
        vy_flight2 = tvec_flight2*g + vy_liftoff        
        y_flight2 = (sim_phase3[-1,1] + vy_liftoff*tvec_flight2 
                    + .5*g*(tvec_flight2**2) )
        x_flight2 = sim_phase3[-1,0] + sim_phase3[-1,3]*tvec_flight2
        vx_flight2 = sim_phase3[-1,3]*ones_like(tvec_flight2)
        z_flight2 = sim_phase3[-1,2] + sim_phase3[-1,5]*tvec_flight2
        vz_flight2 = sim_phase3[-1,5]*ones_like(tvec_flight2)
        #print tvec_flight2
        tvec_flight2 += t_phase3[-1]
        
    
    # todo: return data until error
    if sim_fail:
        return None, None
        
    # finally: concatenate phases    
    x_final = hstack([x_flight1, sim_phase2[:,0], sim_phase3[:,0], x_flight2 ])
    y_final = hstack([y_flight1, sim_phase2[:,1], sim_phase3[:,1], y_flight2 ])
    z_final = hstack([z_flight1, sim_phase2[:,2], sim_phase3[:,2], z_flight2 ])
    vx_final= hstack([vx_flight1, sim_phase2[:,3], sim_phase3[:,3], vx_flight2])
    vy_final= hstack([vy_flight1, sim_phase2[:,4], sim_phase3[:,4], vy_flight2])    
    vz_final= hstack([vz_flight1, sim_phase2[:,5], sim_phase3[:,5], vz_flight2])    
    tvec_final = hstack([tvec_flight1, t_phase2, t_phase3, tvec_flight2 ])

    return tvec_final, vstack([x_final, y_final, z_final, vx_final, vy_final,
        vz_final]).T
    
    
def km_SLIP_step3D(IC, SLIP_params):
    """
    similar to qSLIP_step3D, except for interface changes

    :args:
        IC: initial state vector, containing y0, vx0, vz0
            (x0 is assumed to be 0;
             z0  is assumed to be 0; 
             vy0 = 0 (apex);
            also ground level = 0)
        SLIP_params(array or list):
            k/m
            L0
            
            alpha : "original" angle of attack
            beta  : lateral leg turn
                    foot position relative to CoM in flight:
                        xF = vx0*t_flight + L0*cos(alpha)*cos(beta)
                        yF = -L0*sin(alpha)
                        zF = vz0*t_flight - L0*cos(alpha)*sin(beta)
            dE/m: energy change in "midstance" by changing k and L0

    :returns:
        sim_t, sim_states .  If simulation failed, returns None, None

    """

    buffsize = 8000 #size of simulation buffer

    k_m = SLIP_params[0]
    L0 = SLIP_params[1]
    #m = SLIP_params[2]
    alpha = SLIP_params[2]
    beta = SLIP_params[3]
    dE_m = SLIP_params[4]
    g = -9.81
    

    y0 = IC[0]
    vx0 = IC[1]
    vz0 = IC[2]
    
    y_land = L0*sin(alpha)
    if y0 < y_land:
        raise ValueError, "invalid starting condition"
      
        
    # --- Section 1: time to touchdown  
    t_flight1 = sqrt(-2.*(y0 - y_land)/g) 

    tvec_flight1 = .01*arange(t_flight1*100.)
    vy_flight1 = tvec_flight1*g
    y_flight1 = y0 + .5*g*(tvec_flight1**2)
    x_flight1 = vx0*tvec_flight1
    vx_flight1 = vx0*ones_like(tvec_flight1)
    z_flight1 = vz0*tvec_flight1
    vz_flight1 = vz0*ones_like(tvec_flight1)
    x_TD = vx0*t_flight1
    z_TD = vz0*t_flight1

    # --- Section 2: time to vy = 0

    buf = zeros((buffsize, _slipmdl1.WIDTH), dtype=np.float64)
    IC = array([x_TD, y_land, z_TD, vx0, t_flight1*g, vz0])
    buf[0, 1:] = IC
    buf[0,0] = t_flight1
    
    # for the model in C
    xF = t_flight1*vx0 + L0*cos(alpha)*cos(beta)
    zF = t_flight1*vz0 - L0*cos(alpha)*sin(beta)
    #pars = array([k, L0, m, xF, zF], dtype=np.float64)
    pars = [k_m, L0, 0, xF, zF]

    sim_fail=False
    N = _slipmdl1_km.odeOnce(buf, 2., dt=5e-3, pars=pars)
    if N >= buffsize or buf[N, 0] >= 2.:
        sim_fail = True
        print "simulation aborted (phase 2)"
    
    sim_phase2 = buf[:N, 1:].copy()
    t_phase2 = buf[:N, 0].copy()
    
    # --- Phase 3: simulation until liftoff

    if not sim_fail:
        L = sqrt(sim_phase2[-1,1]**2 
                 + (sim_phase2[-1,0]-xF)**2 
                 + (sim_phase2[-1,2]-zF)**2 )
        
        dk, dL = dk_dL(L0,k_m,L,dE_m)
        pars[0] += dk
        pars[1] += dL


        # new IC and start time is final state and time from previous
        # simulation step
        buf[0, :] = buf[N, :]
        buf[0, 5] = 0. # hard-code vertical velocity set to 0.

        N = _slipmdl2_km.odeOnce(buf, 2., dt=5e-3, pars=pars)
        if N >= buffsize or buf[N, 0] >= 2.:
            sim_fail = True
            print "simulation aborted (phase 3)"
        else:
            sim_phase3 = buf[1:N, 1:]
            t_phase3 = buf[1:N, 0]
        
        
        
    # --- Phase 4: compute until apex
    if not sim_fail:
        # time to apex
        # TODO: make sampling frequency regular
        vy_liftoff = sim_phase3[-1,4] 
        #vz_liftoff = sim_phase3[-1,5]
        t_flight2 = -1.*vy_liftoff/g
        #t_flight = sqrt()
        tvec_flight2 = arange(t_flight2,0,-.01)[::-1]
        vy_flight2 = tvec_flight2*g + vy_liftoff        
        y_flight2 = (sim_phase3[-1,1] + vy_liftoff*tvec_flight2 
                    + .5*g*(tvec_flight2**2) )
        x_flight2 = sim_phase3[-1,0] + sim_phase3[-1,3]*tvec_flight2
        vx_flight2 = sim_phase3[-1,3]*ones_like(tvec_flight2)
        z_flight2 = sim_phase3[-1,2] + sim_phase3[-1,5]*tvec_flight2
        vz_flight2 = sim_phase3[-1,5]*ones_like(tvec_flight2)
        #print tvec_flight2
        tvec_flight2 += t_phase3[-1]
        
    
    # todo: return data until error
    if sim_fail:
        return None, None
        
    # finally: concatenate phases    
    x_final = hstack([x_flight1, sim_phase2[:,0], sim_phase3[:,0], x_flight2 ])
    y_final = hstack([y_flight1, sim_phase2[:,1], sim_phase3[:,1], y_flight2 ])
    z_final = hstack([z_flight1, sim_phase2[:,2], sim_phase3[:,2], z_flight2 ])
    vx_final= hstack([vx_flight1, sim_phase2[:,3], sim_phase3[:,3], vx_flight2])
    vy_final= hstack([vy_flight1, sim_phase2[:,4], sim_phase3[:,4], vy_flight2])    
    vz_final= hstack([vz_flight1, sim_phase2[:,5], sim_phase3[:,5], vz_flight2])    
    tvec_final = hstack([tvec_flight1, t_phase2, t_phase3, tvec_flight2 ])

    return tvec_final, vstack([x_final, y_final, z_final, vx_final, vy_final,
        vz_final]).T
