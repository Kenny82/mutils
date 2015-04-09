# -*- coding: utf-8 -*-
"""
@file

@brief This file provides a very basic implementation of a Levant
differentiator.

@author Moritz Maus, h.maus@imperial.ac.uk

@date 12 March 2015

"""

from numpy import sqrt, sign, linspace

class Levant(object):
    """
    A differentiator class which does step-by-step differentiation / estimation.
    """
    def __init__(self, c_alpha, c_lambda, dt=0.001, x0 = 0, u10=0):
        self.c_alpha = c_alpha
        self.c_lambda = c_lambda
        self.dt = dt
        self.x = x0
        self.u1 = u10
        self.last_y = None
        
    def _sat(self, val, limit):
        if val < -limit:
            return -limit
        elif val > limit:
            return limit
        else:
            return val
        
        
    def digest(self, y, dt=None, return_sig=False):
        """
        process a single measurement
        """
        if dt:
            dt_ = dt
        else:
            dt_ = self.dt
        
        if self.last_y == None:
            self.last_y = y          
                
        # this would be the basic version:
        if False:
            u = self.u1 - self.c_lambda * sqrt(abs(self.x - y)) * sign(self.x - y)
            u1_dot = -1.0 * self.c_alpha * sign(self.x - y)        
            self.u1 = self.u1 + u1_dot * dt_
            self.x = self.x + u * dt_
            return u, self.x, self.u1
        
        # version with refinement:
        n_refine = 7
        last_x = self.x
        seg = linspace(self.last_y, y, n_refine)
        for ys in seg:           

            u1_dot = -self.c_alpha * sign(self.x - ys) # limit this later
            #if abs(self.x - ys) < abs(self.u1 * dt_ / n_refine):
            #    u1_dot = 0
            self.u1 = self.u1 + u1_dot * dt_ / float(n_refine)            
            u = self.u1 - self.c_lambda * sqrt(abs(self.x - ys)) * sign(self.x - ys)       
            self.x = self.x + u * dt_ / float(n_refine)
        
        u = (self.x - last_x) / dt_
        self.last_y = y
        
        if return_sig:
            return u, self.x, self.u1
        else:
            return u
   

def l_der(sig, dt, c_alpha, c_lambda, return_sig = False):
    """
    A convenience function to get the estimated derivative from a signal
    (offline processing, Levant itself is intended for online use).

    @param sig (1d array) the signal to compute the derivative from
    @param dt (float) time increment at every sample 
    @param c_alpha (float) Levant's alpha constant (how fast the derivative may change)
    @param c_lambda (float) Levant's lamba constant (additional tracking of the signal)
    @param return_sig (bool) if true, returns derivative, signal,
            levant_u1_signal, otherwise only derivative

    @return Return value (1 or 3 arrays) depend on parameter return_sig

    """
    d = Levant(c_alpha, c_lambda, dt)
    
    x_res = 0 * sig    
    d_res = 0 * sig
    u1_res = 0 * sig
    for idx, elem in enumerate(sig):
        dd, xx, u1 = d.digest(elem, return_sig=True)
        x_res[idx] = xx
        d_res[idx] = dd
        u1_res[idx] = u1
    
        
    if return_sig:
        return d_res, x_res, u1_res
    else:
        return d_res
