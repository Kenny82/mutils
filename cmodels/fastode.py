#!/usr/bin/python

#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Shai Revzen, U Penn, 2010

import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import os

"""
fastode is a python ctypes interface to an ODE integrator designed for use with
hybrid systems. It is based on the dopri5 integrator code of 
         E. Hairer & G. Wanner
         Universite de Geneve, dept. de Mathematiques
         CH-1211 GENEVE 4, SWITZERLAND
         E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH
 
 The code is described in : E. Hairer, S.P. Norsett and G. Wanner, Solving
 ordinary differential equations I, nonstiff problems, 2nd edition,
 Springer Series in Computational Mathematics, Springer-Verlag (1993).

In addition to the original integrator, the code sports a versatile event 
detector, and refines the final point to the positive zero crossing of the
terminating event.

The ODE must be defined in a .c file, which defines the macros listed below
using #define. The integrator stores the results in slice of a pre-allocated
array, using the initial row in the slice as the initial condition. This allows
the code to quickly continue after detecting events and returning them to 
the python main program. Auxilary outputs can be generated and stored in the
output array as well.

Typical usage is:

>>> ode = fastode.FastODE('circle')
>>> y = zeros((1000,ode.WIDTH),dtype=float64)
>>> y[0,:] = (0,1,0) # initial condition
>>> N = ode.odeOnce( y, 100 ) # integrate until time == 100
>>> plot( y[:N,:] ) # plot the results

1. Structural constants:
  ODE_DIM -- dimension of the system (>0)
  ODE_AUX -- number of auxiliary values returned with each timestep (>=0)
  ODE_NPAR -- number of parameters (>=0)
  ODE_MAX_EVENTS -- number of possible termination events

2. Functions:
  ODE_FUNC(n,t,s,p,d) -- ODE \dot s = functionof( t, s, params )
    n -- int -- dimension (==ODE_DIM)
    t -- double -- time
    s -- double[ODE_DIM] -- state s (read only)
    p -- double[ODE_NPAR] -- parameters (read only)
    d -- double[ODE_DIM] -- derivative \dot s (write only)
    
  ODE_AUXFUNC(n,t,s,p,a) -- auxilary outputs from state
    n -- int -- dimension (==ODE_DIM)
    t -- double -- time
    s -- double[ODE_DIM] -- state s (read only)
    p -- double[ODE_NPAR] -- parameters (read only)
    a -- double[ODE_AUX] -- auxilary outputs (write only)

  ODE_SCAN_EVENTS(t,s,n,p,v) -- event detector functions
    t -- double -- time
    s -- double[ODE_DIM] -- state (read only)
    n -- int -- dimension (==ODE_DIM)
    p -- double[ODE_NPAR] -- parameters (read only)
    v -- double[ODE_MAX_EVENTS] -- value of event functions

3. Numerical algorithm parameters:
  ODE_DT -- double -- maximal timestep
  ODE_MAX -- double -- upper bound on all elements of state
    
"""
def depsChanged( tgt, *deps ):
  """check whether any of the specified "dependencies" of the "target"
  have mtime (modification time) larger than the target ctime (creation
  time). Target is also marked as changed if it is not found.
  
  Raises an OSError if any of the dependencies are missing.
  
  INPUT:
    tgt -- filename
    *deps -- multiple filenames as strings
    
  OUTPUT:
    boolean
  
  Shai Revzen, U. Penn., 2010
  """
  try:
    t = os.lstat(tgt).st_ctime
  except OSError:
    return True
  for dep in deps:
    try:
      if os.lstat(dep).st_mtime > t:
        return True
    except OSError, ose:
      raise OSError("Could not stat dependency '%s'" % dep )
  return False


class FastODE(object):
  """
  FastODE is a ctypes wrapper for a fast ODE integrator. The actual flow
  function and event detectors should be written in C.
  
  Each xxx.c ODE is compiled and linked with the integrator code with
  optimization switched on, to create a _xxx.so.
  
  If necessary, FastODE will recompile the _xxx.so binary before loading
  it. It will also ensure that the dimensions of the arrays used match those
  specified in the ODE_xxx macros in the payload xxx.c file.
  """
  def __init__( self, cfile, path='.', cpath='.' ):
    """Create a FastODE interface to an ODE defined in a c file
    INPUTS:
       cfile -- string -- c filename (trailing '.c' optional)
       path -- string -- path to where the .so should be placed. Must be 
         a writable directory in your sys.path (this isn't checked)
       cpath -- string -- path to where the .c file is found       
    """
    if cfile[-2:]=='.c':
      cfile = cfile[:-2]
    while path[-1]=='/':
      path = path[:-1]  
    while cpath[-1]=='/':
      cpath = cpath[:-1]  
    sofn = '%s/_%s.so' % (path,cfile)
    cfn = '%s/%s.c' % (cpath,cfile)
    #
    # Rebuild the shared library as needed
    #
    if depsChanged(sofn,'fastode.c','integro.c',cfn):
      print "==> Recompiling",sofn,"from",cfn
      if os.system('rm -f ode_payload.c; ln -s %s ode_payload.c' % cfn ):
        raise RuntimeError("Failed to create symlink 'ode_payload.c' to '%s'" % cfn )
      if os.system(
      'gcc -fPIC -g -O3 fastode.c -lm -shared -o %s' % sofn
      ):
        raise RuntimeError('Compilation of %s failed' % sofn )
    #
    # Load the .so into ctypes
    #
    code = ctypes.cdll[sofn]
    #
    ## Set up typechecking for the main functions
    #
    #  long odeOnce( double *yOut, int nmax, int yOut_dim, int startRow,
    #         double tEnd,  double dt,  double *pars,  int pars_dim  )
    code.odeOnce.argtypes = [ 
      ndpointer(dtype=np.double,ndim=2,flags="C_CONTIGUOUS"),
      ctypes.c_int,ctypes.c_int,ctypes.c_int, 
      ctypes.c_double,ctypes.c_double,
      ndpointer(dtype=np.double,flags="C"),ctypes.c_int]
    #
    # int odeComputeAux( double *y0, int y0_dim, double *pars, int pars_dim )
    #  
    code.odeComputeAux.argtypes = [ 
      ndpointer(dtype=np.double,flags="C_CONTIGUOUS"), ctypes.c_int,
      ndpointer(dtype=np.double,flags="C"),ctypes.c_int]
    #
    # int odeEventId(double *y0, int y0_dim) 
    #
    code.odeEventId.argtypes = [
      ndpointer(dtype=np.double,flags="C"),ctypes.c_int ]
    #
    ## Read ODE properties from global vars in the .so
    #
    # int variables
    #
    for nm in ['max_events','dim','width','aux','npar']:
      val = ctypes.c_int.in_dll(code,"ode_%s" % nm).value
      setattr( self, nm.upper(), val )
    #
    # float variables
    #
    for nm in ['atol','rtol','evttol','max']:
      val = ctypes.c_double.in_dll(code,"ode_%s" % nm).value
      setattr( self, nm.upper(), val )
    #
    self.code = code
    self.so_filename = sofn
    self.c_filename = cfn
  #
  def odeOnce( self, dat, tEnd, dt = 0.1,slc = slice(-1), pars = None ):
    """
    Run ode integration once, filling in results in a pre-allocated output
    array. Execution terminates when time is tEnd, space runs out, or an event
    causes itermination.
    
    INPUTS:
      dat -- N x self.WIDTH of float64 -- output array, with initial values
        in the first row of the output slice
      tEnd -- number -- end time
      dt -- number -- maximal timestep
      slc -- slice -- slice of dat rows to use for output; default is all
      pars -- self.NPAR of float64 -- array of parameters for integrator
    
    OUTPUT:
      rows -- integer -- number of rows used for output, including initial cond.
    """
    if not isinstance(dat,np.ndarray) or dat.dtype != np.float64:
      raise TypeError("dat must be a N x %d (contiguous, C layout) numpy array of type float64" % self.WIDTH)
    start,stop,step = slc.indices(dat.shape[0])
    tEnd = float(tEnd)
    dt = float(dt)
    if dat.shape[1] != self.WIDTH:
      raise ValueError("Expected %d and not %d columns" % (self.WIDTH, dat.shape[1]))
    if pars is None:
      if self.NPAR>0:
        raise TypeError("Expected to get %d parameters; got None" % self.NPAR)
      pars = np.array([],dtype=np.float64)
    else:
      pars = np.asarray(pars,dtype=np.float64)
    if pars.size <> self.NPAR:
      raise TypeError("Expected to get %d parameters; got %d" % (self.NPAR, pars.size))
    #  long odeOnce( double *yOut, int nmax, int yOut_dim, int startRow,
    #         double tEnd,  double dt,  double *pars,  int pars_dim  )
    N = self.code.odeOnce( dat, stop, dat.shape[1], start, 
      tEnd, dt, pars, pars.size )
    return N
  #  
  def odeComputeAux( self, dat,  pars = None ):
    """
    Compute auxillary outputs, filling in results in a pre-allocated output row
    
    INPUTS:
      dat -- N x self.WIDTH of float64 -- data array
      pars -- self.NPAR of float64 -- array of parameters for integrator
    """
    if not isinstance(dat,np.ndarray) or dat.dtype != np.float64:
      raise TypeError("dat must be a %d (contiguous, C layout) numpy array of type float64" % self.WIDTH)
    if dat.size < self.WIDTH:
      raise ValueError("Expected size of %d ; got %d" % (self.WIDTH, dat.size))
    if pars is None and self.NPAR>0:
      raise TypeError("Expected to get %d parameters; got None" % self.NPAR)
    pars = np.asarray(pars,dtype=np.float64)
    if pars.size <> self.NPAR:
      raise TypeError("Expected to get %d parameters; got %d" % (self.NPAR, pars.size))
    N = self.code.odeComputeAux( dat, self.WIDTH, pars, pars.size )
    if N:
      raise RuntimeError("odeComputeAux returned error code %d" % N)
  # 
  def odeEventId( self, evt ):
    """
    Find event ID of event that terminated integration, and compute all 
    transitition functions
    
    INPUTS:
      evt -- N x self.MAX_EVENTS of float64 -- output array
      
    OUTPUT:
      evtId -- integer

    NOTE:
      the values in evt[evtId:] are not valid; only evt[:evtId] are computed 
    """
    if not isinstance(evt,np.ndarray) or evt.dtype != np.float64:
      raise TypeError("evt must be a %d long (contiguous, C layout) numpy array of type float64" % self.MAX_EVENTS)
    if len(evt) < self.MAX_EVENTS:
      raise ValueError("Expected length of %d, got %d" % (self.MAX_EVENTS, dat.shape[1]))
    N = self.code.odeEventId( evt, self.MAX_EVENTS )
    return N



