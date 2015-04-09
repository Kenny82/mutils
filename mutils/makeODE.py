#!/usr/bin/python
"""
#  This module is actually a hack to provide a c-written ODE as python library.
#  Here, the C-code is passed as string!
#
#  useage: fastODEobject = makeODE(<c-code string>, modulename, skiphash=False,
#           verbose=False)
#
#  examples are presented in strings:
#  <makeODE>.ex_py_code -> python code to run the object
#  <makeODE>.ex_c_code -> ODE definition in C
#
#  example:
#  >>>import mutils.makeODE as mo    
#  >>>exec mo.ex_py_code
#  for an explanation, read ">>>print mo.ex_py_code "
#
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
# (c) Moritz Maus, TU Darmstadt (Germany), 2013 (module packaging)
"""

import os
import tempfile
import hashlib
import ctypes

from numpy.ctypeslib import ndpointer
import numpy as np


def fromfile(fname):
    """
    Convenience function.
    Returns a string that holds the file.
    """
    return ''.join(open(fname, 'r').readlines())

class FastODE(object):
  """
  
  NOTE: the documentation is a little bit outdated since I added an additional
  wrapper. (Moritz)

  NEW: use makeODE(str, modname) instead

  FastODE is a ctypes wrapper for a fast ODE integrator. The actual flow
  function and event detectors should be written in C.
  
  Each xxx.c ODE is compiled and linked with the integrator code with
  optimization switched on, to create a _xxx.so.
  
  If necessary, FastODE will recompile the _xxx.so binary before loading
  it. It will also ensure that the dimensions of the arrays used match those
  specified in the ODE_xxx macros in the payload xxx.c file.

  """

  def __init__( self, sofilename):
    """Create a FastODE interface to an ODE defined in a c file
        sofilename (str): name of the so module to be loaded (without .so)

    """

    if sofilename[-3:]=='.so':
        sofn = './' + sofilename
    else:
        sofn = './' + sofilename + '.so'
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
    #self.c_filename = cfn
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







def makeODE(codestr, modname, skiphash=False, verbose=False):
    """
    Takes a string with C-code describing the ODE. Compiles it with the
    integrator to a module called modname.

    :args:
        codestr (str): C-code of the ODE
        modname (str): the name of the module (without .so)
        skiphash (bool): if True, do not compute and compare hash values.
        verbose (bool): print additional messages
    """

    topcode = r"""

/*** begin fastode.c ***/

#include <assert.h>
#include <math.h>


/* #include "fastode.h" */
/*** begin fastode.h ***/

#define ODE_NO_EVENT ~0

int odeEventId( double *y0, int y0_dim);

void solout( long nr, double xold, double x, double* y, unsigned n, int* irtrn);

int odeComputeAux( double *y0, int y0_dim, double *pars, int pars_dim);

long odeOnce( double *yOut, int nmax, int yOut_dim, int startRow, double tEnd, double dt, double *pars, int pars_dim);

#define ODE_FUNC rhs
#define ODE_SCAN_EVENTS event
#define ODE_MAX_EVENTS 1



/*** end fastode.h ***/



/* Forward declaration for ode function wrapper */
inline static void safeOdeFunc( 
  unsigned n, double t, double *state, double *par, double *deriv
);

inline static void alwaysCheckEvt( int evid );


/* #include "ode_payload.c" */ /* source include */

    """
    
    bottomcode = r"""

#ifndef ODE_DT
#define ODE_DT 1e-3
#endif

#ifndef ODE_MAX
#define ODE_MAX 1e9
#endif

#ifndef ODE_RTOL
#define ODE_RTOL 1e-9
#endif

#ifndef ODE_ATOL
#define ODE_ATOL 1e-6
#endif

#define ODE_AUX 0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <memory.h>
#include <string.h>


/* #include "integro.h" */

/*** begin integro.h ***/


/*  INTEGRO header file
	------

Mex code based on DOPRI5 to compute the numerical solution of a first order ODE 
y'=f(x,y).  DOPRI5 implements an explicit Runge-Kutta method of order (4)5 due to 
Dormand & Prince, with step size control and dense output.  INTEGRO adds 
event detection and localization features. 

The resulting mex function is intended to be called by the Advance() method 
of the INTEGRO object.
 
The dopri5 code is described in : E. Hairer, S.P. Norsett and G. Wanner, Solving
ordinary differential equations I, nonstiff problems, 2nd edition,
Springer Series in Computational Mathematics, Springer-Verlag (1993).

INTEGRO uses the version of April 28, 1994, of DOPRI5.

INPUT PARAMETERS
----------------

n        Dimension of the system (n < UINT_MAX).

fcn      A pointer the the function definig the differential equation, this
	 function must have the following prototype

	   void fcn (unsigned n, double x, double *y, double *p, double *f)

	 where the array f will be filled with the function result.

x        Initial x value.

*y       Initial y values (double y[n]).

xend     Final x value (xend-x may be positive or negative).

*rtoler  Relative and absolute error tolerances. They can be both scalars or
*atoler  vectors of length n (in the scalar case pass the addresses of
	 variables where you have placed the tolerance values).

itoler   Switch for atoler and rtoler :
	   itoler=0 : both atoler and rtoler are scalars, the code keeps
		      roughly the local error of y[i] below
		      rtoler*abs(y[i])+atoler.
	   itoler=1 : both rtoler and atoler are vectors, the code keeps
		      the local error of y[i] below
		      rtoler[i]*abs(y[i])+atoler[i].

solout   A pointer to the output function called during integration.
	 If iout >= 1, it is called after every successful step. If iout = 0,
	 pass a pointer equal to NULL. solout must must have the following
	 prototype

	   solout (long nr, double xold, double x, double* y, unsigned n, int* irtrn)

	 where y is the solution the at nr-th grid point x, xold is the
	 previous grid point and irtrn serves to interrupt the integration
	 (if set to a negative value).

	 Continuous output : during the calls to solout, a continuous solution
	 for the interval (xold,x) is available through the function

	   contd5(i,s)

	 which provides an approximation to the i-th component of the solution
	 at the point s (s must lie in the interval (xold,x)).

iout     Switch for calling solout :
	   iout=0 : no call,
	   iout=1 : solout only used for output,
	   iout=2 : dense output is performed in solout (in this case nrdens
		    must be greater than 0).

fileout  A pointer to the stream used for messages, if you do not want any
	 message, just pass NULL.

icont    An array containing the indexes of components for which dense
	 output is required. If no dense output is required, pass NULL.

licont   The number of cells in icont.


Setting of parameters
--------------------- 

Parameters setting is expected to be done within matlab.  We list here
the comments from the DOPRI5.h file for reference.

	 Several parameters have a default value (if set to 0) but, to better
	 adapt the code to your problem, you can specify particular initial
	 values.

uround   The rounding unit, default 2.3E-16 (this default value can be
	 replaced in the code by DBL_EPSILON providing float.h defines it
	 in your system).

safe     Safety factor in the step size prediction, default 0.9.

fac1     Parameters for step size selection; the new step size is chosen
fac2     subject to the restriction  fac1 <= hnew/hold <= fac2.
	 Default values are fac1=0.2 and fac2=10.0.

beta     The "beta" for stabilized step size control (see section IV.2 of our
	 book). Larger values for beta ( <= 0.1 ) make the step size control
	 more stable. dopri5 needs a larger beta than Higham & Hall. Negative
	 initial value provoke beta=0; default beta=0.04.

hmax     Maximal step size, default xend-x.

h        Initial step size, default is a guess computed by the function hinit.

nmax     Maximal number of allowed steps, default 100000.

meth     Switch for the choice of the method coefficients; at the moment the
	 only possibility and default value are 1.

nstiff   Test for stiffness is activated when the current step number is a
	 multiple of nstiff. A negative value means no test and the default
	 is 1000.

nrdens   Number of components for which dense outpout is required, default 0.
	 For 0 < nrdens < n, the components have to be specified in icont[0],
	 icont[1], ... icont[nrdens-1]. Note that if nrdens=0 or nrdens=n, no
	 icont is needed, pass NULL.



OUTPUT PARAMETERS
-----------------

y       numerical solution at x=xRead() (see below).

integro() returns the following values

	 4 : computation successful, interrupted by user-defined event
	 3 : computation successful interrupted by coordinate event.
	 2 : computation successful interrupted by solout,
	 1 : computation successful,
	-1 : input is not consistent,
	-2 : larger nmax is needed,
	-3 : step size becomes too small,
	-4 : the problem is probably stff (interrupted).


Several functions provide access to different values :

xRead   x value for which the solution has been computed (x=xend after
	successful return).

hRead   Predicted step size of the last accepted step (useful for a
	subsequent call to integration code).

nstepRead   Number of used steps.
naccptRead  Number of accepted steps.
nrejctRead  Number of rejected steps.
nfcnRead    Number of function calls.


*/


#include <stdio.h>
#include <limits.h>

typedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *p, double *f);

typedef void (*SolTrait)(long nr, double xold, double x, double* y, unsigned n, int* irtrn);

typedef double (*EvFunType)(unsigned n, double x, double *y, double *p);

extern int dopri5
 (unsigned n,      /* dimension of the system <= UINT_MAX-1*/
  double x,        /* initial x-value */
  double* y,       /* initial values for y */
  double *pars,    /* parameters */
  double xend,     /* final x-value (xend-x may be positive or negative) */
  double* rtoler,  /* relative error tolerance */
  double* atoler,  /* absolute error tolerance */
  int itoler,      /* switch for rtoler and atoler */
  int iout,        /* switch for calling solout */
  FILE* fileout,   /* messages stream */
  double uround,   /* rounding unit */
  double safe,     /* safety factor */
  double fac1,     /* parameters for step size selection */
  double fac2,
  double beta,     /* for stabilized step size control */
  double hmax,     /* maximal step size */
  double h,        /* initial step size */
  long nmax,       /* maximal number of allowed steps */
  double magboud,   /* bound on magnitude of solution components */
  int meth,        /* switch for the choice of the coefficients */
  long nstiff,     /* test for stiffness */
  unsigned nrdens, /* number of components for which dense outpout is required */
  unsigned* icont, /* indexes of components for which dense output is required, >= nrdens */
  unsigned licont  /* declared length of icont */
 );

extern double contd5
 (unsigned ii,     /* index of desired component */
  double x         /* approximation at x */
 );

extern long nfcnRead (void);   /* encapsulation of statistical data */
extern long nstepRead (void);
extern long naccptRead (void);
extern long nrejctRead (void);
extern double hRead (void);
extern double xRead (void);

/*** end integro.h ***/

#define DBUG 0
#define VERBOSE 0
#define DENSE 1

static long nfcn, nstep, naccpt, nrejct;
static double hout, xold, xout;
static unsigned nrds, *indir;
static double *yy1, *k1, *k2, *k3, *k4, *k5, *k6, *ysti;
static double *rcont1, *rcont2, *rcont3, *rcont4, *rcont5;

long nfcnRead (void)
{
  return nfcn;
}                               /* nfcnRead */

long nstepRead (void)
{
  return nstep;
}                               /* stepRead */

long naccptRead (void)
{
  return naccpt;
}                               /* naccptRead */

long nrejctRead (void)
{
  return nrejct;
}                               /* nrejct */

double hRead (void)
{
  return hout;
}                               /* hRead */

double xRead (void)
{
  return xout;
}                               /* xRead */
static double sign (double a, double b)
{
  return (b > 0.0) ? fabs (a) : -fabs (a);
}                               /* sign */
static double min_d (double a, double b)
{
  return (a < b) ? a : b;
}                               /* min_d */
static double max_d (double a, double b)
{
  return (a > b) ? a : b;
}                               /* max_d *//* Step-size initialization routine */
static double hinit (unsigned n, double x, double *y,
                    double *pars, double posneg, double *f0, double *f1,
                    double *yy1, int iord, double hmax, double *atoler,
                    double *rtoler, int itoler)
{
  double dnf, dny, atoli, rtoli, sk, h, h1, der2, der12, sqr;
  unsigned i;

  dnf = 0.0;
  dny = 0.0;
  atoli = atoler[0];
  rtoli = rtoler[0];
  if (!itoler)
    for (i = 0; i < n; i++) {
      sk = atoli + rtoli * fabs (y[i]);
      sqr = f0[i] / sk;
      dnf += sqr * sqr;
      sqr = y[i] / sk;
      dny += sqr * sqr;
  } else
    for (i = 0; i < n; i++) {
      sk = atoler[i] + rtoler[i] * fabs (y[i]);
      sqr = f0[i] / sk;
      dnf += sqr * sqr;
      sqr = y[i] / sk;
      dny += sqr * sqr;
    }
  if ((dnf <= 1.0E-10) || (dny <= 1.0E-10))
    h = 1.0E-6;
  else
    h = sqrt (dny / dnf) * 0.01;
  h = min_d (h, hmax);
  h = sign (h, posneg);         /* perform an explicit Euler step */
  for (i = 0; i < n; i++)
    yy1[i] = y[i] + h * f0[i];
  safeOdeFunc (n, x + h, yy1, pars, f1);  /* estimate the second derivative of the solution */
  der2 = 0.0;
  if (!itoler)
    for (i = 0; i < n; i++) {
      sk = atoli + rtoli * fabs (y[i]);
      sqr = (f1[i] - f0[i]) / sk;
      der2 += sqr * sqr;
  } else
    for (i = 0; i < n; i++) {
      sk = atoler[i] + rtoler[i] * fabs (y[i]);
      sqr = (f1[i] - f0[i]) / sk;
      der2 += sqr * sqr;
    }
  der2 = sqrt (der2) / h;       /* step size is computed such that h**iord * max_d(norm(f0),norm(der2)) = 0.01 */
  der12 = max_d (fabs (der2), sqrt (dnf));
  if (der12 <= 1.0E-15)
    h1 = max_d (1.0E-6, fabs (h) * 1.0E-3);
  else
    h1 = pow (0.01 / der12, 1.0 / (double) iord);
  h = min_d (100.0 * h, min_d (h1, hmax));
  
  if (isnan(h)) { __asm__("int $3"); };  
  return sign (h, posneg);
}                               /* hinit */

/* core integrator */
static int dopcor (unsigned n, double x, double *y,
                  double *pars, double xend, double hmax, double h,
                  double *rtoler, double *atoler, int itoler, FILE * fileout,
                  int iout, long nmax, double magbound,
                  double uround, int meth, long nstiff, double safe,
                  double beta, double fac1, double fac2, unsigned *icont)
{
  double facold, expo1, fac, facc1, facc2, fac11, posneg, xph;
  double atoli, rtoli, hlamb, err, sk, hnew, yd0, ydiff, bspl;
  double stnum, stden, sqr;
  int iasti, iord, irtrn, reject, last, nonsti;
  unsigned i, j, loopcnt = 0;
  double c2, c3, c4, c5, e1, e3, e4, e5, e6, e7, d1, d3, d4, d5, d6, d7;
  double a21, a31, a32, a41, a42, a43, a51, a52, a53, a54;
  double a61, a62, a63, a64, a65, a71, a73, a74, a75, a76;  /* vars for event detection */
  double dx, jdx;               /* initializations */

  if (isnan(x) || isnan(h)) { __asm__("int $3"); };

  c2 = 0.2; c3 = 0.3; c4 = 0.8; c5 = 8.0 / 9.0;
  
  a21 = 0.2; 
  
  a31 = 3.0 / 40.0; a32 = 9.0 / 40.0;
  
  a41 = 44.0 / 45.0; a42 = -56.0 / 15.0;  a43 = 32.0 / 9.0;
  
  a51 = 19372.0 / 6561.0; a52 = -25360.0 / 2187.0; 
  a53 = 64448.0 / 6561.0; a54 = -212.0 / 729.0;
  
  a61 = 9017.0 / 3168.0; a62 = -355.0 / 33.0; 
  a63 = 46732.0 / 5247.0; a64 = 49.0 / 176.0; a65 = -5103.0 / 18656.0;
  
  a71 = 35.0 / 384.0; 
  a73 = 500.0 / 1113.0; a74 = 125.0 / 192.0; a75 = -2187.0 / 6784.0; 
  a76 = 11.0 / 84.0;
  
  e1 = 71.0 / 57600.0; e3 = -71.0 / 16695.0; e4 = 71.0 / 1920.0;
  e5 = -17253.0 / 339200.0; e6 = 22.0 / 525.0; e7 = -1.0 / 40.0;
  
  d1 = -12715105075.0 / 11282082432.0; d3 = 87487479700.0 / 32700410799.0;
  d4 = -10690763975.0 / 1880347072.0; d5 = 701980252875.0 / 199316789632.0;
  d6 = -1453857185.0 / 822651844.0; d7 = 69997945.0 / 29380423.0;
  
  facold = 1.0E-4;
  expo1 = 0.2 - beta * 0.75;
  facc1 = 1.0 / fac1;
  facc2 = 1.0 / fac2;
  posneg = sign (1.0, xend - x);
  atoli = atoler[0];
  rtoli = rtoler[0];
  last = 0;
  hlamb = 0.0;
  iasti = 0;
  safeOdeFunc (n, x, y, pars, k1);
  hmax = fabs (hmax);
  iord = 5;
  
  if (DBUG)
    fprintf (stderr, "dopri: initialization done, iout=%d\n", iout);
  if (h == 0.0) {
    h =
      hinit (n, x, y, pars, posneg, k1, k2, k3, iord, hmax, atoler,
             rtoler, itoler);
    if (DBUG) {
      fprintf (stderr, "dopri: hinit() done.\n");
      fflush (stderr);
    }
  }
  nfcn += 2;
  reject = 0;
  xold = x;
  if (iout) {
    irtrn = 1;
    hout = h;
    xout = x;
    solout (naccpt + 1, xold, x, y, n, &irtrn);
    if (irtrn < 0) {
      if (fileout)
        fprintf (fileout, "Exit on initial point\r\n", x);
      return 2;
    }
  }                             /* integration loop */
  if (VERBOSE) {
    fprintf (stderr, "Integrating...\n");
    fflush (stderr );
  };
  if (isnan(h)) { __asm__("int $3"); };
  while (1) {
    if (isnan(h)) { __asm__("int $3"); };
    if (nstep > nmax) {
      xout = x;
      hout = h;
      return -2;
    }
    if (0.1 * fabs (h) <= fabs (x) * uround) {
      if (fileout)
        fprintf (fileout, "Integro (mex): Step size too small h = %.16e\r\n",
                 x, h);
      xout = x;
      hout = h;
      return -3;
    }
    for (i = 0; i < n; i++)
      if (fabs (y[i]) > magbound) {
        fprintf (fileout,
                 "Integro (mex): Solution exceeds bound on component's magnitude  ('magbound'=%f)\n",
                 magbound);
        xout = x;
        hout = h;
        return -5;
      }
    if ((x + 1.01 * h - xend) * posneg > 0.0) {
      h = xend - x;
      last = 1;
    }
    nstep++;
    if (DBUG) {
      fprintf (stderr, "%ld\n", nstep);
      fflush (stderr);
    }                           /* the first 6 stages */
    if (isnan(h)) { __asm__("int $3"); };
    
    for (i = 0; i < n; i++)
      yy1[i] = y[i] + h * a21 * k1[i];
    safeOdeFunc (n, x + c2 * h, yy1, pars, k2);
    for (i = 0; i < n; i++)
      yy1[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    safeOdeFunc (n, x + c3 * h, yy1, pars, k3);
    for (i = 0; i < n; i++)
      yy1[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    safeOdeFunc (n, x + c4 * h, yy1, pars, k4);
    for (i = 0; i < n; i++)
      yy1[i] =
        y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    safeOdeFunc (n, x + c5 * h, yy1, pars, k5);
    for (i = 0; i < n; i++)
      ysti[i] =
        y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] +
                    a65 * k5[i]);
    xph = x + h;
    safeOdeFunc (n, xph, ysti, pars, k6);
    for (i = 0; i < n; i++)
      yy1[i] =
        y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] +
                    a76 * k6[i]);
    safeOdeFunc (n, xph, yy1, pars, k2);
    if (DENSE)
      if (nrds == n)
        for (i = 0; i < n; i++) {
          rcont5[i] =
            h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] +
                 d6 * k6[i] + d7 * k2[i]);
      } else
        for (j = 0; j < nrds; j++) {
          i = icont[j];
          rcont5[j] =
            h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] +
                 d6 * k6[i] + d7 * k2[i]);
        }
    for (i = 0; i < n; i++)
      k4[i] =
        h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] +
             e7 * k2[i]);
    nfcn += 6;                  /* error estimation */
    err = 0.0;
    if (!itoler)
      for (i = 0; i < n; i++) {
        sk = atoli + rtoli * max_d (fabs (y[i]), fabs (yy1[i]));
        sqr = k4[i] / sk;
        err += sqr * sqr;
    } else
      for (i = 0; i < n; i++) {
        sk = atoler[i] + rtoler[i] * max_d (fabs (y[i]), fabs (yy1[i]));
        sqr = k4[i] / sk;
        err += sqr * sqr;
      }
    err = sqrt (err / (double) n);  /* computation of hnew */
    fac11 = pow (err, expo1);   /* Lund-stabilization */
    fac = fac11 / pow (facold, beta); /* we require fac1 <= hnew/h <= fac2 */
    fac = max_d (facc2, min_d (facc1, fac / safe));
    hnew = h / fac;
    if (isnan(h) || isnan(hnew)) { __asm__("int $3"); };
    if (err <= 1.0) {           /* step accepted  */
      facold = max_d (err, 1.0E-4);
      naccpt++;                 /*  stiffness test */
      if (!(naccpt % nstiff) || (iasti > 0)) {
        stnum = 0.0;
        stden = 0.0;
        for (i = 0; i < n; i++) {
          sqr = k2[i] - k6[i];
          stnum += sqr * sqr;
          sqr = yy1[i] - ysti[i];
          stden += sqr * sqr;
        }
        if (stden > 0.0)
          hlamb = h * sqrt (stnum / stden);
        if (hlamb > 3.25) {
          nonsti = 0;
          iasti++;
          if (iasti == 15) {
            if (fileout)
              fprintf (fileout, "Integro (mex): Stiffness detected.\n", x);
            xout = x;
            hout = h;
            return -4;
          }
        } else {
          nonsti++;
          if (nonsti == 6)
            iasti = 0;
        }
      }

      /* -------------------------------------------  */
      /*   Compute data for dense-output function     */
      /* -------------------------------------------- */
      if (DENSE)
        if (nrds == n)
          for (i = 0; i < n; i++) {
            yd0 = y[i];
            ydiff = yy1[i] - yd0;
            bspl = h * k1[i] - ydiff;
            rcont1[i] = y[i];
            rcont2[i] = ydiff;
            rcont3[i] = bspl;
            rcont4[i] = -h * k2[i] + ydiff - bspl;
        } else
          for (j = 0; j < nrds; j++) {
            i = icont[j];
            yd0 = y[i];
            ydiff = yy1[i] - yd0;
            bspl = h * k1[i] - ydiff;
            rcont1[j] = y[i];
            rcont2[j] = ydiff;
            rcont3[j] = bspl;
            rcont4[j] = -h * k2[i] + ydiff - bspl;
          }

      /* -----------------  */
      /*   Advance x        */
      /* ------------------ */
      memcpy (k1, k2, n * sizeof (double));
      memcpy (y, yy1, n * sizeof (double));
      xold = x;
      x = xph;                  
      /* -----------------  */
      /*   Output           */
      /* ------------------ */
      if (iout) {
        hout = h;
        xout = x;
        solout (naccpt + 1, xold, x, y, n, &irtrn);
        if (irtrn < 0) {
          if (fileout && VERBOSE)
            fprintf (fileout,
                     "Integro (mex): Stopping on signal from soulout.\n", x);
          return 2;
        }
      }                         /* normal exit */
      if (last) {
        hout = hnew;
        xout = x;
        return 1;
      }                         /* Adjust new h before new step */
      if (fabs (hnew) > hmax)
        hnew = posneg * hmax;
      if (reject)
        hnew = posneg * min_d (fabs (hnew), fabs (h));
      reject = 0;
    } else {                    /* step rejected */
      hnew = h / min_d (facc1, fac11 / safe);
      reject = 1;
      if (naccpt >= 1)
        nrejct = nrejct + 1;    /* fprintf(stderr, "-- %ld -- step rejected\n", nstep); */
      last = 0;
    }
    h = hnew;
  }; /* ENDS: while */
}                               
/* dopcor */


/* original DOPRI front-end --- mostly a double check since front end is done via Matlab */
int dopri5 (unsigned n, double x, double *y, double *pars,
           double xend, double *rtoler, double *atoler, int itoler,
           int iout, FILE * fileout, double uround,
           double safe, double fac1, double fac2, double beta, double hmax,
           double h, long nmax, double magbound, int meth, long nstiff,
           unsigned nrdens, unsigned *icont, unsigned licont)
{
  int arret, idid;
  unsigned i;                   /* initializations */

  nfcn = nstep = naccpt = nrejct = arret = 0;
  rcont1 = rcont2 = rcont3 = rcont4 = rcont5 = NULL;
  indir = NULL;                 
  /*              ode_pars = pars; */
  /* n, the dimension of the system */
  if (n == UINT_MAX) {
    if (fileout)
      fprintf (fileout, "System too big, max. n = %u\r\n", UINT_MAX - 1);
    arret = 1;
  }
  /* nmax, the maximal number of steps */
  if (!nmax)
    nmax = 100000;
  else if (nmax <= 0) {
    if (fileout)
      fprintf (fileout, "Wrong input, nmax = %li\r\n", nmax);
    arret = 1;
  }
  /* meth, coefficients of the method */
  if (!meth)
    meth = 1;
  else if ((meth <= 0) || (meth >= 2)) {
    if (fileout)
      fprintf (fileout, "Curious input, meth = %i\r\n", meth);
    arret = 1;
  }
  /* nstiff, parameter for stiffness detection */
  if (!nstiff)
    nstiff = 1000;
  else if (nstiff < 0)
    nstiff = nmax + 10;         
  /* iout, switch for calling solout */
  if ((iout < 0) || (iout > 2)) {
    if (fileout)
      fprintf (fileout, "Wrong input, iout = %i\r\n", iout);
    arret = 1;
  }                             
  /* nrdens, number of dense output components */
  if (nrdens > n) {
    if (fileout)
      fprintf (fileout, "Curious input, nrdens = %u\r\n", nrdens);
    arret = 1;
  } else if (nrdens) {          
    /* is there enough memory to allocate rcont12345&indir ? */
    rcont1 = (double *) malloc (nrdens * sizeof (double));
    rcont2 = (double *) malloc (nrdens * sizeof (double));
    rcont3 = (double *) malloc (nrdens * sizeof (double));
    rcont4 = (double *) malloc (nrdens * sizeof (double));
    rcont5 = (double *) malloc (nrdens * sizeof (double));
    if (nrdens < n)
      indir = (unsigned *) malloc (n * sizeof (unsigned));
    if (!rcont1 || !rcont2 || !rcont3 || !rcont4 || !rcont5
        || (!indir && (nrdens < n))) {
      if (fileout)
        fprintf (fileout, "Not enough free memory for rcont12345&indir\r\n");
      arret = 1;
    }                           
    /* control of length of icont */
    if (nrdens == n) {
      if (icont && fileout)
        fprintf (fileout,
                 "Warning : when nrdens = n there is no need allocating memory for icont\r\n");
      nrds = n;
    } else if (licont < nrdens) {
      if (fileout)
        fprintf (fileout,
                 "Insufficient storage for icont, min. licont = %u\r\n",
                 nrdens);
      arret = 1;
    } else {
      if ((iout < 2) && fileout)
        fprintf (fileout, "Warning : put iout = 2 for dense output\r\n");
      nrds = nrdens;
      for (i = 0; i < n; i++)
        indir[i] = UINT_MAX;
      for (i = 0; i < nrdens; i++)
        indir[icont[i]] = i;
    }
  }                             
  
  /* uround, smallest number satisfying 1.0+uround > 1.0 */
  if (uround == 0.0)
    uround = 2.3E-16;
  else if ((uround <= 1.0E-35) || (uround >= 1.0)) {
    if (fileout)
      fprintf (fileout, "DOPRI 'uround' out-of-range (1e-35,1): %.16e\r\n",
               uround);
    arret = 1;
  }                             
  
  /* safety factor */
  if (safe == 0.0)
    safe = 0.9;
  else if ((safe >= 1.0) || (safe <= 1.0E-4)) {
    if (fileout)
      fprintf (fileout, "Curious input for safety factor, safe = %.16e\r\n",
               safe);
    arret = 1;
  }                             
  
  /* fac1, fac2, parameters for step size selection */
  if (fac1 == 0.0)
    fac1 = 0.2;
  if (fac2 == 0.0)
    fac2 = 10.0;                
  
  /* beta for step control stabilization */
  if (beta == 0.0)
    beta = 0.04;
  else if (beta < 0.0)
    beta = 0.0;
  else if (beta > 0.2) {
    if (fileout)
      fprintf (fileout, "Curious input for beta : beta = %.16e\r\n", beta);
    arret = 1;
  }                             
  
  /* maximal step size */
  if (hmax == 0.0)
    hmax = xend - x;            
  
  /* is there enough free memory for the method ? */
  yy1 = (double *) malloc (n * sizeof (double));
  k1 = (double *) malloc (n * sizeof (double));
  k2 = (double *) malloc (n * sizeof (double));
  k3 = (double *) malloc (n * sizeof (double));
  k4 = (double *) malloc (n * sizeof (double));
  k5 = (double *) malloc (n * sizeof (double));
  k6 = (double *) malloc (n * sizeof (double));
  ysti = (double *) malloc (n * sizeof (double));
  if (!yy1 || !k1 || !k2 || !k3 || !k4 || !k5 || !k6 || !ysti) {
    if (fileout)
      fprintf (fileout, "Not enough free memory for the method\r\n");
    arret = 1;
  }                             
  
  /* when a failure has occured, we return -1 */
  if (arret) {
    if (ysti)
      free (ysti);
    if (k6)
      free (k6);
    if (k5)
      free (k5);
    if (k4)
      free (k4);
    if (k3)
      free (k3);
    if (k2)
      free (k2);
    if (k1)
      free (k1);
    if (yy1)
      free (yy1);
    if (indir)
      free (indir);
    if (rcont5)
      free (rcont5);
    if (rcont4)
      free (rcont4);
    if (rcont3)
      free (rcont3);
    if (rcont2)
      free (rcont2);
    if (rcont1)
      free (rcont1);
    return -1;
  } else {
    idid =
      dopcor (n, x, y, pars, xend, hmax, h, rtoler, atoler, itoler,
              fileout, iout, nmax, magbound, uround, meth, nstiff,
              safe, beta, fac1, fac2, icont);
    free (ysti);
    free (k6);
    free (k5);                  /* reverse order freeing too increase chances */
    free (k4);                  /* of efficient dynamic memory managing       */
    free (k3);
    free (k2);
    free (k1);
    free (yy1);
    if (indir)
      free (indir);
    if (rcont5) {
      free (rcont5);
      free (rcont4);
      free (rcont3);
      free (rcont2);
      free (rcont1);
    }
    return idid;
  }
} /* dopri5*/

/* dense output function */
double contd5 (unsigned ii, double x)
{
  unsigned i, j;
  double theta, theta1;

  i = UINT_MAX;
  if (!indir)
    i = ii;
  else
    i = indir[ii];
  if (i == UINT_MAX) {
    printf ("No dense output available for %uth component", ii);
    return 0.0;
  }
  theta = (x - xold) / hout;
  theta1 = 1.0 - theta;
  return rcont1[i] + theta * (rcont2[i] +
                              theta1 * (rcont3[i] +
                                        theta * (rcont4[i] +
                                                 theta1 * rcont5[i])));
} /* contd5 */


/*** Here: original fastode.c ***/


static double *traj;
static long trajLen;
static long trajMaxLen;
static double *params;

#ifndef ODE_ATOL
#define ODE_ATOL 1e-9
#endif

#ifndef ODE_RTOL
#define ODE_RTOL 1e-6
#endif

#ifndef ODE_EVTTOL
#define ODE_EVTTOL 1e-12
#endif

#define ODE_WIDTH (ODE_DIM+ODE_AUX+1)

#ifdef ODE_ALWAYS_ABORT
#define OOPS abort()
#else 
#define OOPS
#endif

/* ODE_FUNC -- computes flow, i.e. the ODE function itself */
/* ODE_AUXFUNC -- computes extra values associated with a state (optional) */
/* ODE_SCAN_EVENTS -- compute all event functions with positive crossings for events */
/* ODE_EVENT -- compute sigle event function for precise detection */
/* ODE_MAX_EVENTS -- number of event types used */

const int ode_max_events = ODE_MAX_EVENTS;
const int ode_dim = ODE_DIM;
const int ode_width = ODE_WIDTH;
const int ode_aux = ODE_AUX;
const int ode_npar = ODE_NPAR;
double ode_atol = ODE_ATOL;
double ode_rtol = ODE_RTOL;
double ode_evttol = ODE_EVTTOL;
double ode_max = ODE_MAX;

double eventValue [ODE_MAX_EVENTS];
int eventActive [ODE_MAX_EVENTS];

inline static void alwaysCheckEvt( int evid ) {
  assert( evid>=0 && evid<ODE_MAX_EVENTS );
  eventActive[evid] = ~0;
}; 


static int currentEvent = ODE_NO_EVENT;

#ifndef isfinite
inline static int isfinite( double x ) { return (x == x) && (2*x != x); };
#else
#warning "builtin isfinite macro found"
#endif

inline static void safeOdeFunc( 
  unsigned n, double t, double *state, double *par, double *deriv
  ) 
{
  int k;
  const char *msg = NULL;
  do {
    if ( n != ODE_DIM ) { 
      msg = "Dimension %d does not match code"; 
      k = n; 
      goto fail;
    };
    if (0!=t && !isfinite(t)) {
      msg = "t is not finite or %d";
      k = 0;
      goto fail;
    };
    for (k=0;k<ODE_DIM;k++)
      if (0!=state[k] && !isfinite(state[k])) {
        msg = "Nan in state[%d]";
        goto fail;
      };
    ODE_FUNC( n, t, state, par, deriv );
    for (k=0;k<ODE_DIM;k++)
      if (0!=deriv[k] && !isfinite(deriv[k])) {
        msg = "Nan in deriv[%d]";
        goto fail;
      };
    return;
    /* never reach this point */
  } while(0);
fail:
  fprintf(stderr,msg,k);
  __asm__("int $3");
}; /* ENDS: safeOdeFunc */

/** Return the ID number of the event that terminated integration 
    and all event detector function results.
    
    NOTE: after the first positive event, no futher functions are called
      so their values will be out of date
*/
int odeEventId(double *y0, int y0_dim) {
  int i;
  if (y0 && y0_dim>0) {
    if (y0_dim>ODE_MAX_EVENTS)
      y0_dim=ODE_MAX_EVENTS;
    for (i=0; i<y0_dim; i++) {
      y0[i] = eventValue[i];
    };
  };
  return currentEvent;
};

#ifndef ODE_MAX_EVENTS
#error "Must define ODE_MAX_EVENTS to the number of events"
#endif

/* ODE_EVENT( int evtId, double x, double *y, int n, double *p )
   Should compute crossover event function for event ID evtId.
   If there's no optimized version --> we run a full event scane, and
   return only the relevant result. 
   Code uses scan function for binary search event detection */
#ifndef ODE_EVENT
#define ODE_EVENT ode_event_from_scan
static double ode_event_from_scan(
  int evtId, double x, double *y, int n, double *p
  ) 
{
  ODE_SCAN_EVENTS( x, y, n, p, eventValue );
  return eventValue[evtId];
};
#endif /* ODE_EVENT */

static double multiEventWrapper( double x, double *y, int n, double *p ) {
  int k;
  double evt = -ODE_EVTTOL;
  double cand;

  ODE_SCAN_EVENTS( x, y, n, p, eventValue );
  for (k=0; k<ODE_MAX_EVENTS; k++) {
    cand = eventValue[k];
    if (cand>ODE_EVTTOL) { /* possible event */ 
      if (eventActive[k]) { /* verify crossing direction */
        evt = cand; 
        currentEvent = k; 
        break;
      };
    } else if (cand<-ODE_EVTTOL) { /* on pre-event side of boundary --> activate it */
      eventActive[k] = ~0;
      /* If cand is closer to threshold --> use it */
      if (cand>evt) { 
        evt = cand; 
      };
    };
  }; /* ENDS: loop to find events */
  
  return evt;
}; /* ENDS: multiEventWrapper */

int findEvent( double *csr, double e1 ){
  double e0, *y0, *y1, x0, x1, ec, xc;
  int k;

  assert( currentEvent>=0 && currentEvent<ODE_MAX_EVENTS && "event was detected" );
  
  if (trajLen<2) {
    fprintf(stderr,"WARNING: event called on trajectory with less than two points\n");
    OOPS;
    return -1;
  };

  /* last two entries in output */
  y1 = csr;
  y0 = csr - ODE_WIDTH;
  
  /* split into x and y parts */
  x1 = *y1; 
  y1++;
  x0 = *y0; 
  y0++;

  /* The event function has value e0 at x0 and e1 at x1
     For an event to occur, e0 < 0 <= e1, and we search the interval [x0 x1]
     for the root of the event function */     
  if (e1<0) {
    fprintf(stderr,"WARNING: event hasn't occured yet\n");
    OOPS;
    return -1;
  };
  
  e0 = ODE_EVENT( currentEvent, x0, y0, ODE_DIM, params );
  /* Make sure that event didn't occur *before* x0 */
  if (e0>=0) {
    fprintf(stderr,"WARNING: event %d happened *before* previous time step\n"
      , currentEvent);
    OOPS;
    /* memcpy(y1,y0,n*sizeof(double)); */
    return -2;
  };
  
  //!!!printf("Refining %d...\n", currentEvent);//!!!
  
  /* Loop until we found a "point" */
  int iter = 0;
  while (fabs(x0-x1)>ode_evttol || fabs(ec)>ode_atol) {
    /* Sanity: break-out after 64 rounds*/
    if (iter++>64) 
      break;
    /* Go to midpoint */
    xc = (x0+x1)/2;
    
    /* Interpolate trajectory at midpoint */
    for (k=0; k<ODE_DIM; k++)
      y1[k] = contd5(k,xc);
    
    /* Evaluate event function */
    ec = ODE_EVENT( currentEvent, xc, y1, ODE_DIM, params );
    
    /* Do binary search */
    if (ec==0) 
      break;
    if (ec<0)
      x0 = xc;
    else
      x1 = xc;
  };/* end of binary search loop */
  
  /* store event value */
  eventValue[currentEvent] = ec;
    
  /* time is at *csr; y1 was incremented */
  *csr = xc;

  //!!!  printf("Current event: %d time %g value %g iter %d\n", currentEvent, xc, ec, iter ); 
  
  return 0;
}; /* ENDS: findEvent */ 

void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn) {
  double *csr;
  double evt;
  unsigned long mask = ~0;
  
  if (nr>=trajMaxLen) { /* If out of storage for output --> bail out */
    *irtrn = -1;
    return;
  };
  
  /* Store the new time and value */
  if (nr != trajLen+1 && nr != 1) OOPS;
  trajLen = nr;
  csr = &traj[ODE_WIDTH*(nr-1)];
  *csr = x;
  memcpy( csr+1, y, sizeof(double)*n );

#ifdef ODE_AUXFUNC
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *csr, csr+1, params, csr+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */

#ifdef ODE_EVENT    
  /* Compute value of event function */
  evt = multiEventWrapper( *csr, csr+1, n, params ); 
  
  /* If positive --> an event occurred; search for it */
  if (evt>0) {
    int evtId = findEvent( csr, evt );
    /* event was found --> abort integration */
    *irtrn = -1;
    if (evtId<0)
      currentEvent = evtId;
  };
#endif /* ENDS: ODE_EVENT */
}; /* ENDS: solout */ 

int odeComputeAux(
  double *y0,
  int y0_dim,
  double *pars,
  int pars_dim
  )
{
  if (y0_dim != ODE_WIDTH) {
    fprintf(stderr,"ERROR: Output storage for dimension %d instead of %d\n",
        y0_dim, (int)ODE_WIDTH );
    OOPS;
    return -2;
  };
  
#ifdef ODE_NPAR
  if (pars_dim != ODE_NPAR) {
    fprintf(stderr,"ERROR: Parameters dimension %d instead of %d\n",
        pars_dim, (int)ODE_NPAR );
    OOPS;
    return -3;
  };
#endif /* ENDS: ODE_NPAR */
#ifdef ODE_AUXFUNC
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *y0, y0+1, pars, y0+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */
  return 0;
}; /* ENDS: odeComputeAux */

long odeOnce(
  double *yOut,
  int nmax,
  int yOut_dim,
  int startRow,
  double tEnd,
  double dt,
  double *pars,
  int pars_dim
  )
{
  static double y0 [ ODE_WIDTH ];
  
  currentEvent = ODE_NO_EVENT;
  
  if (yOut_dim != ODE_WIDTH) {
    fprintf(stderr,"ERROR: Output storage for dimension %d instead of %d\n",
        yOut_dim, (int)ODE_WIDTH );
    OOPS;
    return -2;
  };
  
#ifdef ODE_NPAR
  if (pars_dim != ODE_NPAR) {
    fprintf(stderr,"ERROR: Parameters dimension %d instead of %d\n",
        pars_dim, (int)ODE_NPAR );
    OOPS;
    return -3;
  };
#endif /* ENDS: ODE_NPAR */

  if (nmax<=startRow+1) {
    fprintf(stderr,"ERROR: starting beyond end of the array\n");
    OOPS;
    return -4;
  };
  
  if (pars_dim==0)
    pars=NULL;

  /* Event detector should have access to params too */
  params = pars;

  /* Output location. Contains initial condition */
  traj = yOut + ODE_WIDTH*startRow;
  
  /* Disable all eventss until we see a point at the "right" side */
  int k;
  for (k=0; k<ODE_MAX_EVENTS; k++) {
    eventActive[k] = 0;
  };
  
  /* Compute value of event function 
  if (multiEventWrapper( *traj,traj+1, ODE_DIM, params )>=0) {
    fprintf(stderr,"WARNING: Event occured before initial point was computed.\n" );
    OOPS;
    return -5;
  }; */
  
#ifdef ODE_AUXFUNC
  /* Compute auxiliary values for initial condition */
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *traj, traj+1, params, traj+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */

 /* We already have one point (ICS); start output at pos 1 */
  trajLen = 1;
  trajMaxLen = nmax-startRow;
  
  double atol = ODE_ATOL;
  double rtol = ODE_RTOL; 
  int rc;
  
  /* Poison output array for downstream testing */
  //!!! memset( traj+ODE_WIDTH, 0xee, (trajMaxLen-1)*ODE_WIDTH );
  /* Copy initial condition to temp storage b/c dopri code abuses it */
  memcpy( y0, traj, (1+ODE_DIM) * sizeof(double));
  //!!! printf("Integration start\n");
  rc = dopri5(
    ODE_DIM,  /*  unsigned n,       dimension of the system <= UINT_MAX-1*/
    *y0,      /*  double x,         initial x-value */
    y0+1,     /*  double* y,        initial values for y */
    pars,     /*  double *pars,     parameters */
    tEnd,     /*  double xend,      final x-value (xend-x may be positive or negative) */
    &ode_rtol,/*  double* rtoler,   relative error tolerance */
    &ode_atol,/*  double* atoler,   absolute error tolerance */
    0,        /*  int itoler,       switch for rtoler and atoler: SCALARS */
    2,        /*  int iout,         switch for calling solout: DENSE */
    stdout,   /*  FILE* fileout,    messages stream */
    2.3E-16,  /*  double uround,    rounding unit */
    0.9,      /*  double safe,      safety factor */
    0.2,      /*  double fac1,      parameters for step size selection (lower ratio) */
    10.0,     /*  double fac2,      (upper ratio) */
    0.04,     /*  double beta,      for stabilized step size control */
    dt,       /*  double hmax,      maximal step size */
    dt/2,     /*  double h,         initial step size */
    nmax,     /*  long nmax,        maximal number of allowed steps */
    ode_max,  /*  double magboud,   bound on magnitude of solution components */
    1,        /*  int meth,         switch for the choice of the coefficients */
    -1,       /*  long nstiff,      test for stiffness: NEVER */
    ODE_DIM,  /*  unsigned nrdens,  number of components for which dense outpout is required */
    NULL,     /*  unsigned* icont,  indexes of components for which dense output is required, >= nrdens */
    0         /*  unsigned licont   declared length of icont */
   );
  //!!!printf("Integration ended with %d samples\n", trajLen);
  
  return trajLen-1;
}; /* ENDS: odeOnce */


    """

    # check if recompilation is required

    rebuild = True
    if skiphash:
        if verbose:
            print "skipping hash calculation"
        rebuild=False
    else:
        hashfname = '.' + modname + '.c.sha256'
        s = hashlib.sha256()
        s.update(codestr)
        hashval = s.hexdigest()
        try:
            hf = open(hashfname, 'r')
            l = hf.readline()
            if str(hashval) == l:
                if os.path.isfile(modname+'.so'):
                    if verbose:
                        print "hash values agree. not rebuilding"
                    rebuild = False
        except IOError:
            # no hashfile found
            pass


    if rebuild:

        fn = tempfile.mktemp(suffix=".c", dir='.')
        of = open(fn, 'w')
        of.write(topcode)
        of.write(codestr)
        of.write(bottomcode)
        of.close()
        query = ''.join(["gcc -fPIC -g -O3 ", fn , " -lm -shared -o ",
            modname, ".so"])

        os.system(query)
        os.remove(fn)
        of = open(hashfname, 'w')
        of.write(str(hashval))
        of.close()

    F = FastODE(modname)
    return F


ex_py_code = r"""
# to run this code:
# code = mutils.makeODE.ex_py_code
# exec code

import mutils.makeODE as mo
import numpy as np

c_code = mo.ex_c_code
ode = mo.makeODE(c_code, 'testmod')

# prepare buffer: ode does not allocate memory!
buf = np.zeros((5000, ode.WIDTH), dtype=np.float64)

# set initial conditions
# first column is time
buf[0, 1:] = np.random.randn(ode.WIDTH - 1)

# set parameters
# spring-mass system: params are k, m, d
# format: list of floats
pars = [20., 4., .5]

# run ode until event is detected or maxtime is reached
# call: buffer, max time, max output interval, pars)
# returns: last line of buffer that was written to (other lines: still zero)
# NOTE: the params argument must be named (as below)!
N = ode.odeOnce(buf, 10., .01, pars=pars)

# visualize output
from pylab import figure, plot, show
fig = figure()
plot(buf[:N+1, 0], buf[:N+1, 1], 'r.-')
plot(buf[:N+1, 0], buf[:N+1, 2], 'g.-')
show()
"""

ex_c_code = r"""
/* some declarations for the solver */
/* note: math.h is already included */
#define ODE_DIM 2 /* excluding time */
#define ODE_NPAR 3 /* number of parameters: here, k,m,d*/

#define ODE_DT 1e-3 /* max step / initial step (optional, defaults to 1e-3) */

/* set solver accuracy (optional, default to 1e-9, 1e-6) */
#define ODE_RTOL 1e-11
#define ODE_ATOL 1e-11

void rhs(unsigned n, double t, double *Y, double *p, double *f)
{
    /* right hand side of the ODE
    function signature:
    n: iteration
    t: simulation time
    Y: state
    p: parameter
    f: return value (the "output")
    */
    f[0] = Y[1];
    f[1] = -p[0] * Y[0] / p[1] - p[2]*Y[1];
}

double event(double t, double *Y, int n, double *p, double *res)
{
    /* stop event function
    signature similar to above;
    this function terminates simulation if the output [1D!] has a positive
    zero_crossing 
    */

    /* Trigger: energy < 1 J */
    res[0] = -(.5*p[0]*Y[0]*Y[0] + .5*p[1]*Y[1]*Y[1] - 1.);
}

"""





