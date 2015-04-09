#!/usr/bin/python
#
# Hybrid System Modeller based on fastode
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
# (c) Shai Revzen, Berkeley 2007, U. Penn 2010
#
 
import fastode
import numpy

class HybSys( object ):
  """Wrapper for a hybrid system ODE based on fastode
     The fastode system is expected to take the domain ID in as
     its last parameter in the parameter vector
     
     HybSys code integrates using fastode, detects transition events
     and computes the transition, accumulating the results until 
     a termination condition occurs.
     
     Subclasses must define:
       transition( dom, evt, state ) --> new domain id, or <0 to terminate
  """
  def __init__(self,ode):
    """Initialize hybrid system.
       
       INPUTS:
         ode -- a fastode.FastODE instance
    """
    if not isinstance(ode,fastode.FastODE):
      raise TypeError('ode must be a fastode.FastODE not "%s"' % str(type(ode)))
    self.ode = ode
    self.ev = numpy.zeros(ode.MAX_EVENTS,numpy.double)
    self.param = numpy.zeros(ode.NPAR,numpy.double)
    self.N = 0
    self.y = None
    self.d = [0]
    self.domain = -1
    self.icsMap = []
    self.parMap = []
    self.rtnMap = []
    self.loud = 0
    self.reset()
    
  def setVebose( self, verb ):
    self.loud = verb
    
  def initFrom( self, other ):
    """Copy initial cond, params from another HybSys object"""
    assert self.y.shape[1] == other.y.shape[1]
    assert len(self.param) == len(other.param)
    self.y[0,:] = other.y[0,:]
    self.param = other.param
    return self

  def setParam( self, par ):
    """Set fastode parameters. If par is not already a numpy.ndarray
       its toArray method is called to convert.
       
       The par[-1] entry is expected to be a domain ID, and is overwritten
       later by the HybSys integrate method 
    """
    if type(par) != numpy.ndarray:
      par = par.toArray()
    if type(par) != numpy.ndarray or len(par) != self.ode.NPAR:
      raise TypeError,"Expected a numpy.ndarray of length %d" % self.ode.NPAR
    self.param = numpy.copy(par)
    return self
    
  def prepare( self, N ):
    """Allocate storage for an integration 
        N -- size of trajectory buffer
       
       Integration will never return more than N points
    """
    self.y = numpy.zeros((N,self.ode.WIDTH),numpy.double)
    # Storage for domain ID-s
    self.d = numpy.zeros(N,numpy.int32)
    self.N = N
    return self.reset()
  
  def reset( self ):
    """Reset integrator to its initial state"""
    # If no domain set --> Reset to previous initial domain
    if self.domain<0:
      self.setDomain( self.d[0] )
    # Reset output cursor
    self.csr = 1
    self.seq = []
    return self
  
  def setICS( self, ics ):
    """Set fastode initial conditions. 
       If ics is not already a numpy.ndarray its toArray method is 
       called to convert.
       ics[0] is the starting time; len(ics) == DIM_OF_ODE 
       
       if ics is an object with an attribute 'par', apply setParam(par)
       before initial condition
    """
    if hasattr( ics, 'par' ):
      self.setParam( ics.par )
    if self.y is None:
      raise ValueError, "Initial condition cannot be set before prepare()"
    if len(ics) != self.ode.DIM+1:
      raise TypeError, "Initial condition must be of dim %d" % (self.ode.DIM+1)
    if type(ics) != numpy.ndarray:
      ics = ics.toArray()
    else: # If was a numpy array -- use a copy!
      ics = numpy.copy(ics)
    if type(ics) != numpy.ndarray:
      raise TypeError,"Expected a numpy.ndarray"
    self.y[0,:self.ode.DIM+1] = ics
    return self

  def computeAux( self, st = None ):
    """Extend state to include auxiliary variables
       If st is omitted, extends initial values
    """
    if st is None:
      self.ode.odeComputeAux( self.y[0,:], self.param )
    else:
      self.ode.odeComputeAux( st, self.param )
    
  def setMapIO( self, idxIn=None, idxOut=None, idxPar=None ):
    """Use the integrator as a mapping from initial conditions and 
       parameters to final conditions
         idxIn (optional) -- indices of state vector that arguments modify
         idxOut (optional) -- indices of state vector to return
         idxPar (optional) -- parameter indices to modify
    """
    # If an idx was provided --> update the mapping keys
    if idxIn is not None:
      self.icsMap = numpy.asarray(idxIn)
    # If an idx was provided --> update the mapping keys
    if idxOut is not None:
      self.rtnMap = numpy.asarray(idxOut)
    if idxPar is not None:
      self.parMap = numpy.asarray(idxPar)
    
  def useArgs( self, args ):
    """Apply map arguments as specified by setMapIO()"""
    # Apply arguments
    args = numpy.array(args) # this makes a copy if args was an array
    lic = len(self.icsMap)
    lar = len(args)
    # If arguments only modify the ICS --> apply the change
    if lar<=lic:
      numpy.put(self.y[0,:],self.icsMap[:lar],args)
    else: # else --> apply ICS and parameter changes
      numpy.put(self.y[0,:],self.icsMap,args[:lic])
      numpy.put(self.param,self.parMap[:lar-lic],args[lic:])
    return self

  def getForMap( self, idx ):
    """Implement indexing used by setMapIO()"""
    if iterable(idx):
      return concatenate(( 
        self.y[idx,self.rtnMap],
        [self.param[self.parMap]]*len(idx) 
      ))
    return concatenate(( 
      self.y[idx,self.rtnMap],
      self.param[self.parMap] 
    ))

  def getPost( self ):
    """Return results as specified by setMapIO()"""
    return self.getForMap( self.csr-1 )
    
  def getPre( self ):
    """Return initial values as specified by setMapIO()"""
    return self.getForMap( 0 )
    
  def doMap( self ):
    """Rerun integration"""
    self.reset()
    self.integrate()
    return self
      
  def mapping( self, args ):
    """Use the integrator as a mapping from initial to final conditions
         args -- sequence of argument values. If a nested sequence or rank>1
           array is passed in, generates a list of self.mapping applied to each 
           member.
       args sets initial conditions and parameters, see useArgs(), setMapIO()
    """
    if len(args) and iterable(args[0]):
      return [ self.mapping(arg) for arg in args ]
    return self.useArgs(args).doMap().getPost()
  
  def remap( self ):
    """Use the integrator as a mapping
       Use final condition of last integration as a new initial condition
       
       This function can be used to test purported fixed points
    """
    self.y[0,:] = self.y[self.csr-1,:]
    return self.doMap().getPost()
    
  def sampleAt( self, seq ):
    "Find index of nearest sample point after each seq memeber" 
    res = numpy.searchsorted( self.y[:self.csr,0], seq )
    if len(res) and res[-1]==self.csr:
      res[-1] = res[-1] - 1
    return res
    
  def integrateOnce( self ):
    """(private) integrate the HybSys through a single domain""" 
    if not self.N or self.y.shape[0] != self.N:
      raise TypeError,"Must initialize HybSys with prepare()"
    if self.csr>=self.y.shape[0]:
      raise IndexError,"No space for results"
    if self.domain<0:
      raise KeyError,"Domain is unpecified -- compute event transition first"
    if self.loud>2:
      print "%3d at %10d dom %d" % (len(self.seq),self.csr-1,self.domain),
    # Record domain initial condition
    self.d[self.csr-1] = self.domain
    # Prepare state for fastode
    self.param[-1] = self.domain
    # Integrate and find termination event ID
    tMax, dT = self.timesFor( self.domain )
    l = self.ode.odeOnce( 
      self.y, tMax, dT, slc=slice(self.csr-1,-1), pars=self.param )
    # Integration completed --> Get the termination event in fresh buffer
    self.ev = empty_like(self.ev)
    evt = self.ode.odeEventId( self.ev )
    if self.loud > 2:
      print " evt %-6d len %d" % (evt,l)
    # If event is a normal event condition --> report only relevant values 
    if evt>=0 and l>0:
      rpt = self.ev[:evt+1]
    elif l<0: # if error --> report error-1000 as event code
      rpt = self.ev
      evt = l-1000
      l = 0
    else: # else was abnormal event / error --> report all event values
      rpt = self.ev
    # Record domain in trajectory
    self.d[self.csr:self.csr+l] = self.domain
    # Store transition event details in self.seq
    self.csr = self.csr + l
    self.seq.append( 
        (self.csr-1, self.domain, evt, self.y[self.csr-1,:], rpt)
    )
    # Compute transition to new domain
    dom,st = self.transition( *self.seq[-1][1:] )
    # If event should be ignored --> pop from .seq
    if dom is None:
      self.seq.pop(-1)
    else: # else --> execute transition
      self.domain = dom
      self.y[self.csr-1,:] = st
    
  def integrate( self ):
    """Integrate the HybSys until it transition()-s to a negative domain ID
       returns the domain ID that caused termination""" 
    self.seq.append( 
        (self.csr-1, self.domain, -1, self.y[self.csr-1,:], [])
    )
    if self.loud>1:
      print "Starting at position ", self.csr-1
    while self.domain>=0:
      self.integrateOnce()
    # Store result data
    self.x = self.y[:self.csr,:]
    return self.domain
  
  def clip( self, finish ):
    """Clip integration results at specified sample number"""
    if finish<=0:
      finish=finish+self.csr
    self.csr = finish
    self.x = self.y[:self.csr,:]
    self.seq = [x for x in self.seq if x[0]<self.csr]
   
  def __getitem__( self, key ):
    """Read trajectory data"""
    return self.y[key]
  
  def dom( self, key ):
    """Read domain ID for specified slice of trajectory"""
    return self.d[key]
    
  def __len__( self ):
    """Length of trajectory computed"""
    return self.csr
    
  def getEvents( self, key=None ):
    """Get list of all events
       Format is list of tuples (pos,dom,evt,state):
         pos -- position in trajectory
         dom -- domain in which event was detected
         evt -- event ID from fastode 
         state -- the state returned by fastode at the event
    """
    if key is None:
      return self.seq
    return self.seq[key]
      
  def setDomain( self, dom ):
    """Set the initial domain for the next integration"""    
    if dom<0 or dom != int(dom):
      raise TypeError,"Domain ID must be int >= 0"
    self.domain = dom
    return self

  def timesFor( self, dom ):
    """Specify integration time limits:
            tMax, dT = timesFor( dom ) 
       tMax -- absolute end time
       dT -- time step
       dom -- domain ID for which we are integrating
       
       * NOTE: Subclasses SHOULD override this method
    """
    return (10,1e-3)
    
  def transition( self, dom, evt, st, ev ):
    """Compute the target domain of a hybrid transition event
       
        * NOTE: Subclasses SHOULD override this method
       
       Given initial domain dom, fastode event evt with results ev and
       state st, returns the domain ID of the next domain in which to 
       integrate. 
       
       OUTPUT
         domain -- int or None -- new fastode, None to ignore (skip) an,
           or negative to terminate integration.
         state -- ODE_WIDTH -- new state after transition
    """
    return -1,st # Terminate at first event

