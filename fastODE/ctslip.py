import time, sys, copy
import scipy.optimize
import numpy.random
import pylab
import csync
from fastode import FastODE
from copy import copy as COPY
from pdb import set_trace as BREAKPOINT

execfile("hybsys.py")

FASTODE_CTSLIP = FastODE("ctslip")

def plotCircle( xc, yc, r, steps = 36, arc=(-pi,pi), **kw ):
  """Plot a circle around (xc,yc) with radius r, and specified number of 
     steps in the polygonal approximation"""
  t = linspace(arc[0],arc[-1],steps)
  return plot( xc + cos(t)*r, yc+sin(t)*r, **kw )

def plotZigZag( x0, y0, x1, y1, w, N = 7, **kw ):
  dx = array((x1-x0,y1-y0))
  l = sqrt(sum(dx**2.0))
  dx = dx/l
  dy = array((-dx[1],dx[0]))
  n = arange(0,N+1)
  ax = concatenate(([0], n/float(N), [1] ))
  ay = 0.5*concatenate(([0], (-1.0)**n, [0] ))
  x = ax*l*dx[0] + ay*w*dy[0] + x0
  y = ax*l*dx[1] + ay*w*dy[1] + y0
  return plot(x,y,**kw)

SRC = csync.getSrc("ctslip.c")

class CTSLIP_events( csync.FlexSOD ):
  Fields = csync.getFields(SRC,"CTSLIP_events",2)
    
class CTSLIP_state( csync.FlexSOD ):
  Fields = ['t']+csync.getFields(SRC,"CTSLIP_state",2)
  assert len(Fields) == FASTODE_CTSLIP.DIM+1
  
  def upd50_par( self, nm, val ):    
    if not hasattr( val, 'clkTD') or val.clkTD is None:
      return
    if val.gravityZ>=0:
      return
    # Estimate time 'till touchdown 
    t2td = sqrt( 2 * self.com_z / -val.gravityZ )
    # Create clk IC that cancels phase change while falling
    self.clk = val.clkTD - t2td * val.omega
    print "[upd] fixed initial clk for clkTD"

  def upd90_mdl(self, nm, val):
    x = CTSLIP_aux()
    y0 = numpy.concatenate( (self[:],zeros(len(x))) )
    val.computeAux( y0 )
    x.fromArray(y0[len(self):])
    # If leg 0 is on COM --> move to ref 
    if self.leg_z_0 == 0 and self.leg_x_0 == 0:
      self.leg_x_0 = x.ref_x_0
      self.leg_z_0 = x.ref_z_0
      print "[upd] Locked initial leg 0 position to reference"
    # if a penetrating stance --> fix it      
    if self.com_z+self.leg_z_0<0: 
      self.leg_x_0 *= abs(self.com_z)/abs(self.leg_z_0)
      self.leg_z_0 = -self.com_z      
      print "[upd] Initial leg 0 moved to Z=0"
    # If leg 1 is on COM --> move to ref 
    if self.leg_z_1 == 0 and self.leg_x_1 == 0:
      self.leg_x_1 = x.ref_x_1
      self.leg_z_1 = x.ref_z_1
      print "[upd] Locked initial leg 1 position to reference"
    # if a penetrating stance --> fix it      
    if self.com_z+self.leg_z_0<1: 
      self.leg_x_1 *= abs(self.com_z)/abs(self.leg_z_1)
      self.leg_z_1 = -self.com_z      
      print "[upd] Initial leg 1 moved to Z=0"
    return
  
  def upd50_plane( self, nm, coef ):
    """
    dot([1,state],coef)-val is the event function
    coef -- len<5 -- (val, co_clk, co_vx, co_z, co_vz, co_x), zeros appended
    """
    coef = array((list(coef)+[0]*6)[:6],float)
    if not any(coef):
      coef[0] = 1000
      print "[upd] plane event disabled"
    else:
      p = self.par
      (p.co_value, p.co_clk, 
       p.co_com_vx, p.co_com_z,
       p.co_com_vz, p.co_com_x) = coef
      print "[upd] plane is ", coef
  
  def getPlane( self ):
    p = self.par
    return array([p.co_clk,p.co_com_vx, p.co_com_z,p.co_com_vz, p.co_com_x])
        
  def planeVal( self, dat ):
    p = self.par
    co = self.getPlane()
    dat = asarray(dat)
    if dat.shape[-1] != len(co):
      s = CTSLIP_xst()
      dat = dat[..., [s.clk, s.com_vx, s.com_z, s.com_vz, s.com_x]]
    return dot(dat,co)+ p.co_value      
    
  def copy( self ):
    res = COPY(self)
    if hasattr(res,'par'):
      res.par = res.par.copy()
    return res
      
class CTSLIP_param( csync.FlexSOD ):
  Fields = csync.getFields(SRC,"CTSLIP_param",2)
  assert len(Fields) == FASTODE_CTSLIP.NPAR
    
  def upd20_stWn( self, nm, val ):
    self.stK = val*val
    self.stMu = self.stZeta * 2 * val
    print "[upd] set stK, stMu from stWn, stZeta"
    
  def upd09_hexed( self, nm, val ):
    tc,dc,phs,ph0,Kp,Kd = val
    # Omega is 2*pi/(time-of-cycle), but direction is negative!
    self.omega = -2*pi / tc
    # Our duty cycle is in radians
    self.stHalfDuty = dc * pi
    # Sweep angle is the same
    self.stHalfSweep = phs/2.0
    # Zero angle is "down" for hexed data
    self.stOfs = ph0-pi/2
    # Proportional FB is just like the torsional stiffness
    #   NOTE: for small changes; we use sin(delta) in the eqn
    self.tqKth = Kp
    # Given warning if a nonzero Kd is requested
    if Kd != 0:
      print "[upd] WARNING: Kd=%g requested by Kd not supported" % Kd
    print "[upd] applied hextuple ",repr(val)

  def upd10_stDecayFrac( self, nm, val ):
    # arc subtended by decay time
    ang = val * self.stHalfDuty
    # Natural frequency must finish arc in time
    self.stWn = abs(self.omega) * (2 * pi / ang) 
    print "[upd] set stWn from stDecayFrac"
    
  def upd90_stHalfDuty( self, nm, val ):
    if val<0 or val>pi:
      raise ValueError, "Half duty cycle range is 0..PI"
    if val==0 or val==pi:
      self.stHalfDuty += 1e-99
      print "[upd] fixed invalid stHalfDuty -- don't use 0 and PI!"
    
  def copy( self ):
    return COPY(self)

class CTSLIP_aux( csync.FlexSOD ):
  Fields = csync.getFields(SRC,"CTSLIP_aux",2)
  assert len(Fields) == FASTODE_CTSLIP.AUX
  
class CTSLIP_xst( csync.FlexSOD ):
  "Extended state -- includes auxvars"
  Fields = (    ['t']
           +    csync.getFields(SRC,"CTSLIP_state",2)
           +    csync.getFields(SRC,"CTSLIP_aux",2) 
           )
  assert len(Fields) == FASTODE_CTSLIP.WIDTH

  def upd_dbg( self, name, val ):
    print "UPD called"
    
  def cfgFromParam( self, pars ):
    self.vis_R = pars.len0/5
    self.vis_W = self.vis_R * 0.75
    self.vis_ofs = pars.stOfs
    if abs(pars.stHalfSweep - pars.stHalfDuty)>0.02:
      self.vis_swp = pars.stHalfSweep
    else:
      self.vis_swp = 0
    self.domain = int(pars.domain)
    
  def plot( self ):
    X = self.com_x
    Z = self.com_z+self.zOfs
    l = plotCircle( X, Z, self.vis_R, 36, color='k', linewidth=2 )
    if self.vis_swp:
      l.append( plotCircle( X, Z, self.vis_R, 18,
        arc=(self.vis_ofs-self.vis_swp, self.vis_ofs+self.vis_swp),
        color=[0.6,0.9,0.6], linewidth=5 
      ))
    l.append( plot( 
        [X, X+self.vis_R*1.6*cos(self.clk)],
        [Z, Z+self.vis_R*1.6*sin(self.clk)],
        color='m',linewidth= 2 
      ))
    for leg in xrange(2):
      lx = getattr(self,"leg_x_%d"%leg)
      lz = getattr(self,"leg_z_%d"%leg)
      rx = getattr(self,"ref_x_%d"%leg)
      rz = getattr(self,"ref_z_%d"%leg)
      lc = 'brgmck'[leg]
      l.append( plot([X+rx],[Z+rz],'d'+lc) )
      lkw = { 'color' : lc, 'linewidth' : 2 }
      if self.domain & (1<<leg):
        l.extend(
          plotZigZag( X+lx*0.2, Z+lz*0.2, X+lx*0.8, Z+lz*0.8, 
                      self.vis_W, 7, **lkw )
          + plot( [X,X+lx*0.2],[Z,Z+lz*0.2], **lkw )
          + plot( [X+lx*0.8,X+lx],[Z+lz*0.8,Z+lz], **lkw )
          + plot( [X+lx], [Z+lz], '^'+lc )
        )
      else:
        l.extend( 
          plot( [X, X+lx], [Z, Z+lz], '-', **lkw )
          #+plot( [X+lx], [Z+lz], 'o'+lc )
        )
    return l

class ENUM:
  def __init__(self, *argv ):
    self._tbl={}
    for nm,val in zip(argv,range(len(argv))):
      setattr(self,nm,val)
      self._tbl[val]=nm
  
  def __len__( self ):
    return len(self._tbl)
    
  def __getitem__(self,key):
    try:
      return self._tbl[int(key)]
    except KeyError:
      return "UNKNOWN<%d>" % key     
    except ValueError:
      pass
    try:
      return getattr(self,key)
    except AttributeError:
      pass
    if type(key)==int:
      return key
    return "UNKNOWN<%s>" % key
      
        
DomName = ENUM('flyDown','stand0','stand1','both','flyUp')
EvtName = ENUM(*CTSLIP_events.Fields)
EvtName._tbl[-1] = "(START)"

class CTS( HybSys ):
  F = CTSLIP_xst()
  P = CTSLIP_param()
  
  # Build domain transitions from human readable form
  TRANS = {}
  for x in [ 
    'stand0 lift0 flyUp',
    'stand1 lift1 flyUp',
    'stand0 land1 both',
    'stand1 land0 both',
    'both lift0 stand1',
    'both lift1 stand0',
    'flyDown land0 stand0',
    'flyDown land1 stand1',
    'stand0 apex stand0',
    'stand1 apex stand1',
    'both apex both',
    'flyUp apex flyDown' ]:
    src,ev,dst = x.split(" ")
    TRANS[(DomName[src],EvtName[ev])] = DomName[dst]
    
  def __init__(self):
    HybSys.__init__(self,FASTODE_CTSLIP)
    self.domDt = [ 1e-3 ] * 5 
    self.toApex = False
    self.planeCounter = 0
    self.planeTime = -1
    self.planeGap = 0.001
    self.setGround()
     
  def getParam( self ):
    return CTSLIP_param().fromArray( self.param )
  
  def getICS( self ):
    return CTSLIP_state().fromArray( self.y[0,:self.ode.DIM+1] )
          
  def integrate(self,*argv,**kwarg):
    p = self.getParam()
    assert p.mass > 0, "Params not initialized"
    assert p.stHalfDuty > 0 and p.stHalfDuty<pi, "Invalid duty cycle"
    assert all(abs(self.param) < 1e6), "Param range is sane"
    # Reset plane event latch; only one plane event allowed per domain entry
    self.planeTime = -1
    res = HybSys.integrate(self,*argv,**kwarg)
    return res
  
  def setGround( self, gseq=() ):
    self.zOfsSeq = list(gseq)
    self.zOfs = 0
    self.zOfsPos = self.csr-1
    return self

  def changeGroundZ( self, state ):
    # New ground height
    if self.zOfsSeq:
      nOfs = self.zOfsSeq.pop(0)
      dz = nOfs - self.zOfs
    else:
      nOfs = self.zOfs
      dz = 0
    # Next stride
    state[CTS.F.com_z] -= dz 
    self.y[self.zOfsPos:self.csr,CTS.F.zOfs] = self.zOfs
    # Update ground state
    self.zOfsPos = self.csr-1
    self.zOfs = nOfs
    
  def transition(self, dom, evt, state, ev ):
    res = CTS.TRANS.get((dom,evt),-1)
    if evt == EvtName.plane:
      if state[CTS.F.t]-self.planeTime < self.planeGap:
        #print "<duplicate>", state[CTS.F.t]-self.planeTime
        res = None
      else: # process a plane event
        #print "<plane>", state[CTS.F.t], "dt", state[CTS.F.t]-self.planeTime
        self.planeTime = state[CTS.F.t]
        if self.planeCounter > 0:
          res = dom
          self.planeCounter -= 1
        else:
          res = -1
    # Apex events trigger ground z changes
    elif evt==EvtName.apex:
      if self.toApex:
        res = -1
      #elif hasattr(self,'zOfsSeq'):
      self.changeGroundZ(state)
    # If transitioned into fly-up 
    if res==DomName.flyUp:
      # If moving downwards --> silently transition out of fly-up
      if state[CTS.F.com_vz]<0:
        res = DomName.flyDown    
      else: # else --> liftoff event, adjust clock phases
        u = self.param[CTS.P.xiUpdate]
        c = wrap(state[CTS.F.clk])
        r = self.param[CTS.P.xiLiftoff]
        if c<0:
          r -= pi
        state[CTS.F.clk] = c * (1-u) + r * u
    if res<0:
      if self.loud>1:
        print "Termination (%s,%s)-->%s" % (DomName[dom],EvtName[evt],res)
    return res,state

  def timesFor(self, dom):
    return (1000,self.domDt[int(dom)])
    
  def event(self,key):
    if type(key)!=slice:
      val = [self.getEvents(key)]
    else:
      val = self.getEvents(key)
    st = CTSLIP_xst()
    ev = CTSLIP_events()
    return [ 
      (pos,dom,evt,st.fromArray(st0).toDict(),ev.fromArray(ev0).toDict())
      for (pos,dom,evt,st0,ev0) in val 
    ]

  def narrate( self ):
    for pos,dom,evt,st,ev in self.getEvents():
      print "%-8.5g at %5d %8s --> %s" % (
        st[0],pos,DomName[dom],EvtName[evt] )
        
  def plotAt( self, idx ):
    st = CTSLIP_xst()
    st.cfgFromParam( CTSLIP_param().fromArray( self.param ) )
    if type(idx)==int:
      if idx<0:
        idx = [len(self)+idx]
      else:
        idx = [idx]
    if type(idx)==slice:
      idx=xrange(*idx.indices(len(self)))
    res = {}
    for k in idx:
      st.domain = self.d[k]
      st.fromArray( self.y[k,:] )
      res[k] = st.plot()
    return res

  def axisPan( self, idx ):
    ax = axis()
    st = CTSLIP_xst()
    X = self.y[idx,st.com_x]
    X0 = (ax[0]+ax[1])/2.0
    ax =( ax[0]-X0+X, ax[1]-X0+X ) + ax[2:]
    axis(ax)
    
  def energy( self ):
    s = CTSLIP_xst()
    p = CTSLIP_param().fromArray(self.param)
    # Useful subset of samples
    y = self.y[:self.csr,:]
    # Reference lengths for legs
    l0r = sqrt( y[:,s.ref_x_0]**2 + y[:,s.ref_z_0]**2 )
    l1r = sqrt( y[:,s.ref_x_1]**2 + y[:,s.ref_z_1]**2 )
    # Actual lengths for legs
    l0 = sqrt( y[:,s.leg_x_0]**2 + y[:,s.leg_z_0]**2 )
    l1 = sqrt( y[:,s.leg_x_1]**2 + y[:,s.leg_z_1]**2 )
    # Velocity squared
    v2 = y[:,s.com_vx]**2 + y[:,s.com_vz]**2    
    # Unit vector along gravity
    down = numpy.array([p.gravityX,p.gravityZ])
    g = norm(down)
    down = down / g
    # Height
    h = -down[0]*y[:,s.com_x]-down[1]*(y[:,s.com_z]+y[:,s.zOfs])
    # Kinetic energy
    ek = 0.5*p.mass*v2
    # Gravitational energy
    eg = p.mass*g*h
    # Elastic energy 
    el0 = 0.5*p.stK*(l0r-l0)**2
    #   set to 0 when leg is not in stance
    numpy.put( el0, self.d[:self.csr] & 1 == 0, 0 )
    # Elastic energy 
    el1 = 0.5*p.stK*(l1r-l1)**2
    #   set to 0 when leg is not in stance
    numpy.put( el1, self.d[:self.csr] & 1 == 0, 0 )
    
    return numpy.c_[el0+el1+ek+eg,ek,eg,el0,el1,l0r,l0,l1r,l1,sqrt(v2)]
  
  def sectionmap( self, args, depth = 1, 
        preslc = slice(None,-1), 
        postslc = slice(1,None),
        skip=0 ):
    self.toApex = False
    pre = []
    post = []
    vel = []
    for arg in args:
      self.planeCounter = depth
      self.planeTime = -1
      self.mapping( arg )
      # Get positions of plane events
      slc = asarray([ ev[0] for ev in self.getEvents() 
        if (ev[2] == EvtName.plane 
            and ev[1] in [DomName.flyUp, DomName.flyDown]
            and ev[0] > skip)
      ])
      # Get indices of pre-slice entries
      prei = slc[preslc]
      posti = slc[postslc]
      
      if not prei or not posti:
        print "!",#"WARNING: no plane events"
        continue
        
      # Get values at pre, pre+1, post
      pre0 = self.y[prei,:]
      delta = self.y[prei+1,:] - pre0
      post0 = self.y[posti,:]
      
      if 0:
        print "<>",len(slc),len(self.seq),y0.planeVal(pre0)
        subplot(211)
        plot( self.y[slc,s.t], self.y[slc,s.com_vz], '.r')
        plot( self.x[-1,s.t], self.x[-1,s.com_vz], 'or')
        plot( self.x[:,s.t], self.x[:,s.com_vz])
        plot( pre0[:,s.t], pre0[:,s.com_vz], 'vk')
        plot( post0[:,s.t], post0[:,s.com_vz], '^y')
        subplot(212)
        plot( self.y[slc,s.t], y0.planeVal(self.y[slc,:]), '.r')
        plot( self.x[:,s.t], y0.planeVal(self.x), ',-' )
        plot( pre0[:,s.t], y0.planeVal(pre0), 'vk')
        plot( post0[:,s.t],y0.planeVal(post0) , '^y')
      
      pre.append(pre0)
      vel.append(delta)
      post.append(post0)
    if 0:
      subplot(211); grid(1)
      subplot(212); grid(1)
    print "Found ", len(pre)
    return (
      concatenate(pre,axis=0),
      concatenate(post,axis=0),
      concatenate(vel,axis=0)
    )

  def apexMap( self, args ):
    """
    Compute the apex map, penalizing termination due to other events 
    """
    self.toApex = True
    post = self.mapping(args)
    if self.seq[-1][2] != EvtName.apex:
      post += 1e5
    return post
  
  def apexError( self, args ):
    """
    Compute sum-squared error of apex map. First args entry must be clock phase
    """
    idx = [ self.F.clk, self.F.com_vx, self.F.com_z ]
    post = asarray(self.apexMap(args))
    pre = self.x[0,idx]
    post = self.x[-1,idx]
    post[0] = post[0] % pi
    return norm((post-pre)*array([0.1,1,1]))
  
  def apexStability( self, args ):
    assert self.F.clk == self.icsMap[0]
    assert self.F.clk == self.rtnMap[0]
    args = asarray(args)
    args[0] = args[0] % pi
    post1 = asarray(self.apexMap(args))
    post1[0] = post1[0] % pi
    post2 = asarray(self.apexMap(post1))
    post2[0] = post2[0] % pi
    return norm(post1-args) + norm(post2-post2)/norm(post1-args)
    
  def apexFixedPoint( self ):
    """
    Find a nearby fixed-point of the apex map
    """
    ic = concatenate((self.y[0,self.icsMap],self.param[self.parMap]))
    opt = scipy.optimize.fixed_point( self.apexMap, ic, xtol=1e-5 )
    self.useArgs( opt )
    return opt
  
  def mapseq( self, maxlen=100 ):
    r = []
    while self.seq[-1][2] in [EvtName.apex, EvtName.plane] and len(r)<maxlen:
      r.append(self.y[0,:].copy())
      self.remap()
    r = array(r)
    return r
   
    
# ENDS: class CTS

def wrap( ang ):
  ang = numpy.asarray(ang)
  return ang - floor((ang+pi)/(2*pi))*(2*pi)
  
