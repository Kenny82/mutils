execfile('ctslip.py')

from numpy import r_,c_,unwrap,arctan2
from pylab import *

def CVG():
  cm.y[0,:]=cts.y[0,:]
  cm.setParam(p)
  cm.fixedPoint()
  r = cm.mapseq()
  ldr = log(abs(diff(r))+1e-99)
  plot(ldr[0,:])
  plot(ldr[1,:])
  print ldr.shape
  return r
    
def PLT(cts, thr=9):
  N = cts.csr
  y = cts.y[:N,:].copy()
  plot([min(y[:,s.com_x]),max(y[:,s.com_x])],[0,0],color=[0.3,0.3,0],linewidth=2)
  plot(y[:,s.com_x]+y[:,s.leg_x_0],y[:,s.com_z]+y[:,s.leg_z_0],'b')
  plot(y[:,s.com_x]+y[:,s.leg_x_1],y[:,s.com_z]+y[:,s.leg_z_1],'r')
  L = plot(y[:,s.com_x],y[:,s.com_z],'k')
  cts.plotAt([t[0] for t in cts.getEvents() if t[0]<N ])
  return L
  #cts.plotAt(cts.sampleAt(linspace(cts.y[0,0],cts.y[cts.csr-1,0],8)))

def psweep( spec, nm, seq, vis=PLT ):
  cm = spec.mdl
  ics = spec.copy()
  L=[]
  V=[]
  for val in seq:
    setattr( ics.par, nm, val )
    cm.setICS(ics).reset().integrate()
    L.append(vis(cm))
    V.append('%.3g' % val)
  legend(L,V,loc=0)

def sweep( spec, nm, seq, vis=PLT ):
  cm = spec.mdl
  ics = spec.copy()
  L=[]
  V=[]
  for val in seq:
    setattr( ics, nm, val )
    cm.setICS(ics).reset().integrate()
    L.append(vis(cm))
    V.append('%.3g' % val)
  legend(L,V,loc=0)

def visRtn(idx):
  def vis(cm):
    x0 = cm.y[0,idx]
    x1 = cm.y[cm.csr-1,idx]
    plot([x0,x1],[x0,x1],'k')
    return plot([x0],[x1],'o')
  return vis
  
def visXiZ(cm):
  y = cm.x.copy()
  res = plot(wrap(y[:,s.clk]*2)/2,y[:,s.com_z])
  xlabel('Clock phase (rad)')
  ylabel('Z (m)')
  return res

def visXZ(cm):
  y = cm.x.copy()
  plot(y[:,s.com_x],y[:,s.zOfs],'k')
  res = plot(y[:,s.com_x],y[:,s.com_z]+y[:,s.zOfs])
  xlabel('X (m)')
  ylabel('Z (m)')
  return res
  
def visVZ(cm):
  y = cm.x.copy()
  res = plot(y[:,s.com_vz],y[:,s.com_z])
  xlabel('Vz (m/s)')
  ylabel('Z (m)')
  return res
  
def visTZ(cm):
  y = cm.x.copy()
  plot(y[:,s.t],y[:,s.zOfs],'k')
  res = plot(y[:,s.t],y[:,s.com_z]+y[:,s.zOfs])
  xlabel('T (sec)')
  ylabel('Z (m)')
  return res
  
def visTX(cm):
  y = cm.x.copy()
  plot(y[:,s.t],y[:,s.zOfs],'k')
  res = plot(y[:,s.t],y[:,s.com_x])
  xlabel('T (sec)')
  ylabel('X (m)')
  return res
  
def visTVX(cm):
  y = cm.x.copy()
  plot(y[:,s.t],y[:,s.zOfs],'k')
  res = plot(y[:,s.t],y[:,s.com_vx])
  xlabel('T (sec)')
  ylabel('Vx (m/sec)')
  return res
  
def visTC(cm):
  p = CTSLIP_param()
  om = cm.param[p.omega]
  y = cm.x.copy()
  res = plot(y[:,s.t],wrap(y[:,s.clk]-y[:,s.t]*om))
  xlabel('T (sec)')
  ylabel('clk-omega')
  return res

def visCDT(cm):
  y = cm.x.copy()
  t = y[:,s.t]
  c = unwrap(y[:,s.clk])
  res = plot(t,c-linspace(c[0],c[-1],len(c)))
  xlabel('T (sec)')
  ylabel('clk-trend')
  return res
  
def visXE(cm):
  y = cm.x.copy()
  E = cm.energy()
  plot(y[:,s.com_x],E[:,0])
  res = plot(y[:,s.com_x],E[:,1])
  xlabel('X (m)')
  ylabel('Energy (Joul)')
  return res
  

def hilbert( z ):
  """Compute the Hilbert transform of a real time series"""
  f = numpy.fft.fft(z)
  N = floor(len(f)/2)
  h = numpy.real(numpy.fft.ifft(numpy.r_[-f[:N]*1j,f[N:]*1j]))
  return h

def phiOf( sig ):
  """Use Hilbert transform to compute phase"""
  sig = numpy.asarray(sig)
  a = numpy.arctan2(hilbert(sig),sig-mean(sig))
  return unwrap(a)
  
def zscore( X ):
  """Compute the Z scores of the columns of X"""
  m = mean(X)
  s = std(X)
  # So simple -- isn't array broadcasting wonderful? 
  return (X-m)/s

def pca( X ):
  """Compute the PCA of the rows of X""" 
  m = mean(X)
  X = X-m
  U,S,V = numpy.linalg.svd( dot(X.T,X) )
  print "SVD ",U.shape,S.shape,V.shape
  return U, dot(numpy.diag(S),V), S

def lfit( dat ):
  """Fit a linear model to the data
     Data is one sample per column of dat
     Returns a dat.shape[0] by 2 matrix: intercepts, slopes
  """
  if len(dat.shape)<2:
    dat = numpy.mat(dat)
  N = dat.shape[1]
  reg = c_[ ones(N), arange(N) ]
  return numpy.linalg.lstsq( reg, dat.T )[0]

def plots( dat, *argv, **kw ):
  return [ plot(d,*argv, **kw) for d in dat ]
  
def phest( y ):
  """Return a phase estimate from an array of CTSLIP states
     Currently uses com_z and com_vz only
  """
  s = CTSLIP_xst()
  z0 = c_[y[:,s.com_z],y[:,s.com_vz],y[:,s.com_vx]]
  z = (z0 - mean(z0)) / std(z0)
  #return unwrap(arctan2(z[:,1],z[:,0]))
  zh = r_[map(hilbert,z.T)]
  a = r_[map(unwrap,arctan2( z.T, zh ))]
  return 2*a[0,:]-a[1,:]
  
def fromJustin( y0 ):
  def justPar( y0 ):
    # Verbatim copy of params from Justin's code
    m = 2.5e-3; 
    g = 9.81; 
    lo = 0.024; 
    k = 15; 
    c = 0.06; 
    mu = 1000; 
    # =  [tc;     ts/tc;  phs;    ph0;    Kp;     Kd]
    cp = [0.104,  0.51,   0.76,   0+0.25,   1/40.0,     0];
    # Conversion into our python format
    y0.par.set(
      hexed = cp,
      mass = m,
      len0 = lo,
      stK = k,
      stMu = c,
      gravityX = 0,
      gravityZ = -g
    )
  justPar(y0)
  y0.set(
    com_z = 0.023,
    com_x = 0,
    com_vz = 0,
    com_vx = 0.2,
    clk = pi/2,
  ).par.set(
    flKp = 1000,
    flKd = 1,
  )
  y0.mdl.prepare(2000)
  y0.mdl.setDomain(DomName['flyDown'])
  cts.domDt=[0.001]*5
  figure(1); clf();
  psweep( y0, 'stK', [y0.par.stK], visXZ )

def showModelTq( y0 ):
  y0.update()
  y0.par.update()
  x = y0.mdl.x
  l1 = sqrt(x[:,s.leg_z_1]**2+x[:,s.leg_x_1]**2)
  l0 = sqrt(x[:,s.leg_z_0]**2+x[:,s.leg_x_0]**2)
  a0 = numpy.arctan2(x[:,s.leg_z_0],x[:,s.leg_x_0])
  a1 = numpy.arctan2(x[:,s.leg_z_1],x[:,s.leg_x_1])
  ux0 = x[:,s.leg_x_0]/l0
  uz0 = x[:,s.leg_z_0]/l0
  ux1 = x[:,s.leg_x_1]/l1
  uz1 = x[:,s.leg_z_1]/l1
  figure(2);clf();
  subplot(311);
  plot(y0.par.stK*(y0.par.len0-l1),'o-g')
  plot(y0.par.stK*(y0.par.len0-l0),'o-r')
  plot(y0.par.tqKth*sin(x[:,s.theta_1]-a1)/l1,'o-c')
  plot(y0.par.tqKth*sin(x[:,s.theta_0]-a0)/l0,'o-m')
  ylabel('Force (mag, N)')
  subplot(312);
  plot(y0.par.stK*(y0.par.len0-l1)*ux1,'o-g')
  plot(y0.par.stK*(y0.par.len0-l0)*ux0,'o-r')
  plot(y0.par.tqKth*sin(x[:,s.theta_1]-a1)/l1*uz1,'o-c')
  plot(y0.par.tqKth*sin(x[:,s.theta_0]-a0)/l0*uz0,'o-m')
  ylabel('Force X (N)')
  subplot(313);
  plot(y0.par.stK*(y0.par.len0-l1)*uz1,'o-g')
  plot(y0.par.stK*(y0.par.len0-l0)*uz0,'o-r')
  plot(y0.par.tqKth*sin(x[:,s.theta_1]-a1)/l1*ux1,'o-c')
  plot(y0.par.tqKth*sin(x[:,s.theta_0]-a0)/l0*ux0,'o-m')
  ylabel('Force Z (N)')

def setCockroach(y0):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.02262789548, #0.023,
    com_vx = 0.231001422112, #0.2,
    com_vz = 0,
    clk = -2.05201652, #1.57079632679 
  ).par.set(
    mass = 0.0025,
    len0 = 0.024,
    stK = 15,
    stMu = 0.06,
    stNu = 0,
    tqKth = 0.025,
    tqKxi = 0,
    tq0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = -60.4152433383,
    stHalfDuty = 1.60221225333,
    stHalfSweep = 0.38,
    stOfs = -1.32079632679,
    gravityX = 0,
    gravityZ = -9.81,
    xiLiftoff = 1.45399322,
    xiUpdate = 0.0,
    maxZ = 0.1,
    minZ = 0.003
  )
  y0.mdl.setDomain(DomName['flyDown'])

def setC( y0 ):
  setCockroach(y0)
  y0.set(
    com_z = 0.022,
    com_vz = -0.077,
    com_vx = 0.3426,
    clk = 1.581,
    leg_x_0 = 0,
    leg_z_0 = 0,
    leg_x_1 = 0,
    leg_z_1 = 0,    
  ).par.set(
    tqKth = 0.01,
    stMu = 0.1,
    omega = -61,
    stHalfDuty = 2.5,
    stHalfSweep = 0.6,
    stOfs = -pi/2,
  )
  y0.mdl.setDomain(DomName.both)
    
def setK0(y0):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.00451,
    com_vx = 0.172,
    com_vz = 0,
    clk = -1.339,#-pi/2, 
    # Use ref position
    leg_x_0 = 0,
    leg_z_0 = 0,
    leg_x_1 = 0,
    leg_z_1 = 0,
  ).par.set(
    mass = 0.006,
    len0 = 0.0033,
    stK = 700,
    stMu = 0.15,
    stNu = 0,
    tqKth = 5e-3,
    tqKxi = 0,
    tq0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = -75,
    stHalfDuty = pi/4,
    stHalfSweep = 0.5,
    stOfs = -1.657,#-pi/2, #-1.13
    gravityX = 0,
    gravityZ = -9.81,
    xiLiftoff = 3.023,
    xiUpdate = 0,#1.0
    maxZ = 0.05,
    minZ = 0.0005
  )
  y0.mdl.domDt=[0.001]*5
  y0.mdl.setDomain(DomName['flyDown'])

def setK1(y0):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.0045,
    com_vx = 0.174,
    com_vz = 0,
    clk = -1.48,#-pi/2, 
    # Use ref position
    leg_x_0 = 0,
    leg_z_0 = 0,
    leg_x_1 = 0,
    leg_z_1 = 0,
  ).par.set(
    mass = 0.006,
    len0 = 0.0033,
    stK = 700,
    stMu = 0.15,
    stNu = 0,
    tqKth = 4.5e-3,
    tqKxi = 0,
    tq0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = -75,
    stHalfDuty = pi/2,
    stHalfSweep = 1,
    stOfs = -pi/2, #-1.13
    gravityX = 0,
    gravityZ = -9.81,
    xiLiftoff = 2.88315,
    xiUpdate = 0.2,#1.0
  )
  y0.mdl.domDt=[0.001]*5
  y0.mdl.setDomain(DomName['flyDown'])

def setK(y0):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.0052,
    com_vx = 0.135,
    com_vz = 0,
    clk = 1.53,#-pi/2, 
    # Use ref position
    leg_x_0 = 0,
    leg_z_0 = 0,
    leg_x_1 = 0,
    leg_z_1 = 0,
  ).par.set(
    mass = 0.0045,
    len0 = 0.005,
    stK = 150,
    stMu = 0.345,
    stNu = 0,
    tqKth = 3e-4,
    tqKxi = 0,
    tq0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = -81,
    stHalfDuty = pi/2,
    stHalfSweep = 1.13,
    stOfs = -pi/2, #-1.13
    gravityX = 0,
    gravityZ = -9.81,
    xiLiftoff = 2.25,
    xiUpdate = 0,#0.2,#1.0
  )
  y0.mdl.domDt=[0.001]*5
  y0.mdl.setDomain(DomName['flyDown'])

def setL(y0):
  setK(y0)
  y0.par.xiUpdate = 1.0

def setClockRoach(y0):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.02262789548, #0.023,
    com_vx = 0.231001422112, #0.2,
    com_vz = 0,
    clk = -2.05201652, #1.57079632679 
  ).par.set(
    mass = 0.0025,
    len0 = 0.024,
    stK = 15,
    stMu = 0.06,
    stNu = 0,
    tqKth = 0,
    tqKxi = 1.33e-3,
    tq0 = 0.02e-3,
    xi0 = -0.667,
    tqOmRatio = 2.6,
    flKp = 1000,
    flKd = 1,
    omega = -60.4152433383,
    stHalfDuty = 1.60221225333,
    stHalfSweep = 0.38,
    stOfs = -1.32079632679,
    gravityX = 0,
    gravityZ = -9.81,
    xiLiftoff = 1.45399322,
  )
  y0.mdl.setDomain(DomName['flyDown'])
  
s = CTSLIP_xst()

def newSimOf( N ):
  y0 = CTSLIP_state()
  y0.fromArray([0]*len(y0))
  y0.par = CTSLIP_param()
  y0.par.fromArray([0]*len(y0.par))
  y0.mdl = CTS()
  y0.mdl.prepare(N)
  return y0

y0 = newSimOf(5000)  
setC(y0)
y0.mdl.toApex = False
#y0.mdl.toApex = True
y0.mdl.setMapIO( 
    idxIn = [s.com_z,s.com_vx],
    idxOut = [s.com_z,s.com_vx,s.clk]
    )
y0.mdl.setICS(y0).reset().integrate()
cm = y0.mdl
 
def assay( prepFunc, roi, scl=0.001 ):
  pre = [0.0]*20
  stim = [ ('Flat', pre),
    ('Bump' , pre + [scl,scl,0]),
    ('Step' , pre + [scl]),
    ('Ramp' , pre + list(arange(0,scl*30,scl/2.0)) )
  ]
  res = []
  for key,ground in stim:
    y0 = newSimOf(10000)
    y0.simName = key
    prepFunc(y0)
    y0.mdl.setICS(y0).reset().setGround(ground).integrate()
    xi = y0.mdl.sampleAt(roi)
    y0.simRes = numpy.take( y0.mdl.y, xi, 0 )
    y0.simPhi = phest( y0.simRes )
    res.append( y0 )
  return res

def pltPhi(A):
  B = scipy.signal.butter(3,0.02)
  s = CTSLIP_xst()
  def P(A,k):
    return plot( A[0].simRes[:,s.t],
        (scipy.signal.lfilter( B[0],B[1],
        (A[k].simPhi - A[0].simPhi ))))
  for k in [1,2,3]:
    subplot(310+k)
    P(A,k)
    ax = axis()
    axis([A[0].simRes[0,s.t],A[0].simRes[-1,s.t]]+ax[2:])
    ylabel(A[k].simName)
    grid(True)
  xlabel('time')

def pltClk(ass):
  B = scipy.signal.butter(4,0.2)
  s = CTSLIP_xst()
  def P(A,k):
    return plot( A[0].simRes[:,s.t],
        scipy.signal.lfilter( B[0],B[1],
        wrap(A[k].simRes[:,s.clk] - A[0].simRes[:,s.clk] )))
  return [ P(ass,1), P(ass,2), P(ass,3) ]

def anim( mdl, ax, rng, fmt=0.1 ):
  if rng[-1]>mdl.csr:
    raise IndexError, "Sampled outside of intergation range"
  s = CTSLIP_xst()
  xi = mdl.sampleAt(rng)
  x = numpy.take( mdl.y, arange(xi[0],xi[-1]), 0 )
  cx = x[:,s.com_x]
  cz = x[:,s.com_z] + x[:,s.zOfs]
  for ki in xrange(len(xi)):
    k = xi[ki]-xi[0]
    clf()
    plot(cx,x[:,s.zOfs],color=[0.3,0.3,0],linewidth=2)
    plot(cx[:k]+x[:k,s.leg_x_0],cz[:k]+x[:k,s.leg_z_0],color=[0.5,0.5,1])
    plot(cx[:k]+x[:k,s.leg_x_1],cz[:k]+x[:k,s.leg_z_1],color=[1,0.5,0.5])
    plot(cx,cz,color=[0.5,0.5,0.5])
    plot(cx[:k],cz[:k],color=[0.6,0.6,0.6],linewidth=3)
    mdl.plotAt([xi[ki]])
    axis(ax)
    if type(ax)==tuple:
      axis('equal')
    mdl.axisPan(xi[ki])
    if type(fmt)==str:
      savefig(fmt % (ki-1))
      print k,
    elif fmt>0:
      draw()
      time.sleep(fmt)
    else:
      show()
      draw()
    
"""  
t = linspace(0.5,2,2000)
S = setL
a = [
    assay(S, t, 0.00005),  
    assay(S, t, 0.0001),  
    assay(S, t, 0.00015),   
    assay(S, t, 0.0002)
]
clf()
[ pltPhi(x) for x in a ]

"""
#rng = arange(2.327786,2.638,0.001)
rng = arange(0.3,0.75,0.001)
ax = [0.37, 0.385, -0.001, 0.015*3/4-0.001]
ax = [0,0.06,-0.001,0.05]

def visTAZ( mdl ):
  s = CTSLIP_xst()
  xi = mdl.sampleAt( linspace(mdl.x[0,s.t],mdl.x[-1,s.t],3000) )
  x = numpy.take( mdl.y, xi, 0 )
  az = diff(x[:,s.com_vz]) / diff(x[:,s.t])
  plot( (x[1:,s.t]+x[:-1,s.t])/2, az )

"""
ass = assay(setCockroach, linspace(0.7,1.7,1000))
plot( ass[1].simPhi - ass[0].simPhi )
plot( ass[2].simPhi - ass[0].simPhi )
plot( ass[3].simPhi - ass[0].simPhi )
"""

# :mode=python:folding=indent:
