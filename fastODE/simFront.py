import util
reload(util)

execfile('sim2010.py')

def PLT(cts, thr=9):
  N = cts.csr
  y = COPY(cts.y[:N,:])
  plot([min(y[:,s.com_x]),max(y[:,s.com_x])],[0,0],color=[0.3,0.3,0],linewidth=2)
  plot(y[:,s.com_x]+y[:,s.leg_x_0],y[:,s.com_z]+y[:,s.leg_z_0],'b')
  plot(y[:,s.com_x]+y[:,s.leg_x_1],y[:,s.com_z]+y[:,s.leg_z_1],'r')
  L = plot(y[:,s.com_x],y[:,s.com_z],'k')
  cts.plotAt([t[0] for t in cts.getEvents() if t[0]<N ])
  return L
  #cts.plotAt(cts.sampleAt(linspace(cts.y[0,0],cts.y[cts.csr-1,0],8)))


def visXZ(cm):
  y = COPY(cm.x)
  plot(y[:,s.com_x],y[:,s.zOfs],'k')
  res = plot(y[:,s.com_x],y[:,s.com_z]+y[:,s.zOfs])
  xlabel('X (m)')
  ylabel('Z (m)')
  return res


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

def Jaccd( func, scl ):
  scl = asarray(scl).flatten()
  N = len(scl)  
  d0 = identity(N)
  plan = zip(scl,identity(N))
  def centralDifferenceJacobian( arg ):
    x0 = asarray(arg).flatten()
    return asarray([ 
      (func(x0+dx*s)-func(x0-dx*s))/(2*s) 
      for s,dx in plan 
    ]).T
  return centralDifferenceJacobian

def Jaccdas( func, scl, lint=0.8, tol=1e-12, eps = 1e-30, withScl = False ):
  scl = asarray(scl).flatten()
  N = len(scl)  
  def centDiffJacAutoScl( arg ):
    """
    Algorithm: use the value of the function at the center point
      to test linearity of the function. Linearity is tested by 
      taking dy+ and dy- for each dx, and ensuring that they
      satisfy lint<|dy+|/|dy-|<1/lint
    """
    x0 = asarray(arg).flatten()    
    y0 = func(x0)
    s = scl.copy()
    print "Jac at ",x0
    idx = slice(None)
    dyp = empty((len(s),len(y0)),x0.dtype)
    dyn = empty_like(dyp)
    while True:
      print "Jac iter ",s
      d0 = diag(s)
      dyp[idx,:] = [ func(x0+dx)-y0 for dx in d0[idx,:] ]
      dypc = dyp.conj()
      dyn[idx,:] = [ func(x0-dx)-y0 for dx in d0[idx,:] ]
      dync = dyn.conj()      
      dp = sum(dyp * dypc,axis=1)
      dn = sum(dyn * dync,axis=1)
      nul = (dp == 0) | (dn == 0)
      if any(nul):
        s[nul] *= 1.5
        continue
      rat = dp/(dn+eps)
      nl = ((rat<lint) | (rat>(1.0/lint)))
      # If no linearity violations found --> done
      if ~any(nl):
        break
      # otherwise -- decrease steps
      idx, = nl.flatten().nonzero()
      s[idx] *= 0.75
      if any(s[idx])<tol:
        break;          
    res = ((dyp-dyn)/(2*s[:,newaxis])).T
    if withScl:
      return res, s
    return res
  return centDiffJacAutoScl 

def Jacrnd( func, scl, mul=10 ):
  scl = asarray(scl).flatten()
  N = len(scl)  
  dx = randn(N*mul, N) * scl[newaxis,:]
  dx = r_[dx,-dx]
  def randomizedJacobian( arg ):
    x0 = asarray(arg).flatten()[newaxis,:] + dx
    y0 = array([func(x) for x in x0])
    dy = y0 - mean(y0,axis=0)[newaxis,:]
    jact = scipy.linalg.lstsq( dx, dy )[0]
    return jact.T
  return randomizedJacobian
  
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

def meshgridN( *args  ):
  """
  Create a meshgrid of arbitrary dimension
  
    (g_1,g_2,..., g_N) = meshgridN( a_1, a_2, ..., a_N )
    
  makes g_k arrays of rank N and equal size, such that
  g_k varies only along axis k and takes on values a_k
  """  
  ax = [ asarray(q).flatten() for q in args ]
  N = len(ax)
  z = zeros(tuple([ len(a) for a in ax ]))  
  for k in xrange(N):
    d = len(ax[k])
    ax[k].shape = (1,)*k + (d,) + (1,)*(N-k-1)
    ax[k] = ax[k] + z
  return ax

def cartesian( *args ):
  """
  Return array whose rows are the cartesian product of sequences
  passed as parameters  
  """
  return concatenate( [
    x.flatten()[:,newaxis] 
    for x in meshgridN(*args)
  ],axis=1)
  
import optutil
reload(optutil)


if 0:
  import IPython.kernel.client
  mec = IPython.kernel.client.MultiEngineClient()
  mec.activate()
  
  ics = zeros((200,3))
  ics[:,2] = linspace(-0.5,0.5,200)
  ics[:,:len(m.ic0)] += m.ic0
  
  mec.scatter('ics', ics)
  mec.execute( 'res = apexOpt(ics)' )
  res = mec.gather('res')

def visXE(cm):
  y = cm.x.copy()
  E = cm.energy()
  plot(y[:,s.com_x],E[:,0])
  res = plot(y[:,s.com_x],E[:,1])
  xlabel('X (m)')
  ylabel('Energy (Joul)')
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
  
def anim( mdl, ax, rng, fps=10, fmt=0.1 ):
  fig = gcf()
  ani = util.Animation(fmt, fps=fps)
  if rng[-1]>mdl.csr:
    raise IndexError, "Sampled outside of intergation range"
  s = CTSLIP_xst()
  xi = mdl.sampleAt(rng)
  x = numpy.take( mdl.y, arange(xi[0],xi[-1]), 0 )
  cx = x[:,s.com_x]
  cz = x[:,s.com_z] + x[:,s.zOfs]
  for ki in xrange(len(xi)):
    k = xi[ki]-xi[0]
    fig.clf()
    plot(cx,x[:,s.zOfs],color=[0.3,0.3,0],linewidth=2)
    plot(cx[:k]+x[:k,s.leg_x_0],cz[:k]+x[:k,s.leg_z_0],color=[0.5,0.5,1])
    plot(cx[:k]+x[:k,s.leg_x_1],cz[:k]+x[:k,s.leg_z_1],color=[1,0.5,0.5])
    plot(cx,cz,color=[0.5,0.5,0.5])
    plot(cx[:k],cz[:k],color=[0.6,0.6,0.6],linewidth=3)
    mdl.plotAt([xi[ki]])
    axis(ax)
    mdl.axisPan(xi[ki])
    ani.step()
  ani.stop()
  
def stab(q = None,par = None, scl=1e-5, retall=False):
  idx = [s.clk,s.com_vx]
  m.setMapIO( idxIn = idx, idxOut = idx )
  m.toApex = True
  if par is None:
    pass
  elif par=='com_z':
    y0.com_z = q
  else:
    setattr( y0.par, par, float(q) ) 
  m.setICS(y0).reset()
  fm = optutil.fmin( m.apexError, m.y[0,m.icsMap], xtol=scl, ftol=scl )
  opt = list(fm)+[m.y[0,s.com_z]]
  idx = [s.clk,s.com_vx,s.com_z]
  assert len(opt)==len(idx)
  m.setMapIO( idxIn = idx, idxOut = idx )
  mJac = Jaccd( m.mapping, [scl]*len(idx) )  
  #ics =m.y[0,m.icsMap] 
  print q,"Fixed point is at:",opt,"-->",m.mapping(opt)+[pi,0,0]
  DF = mJac(opt)
  egs, egv = eig(DF)
  #print q,"Eigenvalues:",list(egs.round(2))
  print q,"Abs eig:",",".join(["%.2g" % x for x in sorted(abs(egs))])
  if retall:
    return DF
  # "special" error metric for 
  mx = max(abs(egs))
  if mx>1:
    res = mx+len(egs)
  else:
    res = mx+sum(abs(egs.imag))
  print q,"ret:",res
  return res

def cleanOuts( q, sd=4 ):
  dq = diff(q,axis=0)
  bad = abs(dq) > (std(dq,axis=0)*sd)[newaxis,:]
  b = zeros(q.shape,bool)
  b[:-1,:]=bad
  b[1:,:]|=bad
  b[0,:] = False
  b[-1,:] = False
  p = zeros_like(q)
  kk = arange(q.shape[0])
  for k in xrange(q.shape[1]):
    p[:,k] = interp( kk, find(~b[:,k]), q[~b[:,k],k] )
    p[isnan(p[:,k]),k] = 0
  return p

def uniStep( mdl, N=512, slc=slice(None) ):
  """Note: assumes integration was already done"""
  st,en,_ = slc.indices(len(mdl.x))
  t = linspace(mdl.x[st,s.t],mdl.x[en,s.t],N)
  mdl.x[slc,s.clk] = unwrap( mdl.x[slc,s.clk] )
  rng = t[0],t[-1]
  x = [
    interp( t, mdl.x[slc,s.t], mdl.x[slc,k] )
    for k in xrange(mdl.x.shape[1])
  ]
  x.append(linspace(0,2*pi,N))
  a = asarray(x).T
  a[1:,s.clk] = diff(a[:,s.clk])
  a[0,s.clk] = a[1,s.clk] 
  return a

def real_floqScaleEig( eigs, t ):
  eigs = asarray(eigs)
  t = asarray(t)
  tshp = t.shape
  t = t.flatten()
  if eigs.ndim==2: # If (block) diagonal real eigenvalue matrix
    # --> convert to complex form
    eigs = (diag(eigs) 
      + 1j * concatenate([ diag(eigs,1), [0] ] )
      - 1j * concatenate([ [0], diag(eigs,1) ] )
    )
  r = []
  for ev in eigs:
    sc = exp(-log(abs(ev))*t)
    if ev.imag == 0:
      d = sc
      if ev.real<0:
        d *= cos(t*pi)
    else:
      ang = angle(ev)
      d = sc*cos(ang*t)+1j*sc*sin(ang*t)
    r.append(d)
  r=asarray(r)
  res = array([ diag(ri.real) 
    + diag( ri[1:].imag*(ri[1:].imag>0),1) 
    + diag(-ri[1:].imag*(ri[1:].imag>0),-1)
    for ri in r.T
    ])
  res.shape = tshp+(len(eigs),len(eigs))
  return res.squeeze()

def contStep( y0, ic = None ):
  if ic is not None:
    y0[idx] = ic
  m.setICS(y0).reset().integrate()
  print m.x[0,idx]
  print m.x[-1,idx]+([pi]+[0]*(len(idx)-1))
  aa = m.mapseq()
  clf()  
  dx = c_[ 
    (aa[:,s.clk] % pi- aa[-1,s.clk] % pi),
    aa[:,idx[1:]]-aa[-1,idx[1:]][newaxis,:]
  ]
  if dx.shape[0]<30:
    if raw_input("trunc?")[:1]=='x':
      return None
    else:
      print "==> !!! truncated run"
      return ic
  semilogy(abs(dx))
  draw()
  if max(abs(dx[-30,:])> 1e-4) and (raw_input("converge?")[:1]=='x'):
    return None
  ic = aa[-1,idx]
  print "==> ", ic
  y0[idx] = ic
  return ic

def contOmega(dx = -0.2):
  res = []
  while True:
    fp = contStep( y0 )
    if fp is None:
      break
    res.append( [y0.par.omega]+list(fp) )
    y0.par.omega = y0.par.omega + dx
  return array(res)

def contAnyPar(par, dx):
  res = []
  pidx = y0.par.Fields.index(par)
  while True:
    fp = contStep( y0 )
    if fp is None:
      break
    res.append( [float(y0.par[[pidx]])]+list(fp) )
    y0.par[[pidx]] += dx
  return array(res)

def contEigsAnyPar( par, ocd, scl=1e-3, Jac = Jaccd ):
  pidx = y0.par.Fields.index(par)
  m.setMapIO( idxIn = idx, idxOut = idx )
  m.toApex = True
  y0.par[[pidx]] = [ocd[0]]
  y0[idx] = ocd[1:]
  mJac = Jac( m.mapping, [scl]*len(idx) )
  opt = ocd[1:]   
  #ics =m.y[0,m.icsMap] 
  DF = mJac(opt)
  egs, egv = eig(DF)
  return egs


### Set up model and ensure that we have its fixed point

setCockroach2(y0)
# Integrate a single step
m.toApex = 1
m.setICS(y0).reset().integrate()
# overall collection of state variables
sidx = [s.clk,s.com_vx,s.com_z,s.com_vz,s.t]
# Set Poincare map coordinates for apex
idx = sidx[:3]
#assert allclose( m.x[0,idx], m.x[-1,idx]+[pi,0,0],atol=1e-3 ),"Must be a fixed point"
if 0:
  #y0.plane = array([ 1.08950759e+00,   2.30999554e-01,   2.26752473e-02,
  #      -3.38345439e-07, 6.69391014e-05,  -4.96596556e-05,  -7.11933162e-05,
  #       6.74955521e-07])/2+array([-2.6677669117059519,
  # 3.1891465126555222e-05,
  # 0.76050167471271579,
  # 110.13582669234054])
  y0.plane = [  3.81297972e-05,   9.78993350e-01,   1.33373945e+02,
        -6.66514066e-01,   0.00000000e+00]
  m.toApex = 0
  m.planeCounter = 100
  y0[sidx] = array([  1.08950759e+00,   2.30999554e-01,   2.26752473e-02,
        -3.38345439e-07,   0.00000000e+00])
  m.setICS(y0).reset().integrate()
  m.setICS(y0).reset().integrate()
  plot( m.x[:,s.t], y0.planeVal(m.x), '.-' )
  slc = asarray([ ev[0] for ev in m.getEvents() 
    if ev[2] == EvtName.plane ])
  plot( m.x[slc,s.t], y0.planeVal(m.x[slc,:]), 'dk' )
  raise "arf"
### Remember initial conditions

if 0: # half-sweep gait
  y0.par.stHalfSweep = 0.54 
  y0[idx] = [-1.98322111,  0.36504787,  0.02260134]
ics = y0[idx]
eics = array(list(ics)+[0])


### Use Jaccass to bound scales at which mapping is linear

def mapAndShow(*argv):
  res = m.mapping(*argv)
  visXZ(m)
  return res
m.setMapIO( idxIn = sidx[:-1], idxOut = sidx )
m.setICS(y0).reset().integrate()
D1, scl = Jaccdas(mapAndShow, [1e-4]*len(eics), withScl = 1 )(eics)

### Specify section plane as apex
y0.plane = [-1e-5, 0, 0, 0, -1]
m.setICS(y0).reset().integrate()
m.setICS(y0).reset().integrate()

### Compute a section to section map over random initial conditions
r = randn(50,len(eics)) * scl[newaxis,...]/2
r = r_[r,-r]
x,y,d = map(array,m.sectionmap( 
  eics[newaxis,:] + r , 
  2, preslc=slice(-2,-1),postslc=slice(-1,None) 
))

v0 = d[:,sidx] / d[:,s.t][:,newaxis]
v = mean(v0,axis=0)
print "v = ",v
print " +/-",std(v0,axis=0)

### Compute map jacobian from section to section

dx0 = x[:,sidx[:-1]] - eics

yct = mean(y[:,sidx[-1]],axis=0); 
dy0 = y[:,sidx] - concatenate( (eics,[yct]) )
#yc = mean(y[:,sidx],axis=0); 
#dy0 = y[:,sidx] - yc

Ux,sx,Vx = svd(dx0,full_matrices=0)
Rx = Vx[:-1,:].T
dx0t = dot( dx0, Rx )

Uy,sy,Vy = svd(dy0,full_matrices=0) 
Ry = Vy[:-1,:].T
dy0t = dot( dy0, Ry )

D2t = scipy.linalg.lstsq( dx0t, dy0t )[0]
D2 = dot( dot( Rx, D2t ), Ry.T )

### Compute normal to intersection of isochron and section 
tau = D2[:,-1]

# Compute normal to isochron based on the fact that orbit velocity
#   has a projection of 1 on this normal
# dot(niso,v) = dot(tau,v[:-1]) + foo*v[-1] = 1 
#  --> foo = (1-dot(tau,v[:-1]))/v[-1]
niso = concatenate((tau,((1-dot(tau,v[:-1]))/v[-1],)))
print "Arrival time std-dev: ", std(dy0[:,-1])
print "     reduction ratio: ", std(dot(dx0,niso[:-1]) - dy0[:,-1]) / std(dy0[:,-1])

### Place section on isochron

def goalIsoch( delta ):
  # Compute section value at initial condition, which is also a transition
  # point for the new section
  y0.plane = [-dot(eics,delta+niso[:-1])*10] + list((delta+niso[:-1])*10)
  m.toApex = 1
  m.setICS(y0).reset().integrate()
  m.setICS(y0).reset().integrate()
  m.toApex = 0
  
  #figure(2); clf()
  xi,yi,di = map(array,m.sectionmap( 
    eics[newaxis,:] + r/2, 
    3, preslc=slice(0,1),postslc=slice(1,2), skip=20 
  ))
  #draw()
  #time.sleep(0.1)
  vt =  std(yi[:,s.t] - xi[:,s.t])
  print std(dot(xi[:,sidx[:-1]],delta+niso[:-1])), vt, xi.shape
  return vt  

try:
  print "Using cached niso2", niso2
except NameError:
  from scipy.optimize import fmin as fMin
  delta = fMin( goalIsoch, [0,0,0,0] )  
  niso2 = delta+niso[:-1]
  
y0.plane = [-dot(eics,niso2)] + list(niso2)
m.toApex = 1
m.setICS(y0).reset().integrate()
m.setICS(y0).reset().integrate()
m.toApex = 0
xi,yi,di = map(array,m.sectionmap( 
    eics[newaxis,:] + r/10, 
    3, preslc=slice(0,1),postslc=slice(1,2), skip=20 
  ))
tt = y[:,s.t] - x[:,s.t]
tti = yi[:,s.t] - xi[:,s.t]

v0i = di[:,sidx] / di[:,s.t][:,newaxis]
vi = mean(v0i,axis=0)
print "vi = ",vi[:-1]
print " +/-",std(v0i[:-1],axis=0)

print "v = ",v[:-1]
print " +/-",std(v0[:-1],axis=0)

print "Time variance was ",std(tt)," improved by ",std(tti)/std(tt)

## Mapping from isochron to isochron

def fun( dx, pc=5 ):
  dx = asarray(dx)
  if dx.ndim>1:
    return array([ fun(dxi) for dxi in dx ])
  # prj = niso2 * dot(niso2,dx) / sum(niso2*niso2)
  m.planeCounter=pc
  m.toApex = 0
  m.mapping( dx + eics ) #- prj )
  slc = [ ev[3] 
      for ev in m.getEvents() 
        if ev[2] == EvtName.plane 
           and abs(ev[3][s.com_vz])<0.04
           and ev[3][s.t]>0.01
  ]
  y = slc[0][sidx]
  dy = y[:-1] - eics
  dy[0] = (dy[0] + pi/2)% pi - pi/2
  return dy

## Refine equilibrium

# TODO refinement, sampling are iffy
q = [0,0,0,0]
ql = []
for k in xrange(100):
  try:
    q = fun(q)
  except IndexError:
    q = q + randn(*q.shape)*sqrt(len(q)) * norm(q) / 100
  ql.append(q)
  raise "barf"
ql = array(ql)
semilogy(abs(ql-ql[-1,:]))
title('Improvements in fixed point')
eics[:] = eics + q

## Get new Jacobian
D3 = Jaccd( fun, [1e-7]*4 )([0,0,0,0])

## Validate new Jacobian
rr = randn(100,4) * 3e-7
qq = fun(rr).T
qq0 = dot(D3,rr.T)

print "Variance reduction from jacobian", mean(std( qq-qq0, 0 ) / std(qq,0))

if 0: ## regression for section map jacobian  
  dx0i = xi[:,sidx[:-1]] - eics
  
  dy0i = yi[:,sidx[:-1]] - eics
  #yc = mean(y[:,sidx],axis=0); 
  #dy0 = y[:,sidx] - yc
  
  Uxi,sxi,Vxi = svd(dx0i,full_matrices=0)
  Rxi = Vxi[:-1,:].T
  dx0ti = dot( dx0i, Rxi )
  
  Uyi,syi,Vyi = svd(dy0i,full_matrices=0) 
  Ryi = Vyi[:-1,:].T
  dy0ti = dot( dy0i, Ryi )
  
  D2ti = scipy.linalg.lstsq( dx0ti, dy0ti )[0]
  D3 = dot( dot( Rxi, D2ti ), Ryi.T )

if 0:
  # Run for a while to obtain niso for apex states
  aa = m.mapseq()
  n = dot(aa[:,sidx[:-1]],niso)
  y0.par.co_clk, y0.par.co_com_vx, y0.par.co_com_z, y0.par.co_com_vz = niso
  y0.par.co_value = -prctile( n, 25 )

if 0:
  m.setMapIO( idxIn = sidx[:-1], idxOut = tidx )
  x0 = y0[sidx[:-1]][newaxis,:]
  def fun( x ):
    xiso = dot(asarray(x),L.T)
    return asarray(m.mapping( (xiso + x0).squeeze() ))
  
  Df = Jaccdas( fun, [1e-4] * L.shape[1] )(zeros(L.shape[1]))
  dx0 = randn(200,L.shape[1])*1e-6
  dx1 = array(fun( dx0 )) - fun( zeros_like(L[0,:]) )[newaxis,:]
  dt = dx1[:,-1]
  print mean(dt),"+/-",std(dt),"from",std(dx1[:,-1])
  
  D = Df[:L.shape[1],:L.shape[1]]

  # Compute real eigenvalues and eigenvectors at fixed point
  egm,egv = util.real_eig( D )

# Compute real eigenvalues and eigenvectors at fixed point
D = D3
egm,egv = util.real_eig( D3 )

print "Eigenvectors mapped through function:",
for eg in egv.T:
  print fun(eg*1e-7)/(1e-7*eg)

## Obtain an equilibrium stride at uniform sampling
def sampleStride( m, eics ):
  m.planeCounter = 3
  m.planeTime = -1
  m.mapping( eics )
  # Get positions of plane events
  slc = asarray([ ev[0] for ev in m.getEvents() 
    if (ev[2] == EvtName.plane 
        and ev[1] in [DomName.flyUp, DomName.flyDown]
        and ev[0] > 20)
  ])
  return uniStep( m, slc=slice( slc[0], slc[1] ) )

u0 = sampleStride(m, eics)
mag = -1e-6 # Perturbation magnitudes
flm = [] # Floquet modes
for eg,ev0 in zip(egm,egv.T): # Loop on all real eigenvectors
#  ev = dot(L,ev0)
#  ev /= sign(ev[-1])
#  # Initialize perturbed state
#  m.mapping(x0 + ev*mag)
#  #y0[idx] = ics + ev*mag
#  #m.setICS(y0).reset().integrate()  
#  # Compute perturbation trajectory with initial = eigenvalue
#  uv = (uniStep(m) - u0)/mag
  ev = ev0
  ev /= sign(ev[-1])
  try:
    uv = (sampleStride(m, eics + ev*mag) - u0)/mag
  except IndexError:
    uv = (sampleStride(m, eics - ev*mag) - u0)/-mag
  # Time and phase are not relative -- they are absolute
  uv[:,s.t] = u0[:,s.t] + dot(ev*mag,niso2[:len(ev)])
  uv[:,-1] = u0[:,-1]
  # Store Floquet mode
  flm.append(uv[...,newaxis])

flm = concatenate(flm,axis=2)
t = u0[:,s.t]
# Compute rescaling / rotation needed to make Floquet coordinates periodic
sc = real_floqScaleEig( egm, linspace(0,1,len(t)) )

## Verify invariants
#assert allclose(dot(D,flm[0,idx,:]),flm[-1,idx,:],atol=4e-2),"Jacobian is a model of return map"
#assert allclose(D, dot( egv, dot( egm, inv( egv  )) )),"Eigenvalue decomposition is good"
#assert allclose(flm[0,idx,:],egv),"Initial state of modes is exactly eigenvector matrix"
#assert allclose( identity(egm.shape[0]), dot(egm,sc[-1,...])),"Scaling matrices iterpolate eigenvalue matrix inverse"

NM = flm.shape[2]
ND = len(sidx)-1
cflm = cleanOuts( flm[:,sidx[:-1],:]
    .reshape(flm.shape[0],NM*ND)
    , sd=1.5 )
cflm.shape = (flm.shape[0],ND,NM)

# Compute Floquet coordinate
flq = array([
  dot(q0,sc0) for q0,sc0 in zip(cflm,sc)
  ])
flq.shape = [flq.shape[0],prod(flq.shape[1:])]
print 'Coor mismatch (sd)',abs(flq[-1,...]-flq[0,...])/std(flq,axis=0)
cflq = flq + linspace(0,1,flq.shape[0])[:,newaxis] * (flq[0,:]-flq[-1,:])
# Build FourierSeries models of Floquet coordinates
th = linspace(-pi,pi,flq.shape[0])
flqs = util.FourierSeries().fit( 13, th, cflq.T )
th2 = linspace(-2,2,512) * pi
fc = flqs.val(th2).reshape(len(th2),ND,NM)

if 1: # modes
  f = figure(79, figsize=(8,6)); clf(); 
  sd =sqrt(sum(std(fc,axis=0)**2,axis=1))
  for k in xrange(4):
    subplot(4,1,1+k)
    plot( th2[[0,-1]]/(2*pi), [0,0], 'k', lw=2 )
    h = plot( th2/(2*pi), fc[:,k,:-1]/sd[k], lw=2 )
    grid(1)
    ylabel(s.Fields[sidx[k]])
    if k<3:
      gca().set(xticklabels=[])
    else:
      xlabel('phase (cycles)')
      legend(h,["%.2g%+.2gi" % (z.real,z.imag) for z in eigvals(D)[:len(h)]])
  #savefig('modes.png')
  #savefig('modes.pdf')

if 0:
  #for its in xrange(5):
    om0 = y0.par.omega
    om = -pi / m.seq[-1][3][s.t]
    print m.seq[1][3][s.clk]
    clk = m.seq[1][3][s.clk] - om*m.seq[1][3][s.t]
    swp = y0.par.stHalfSweep * om0/om
    y0.set( clk = clk )
    y0.par.set( omega = om, xiUpdate=1, xiLiftoff=2.4, tqKth=0.5, com_z=1.5, stHalfSweep=swp )
    stab()
    #m.setICS(y0).reset().integrate()
    visXZ(m)
 

#y0.par.set(stK = 1000, tqKth=0, xiUpdate=0, omega=-1.45, clk = 1.36 )
if 0: # scan for parameter effects
  res = []
  dq = 1e-3
  for q in xrange(len(y0.par)):
    if y0.par[[q]] == 0:
      continue
    egq = []
    p0 = y0.par[[q]]
    for sg in [-1,1]:
      idx = [s.clk,s.com_vx]
      m.setMapIO( idxIn = idx, idxOut = idx )
      m.toApex = True
      y0.par[[q]] = p0 + sg*dq 
      m.setICS(y0).reset()
      opt = optutil.fmin( m.apexError, m.y[0,m.icsMap] )
      idx = [s.clk,s.com_vx,s.com_z]
      m.setMapIO( idxIn = idx, idxOut = idx )
      mJac = Jaccd( m.mapping, [1e-5]*len(idx) )  
      ics =m.y[0,m.icsMap] 
      print q,"Fixed point is at:",ics,"-->",m.mapping(ics)+[pi,0,0]
      DF = mJac(ics)
      egs, egv = eig(DF)
      #print q,"Eigenvalues:",egs.round(2)
      print q,"Abs eig:",abs(egs),sum(abs(egs)**2)
      egq.append( egs )
    y0.par[[q]] = p0
    res.append( (y0.par.Fields[q],
      (abs(egq[-1])-abs(egq[0]))/(2*dq)) )
    print res[-1]
  r = array([r[1] for r in res])
  r2  = array(100 * r / r[:,0][:,newaxis], int)
  
if 0:# Run gait for a while
  m.toApex = False
  m.setICS(y0).reset().integrate()
  close('all')
  PLT(m)
  axis('equal')
  
  
if 0:
  setCockroach2(y0)
  y0.par.stHalfSweep = 0.54 
  y0[idx] = [-1.98322111,  0.36504787,  0.02260134]
  m.toApex = False
  m.setICS(y0).reset().integrate()
  #figure(100)
  #PLT(m)
  figure(103)
  visXZ(m)
  
  setCockroach2(y0)
  y0.par.stK = 21.5
  y0[idx] = [ 0.96522161,   0.29472212,   0.02318962]
  m.toApex = False
  m.setICS(y0).reset().integrate()
  #figure(101)
  #PLT(m)
  figure(103)
  visXZ(m)
  
if 0:
  import util,cPickle
  contData =  cPickle.load(open('kContData.pckl','r'))
  par = 'stK'
  if 1:
    egs1 = array([(m.setICS(y0).reset().integrate(),
       contEigsAnyPar( par, ocd, 1e-4, Jacrnd ))[-1] 
       for ocd in contData ])
  egs2 = array([(m.setICS(y0).reset().integrate(),
       contEigsAnyPar( par, ocd, 1e-3, Jaccdas ))[-1] 
       for ocd in contData ])
  if 1: 
    egs3 = array([(m.setICS(y0).reset().integrate(),
       contEigsAnyPar( par, ocd, 1e-3, Jaccd ))[-1] 
       for ocd in contData ])
    
if 0: # root locus animation
  figure(80)
  #A = util.Animation(0.1)
  A = util.Animation('cont-sweep-eig.avi',fps=10) #0.1)
  for k in xrange(egs1.shape[0]-7):
    cla();     axis('equal')
    plot( egs2.real, egs2.imag, 'g,', alpha=0.3)
    plot( egs1.real, egs1.imag, 'r,', alpha=0.3)
    plot( egs3.real, egs3.imag, 'b,', alpha=0.3)
    plot( egs1.real[k:k+7,:], egs1.imag[k:k+7,:], 'rs', alpha=0.5, ms=10  )
    plot( egs2.real[k:k+7,:], egs2.imag[k:k+7,:], 'go', alpha=0.5,ms=10   )
    plot( egs3.real[k:k+7,:], egs3.imag[k:k+7,:], 'bd', alpha=0.5, ms=10  )
    plotCircle( 0, 0, 1, steps = 360, color=[0,0,0], lw=2 )
    axis([-1.1,1.1,-1.1,1.1])
    grid(1)
    # title("%.2f Hz" % (abs(contData[k,0]/(2*pi))))
    title("%s = %.3f" % (par, contData[k,0]))
    #title("K = %.2f (N/m)" % (contData[k,0]))
    A.step()
  A.stop()
