from numpy import *
from pylab import *

import optutil
reload(optutil)

execfile('ctslip.py')

def newSimOf( N ):
  y0 = CTSLIP_state()
  y0.fromArray([0]*len(y0))
  y0.par = CTSLIP_param()
  y0.par.fromArray([0]*len(y0.par))
  y0.mdl = CTS()
  y0.mdl.prepare(N)
  return y0

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
    minZ = 0.003,
    co_clk = 0,
    co_com_vz = 0,
    co_com_z = 0,
    co_com_vx = 0,
    co_com_x = 0,
    co_value = -1,
  )
  y0.mdl.setDomain(DomName['flyDown'])

def setCockroach2( y0 ):
  y0.set(
    t = 0,
    com_x = 0,
    com_z = 0.022627,
    com_vx = 0.231017,
    com_vz = 0,
    clk = 1.0895 )
  y0.par.set(
    mass = 0.0025,
    len0 = 0.024,
    stK = 15,
    stMu = 0.06,
    stNu = 0,
    tqKth = 0.025,
    tqKxi = 0,
    tqOmRatio = 0,
    xi0 = 0,
    tq0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = -60.4152433383,
    stHalfDuty = 1.60221225333,
    stHalfSweep = 0.38,
    stOfs = -1.32079632679,
    gravityX = 0,
    gravityZ = -9.81,
    xiUpdate = 0.0,
    xiLiftoff = 1.45399322,
    bumpW = 0,
    bumpZ0 = 0,
    bumpZ1 = 0,
    maxZ = 0.1,
    minZ = 0.003,
    co_clk = 0,
    co_com_vz = 0,
    co_com_z = 0,
    co_com_vx = 0,
    co_com_x = 0,
    co_value = -1,
    domain = 0 )
  y0.mdl.setDomain(DomName['flyDown'])
    
  
def setNDim( y0 ):
  y0.set(
      t=0,
      com_x=0,
      com_z=1.1,
      com_vx=2.,
      com_vz=0,
      clk=1.5)
  y0.par.set(
    mass=1, 
    len0=1, 
    stK=10, 
    stMu=0.05, 
    stNu=0, 
    tqKth=0.89,
    tqKxi=0, 
    tq0=0,
    flKp=1000,
    flKd=1,
    omega=-1,
    stHalfDuty=1,
    stHalfSweep=1,
    stOfs=-1.4,
    gravityX=0,
    gravityZ=-1, 
    maxZ = 4,
    minZ = 0.05,
    xiLiftoff=1.6,
    xiUpdate=0)
  y0.mdl.setDomain(DomName['flyDown'])

def stabNDim1( y0 ):
  setNDim(y0)
  y0.par.set(
    mass = 0.35,
    omega = -0.62,
    xiUpdate = 0.5,
    xiLiftoff = -0.1
  )
  y0.set(
    clk = 0.66,
    com_vx = 0.44,
    com_z = 1.55
    )

def stabNDim2( y0 ):
  stabNDim1( y0 )
  y0.par.set(
    tqKth = 40,
    stK = 3,
    xiLiftoff = -0.06
    )
  y0.set(
    clk = 0.83,
    com_vx = 0.57,
    com_z = 1.3
    )

def stabNDim3( y0 ):
  setNDim(y0)
  y0.set(
    com_z = 1.4017,
    com_vx = 0.5925,
    clk = -2.4642
  )
  y0.par.set(
    mass = 0.35,
    len0 = 1.0,
    stK = 10,
    stMu = 0.05,
    tqKth = 0.89,
    omega = -0.7,
    xiUpdate = 0.5,
    xiLiftoff = -0.1,
  )
    
def setHuman( y0 ):
  y0.set(
      t=0,
      com_x=0,
      com_z=0.84,
      com_vx=2.04,
      com_vz=0,
      clk=1.5)
  y0.par.set(
    mass=75, 
    len0=0.9, 
    stK=3000, 
    stMu=0.05, 
    stNu=0, 
    tqKth=0,
    tqKxi=0, 
    tq0=0,
    flKp=1000,
    flKd=1,
    omega=-18,
    stHalfDuty=1,#1.5,
    stHalfSweep=1,#0.4,
    stOfs=-1.4,
    gravityX=0,
    gravityZ=-9.81, 
    xiLiftoff=1.5,
    xiUpdate=0)
  y0.mdl.setDomain(DomName['flyDown'])

s = CTSLIP_xst()
y0 = newSimOf(8000)  
setNDim(y0)
m = y0.mdl
m.domDt = [0.1,0.05,0.05,0.05,0.1]
m.setMapIO(
  idxIn=[s.clk,s.com_vx,s.com_z], 
  idxOut=[s.clk,s.com_vx] )
m.ic0 = y0[m.rtnMap]

def apexOpt( ics, ic0=None ):
  """
  Minimize apexError starting with each of the specified initial conditions
  
  return array containing [err, ic, opt] for each ic
  
  NOTE: this is particularly useful when:
    len(m.outMap) == len(ic0) < len(ics[k]) <= len(m.icsMap)
    so the optimizer searches for optimum with the last entries of ics[k] held
    fixed. 
  """
  if not iterable(ics[0]):
    ics = [ics]
  if ic0 is None:
    ic0 = m.ic0
  res = []
  for ic in ics:
    m.setICS(y0).reset().useArgs( ic )
    opt = optutil.fmin( m.apexError, ic0, disp=False )
    err = m.apexError(opt)
    res.append([err] + list(ic) + list(opt))
  return asarray(res)
  
def apexRate( ics, ic0 = None, maxerr=1e-3 ):
  if not iterable(ics[0]):
    ics = [ics]
  if ic0 is None:
    ic0 = m.ic0
  res = []
  for ic in ics:
    lic = list(ic)
    lic[:len(ic0)] = ic0
    m.reset().useArgs( lic )    
    err = m.apexError(ic0)
    if err>maxerr:
      res.append([inf,err] + lic)
      continue      
    msq = m.mapseq(3)
    if len(msq)<3:
      res.append([inf,err] + lic)
      continue      
    d = diff(log(abs(diff(msq,axis=0))),axis=0)
    res.append([d,err] + lic)
  return res
  
if 1: # find fixed point
  idx = [s.clk,s.com_vx]
  m.setMapIO( idxIn = idx, idxOut = idx )
  m.setICS(y0).reset()
  #m.apexFixedPoint()
  #y0[m.icsMap]=m.getPre()
  if 1:
    res = []
    for z0 in linspace(1.3,1.7,32):
      y0.com_z = z0
      m.setICS(y0).reset()
      opt = optutil.fmin( m.apexError, m.y[0,m.icsMap] )
      err = m.apexError(opt)
      res.append([z0,err] + list(opt))
    res = asarray(res)

