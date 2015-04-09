execfile("tryCTSLIP.py")

def test_setDefaults( y0 ):
  """Set default values for some parameters"""
  y0.fromArray([0]*len(y0))
  y0.par = CTSLIP_param()
  y0.par.fromArray([0]*len(y0.par))
  y0.par.set(
    mass = 1,
    len0 = 1,
    flKp = 10000,
    flKd = 1,
  )
  
def test_axialNatFreq( y0 ):
  """test the natural frequency of axial compression"""
  y0.set(
    com_z = 1,
    com_x = 0,
    com_vz = -0.1,
    com_vx = 0,
    leg_z_0 = -1,
    clk = 0
  ).par.set(
    stK = 400,
    stMu = 0,
    stNu = 0,
    tqKth = 0,
    tqKxi = 0,
    xi0 = 0,
    omega = 0,
    stHalfDuty = 1,
    stHalfSweep = 1,       
    stOfs = pi/2,
    gravityZ = -9.81,
  )
  y0.mdl.setDomain(1)
  psweep( y0, 'stK', [y0.par.stK], visXZ )
  dt = mean(diff(numpy.array([ 
    xx[3][s.t] for xx in y0.mdl.seq if xx[1]==4 
  ])))
  nat = sqrt( y0.par.stK / y0.par.mass )
  print "Natural freq:",nat
  print "Measured:    ",2*pi/dt

def test_tqNatFreq( y0 ):
  """test the natural frequency of torque spring"""
  y0.set(
    com_z = 0.9,
    com_x = 0,
    com_vz = 0,
    com_vx = 0.1,
    leg_z_0 = -0.9,
    clk = 0
  ).par.set(
    stK = 49,
    stMu = 0,
    stNu = 0,
    tqKth = 400,
    tqKxi = 0,
    xi0 = 0,
    flKp = 10000,
    flKd = 1,
    omega = 0,
    stHalfDuty = 1,
    stHalfSweep = 1,
    stOfs = pi/2,
    gravityX = 0,
    gravityZ = -9.81,
  )
  y0.mdl.setDomain(1)
  y0.mdl.domDt=[0.1]*5
  psweep( y0, 'stK', [y0.par.stK], visXZ )
  az = phiOf( cts.x[0:-1:10,s.com_z] )
  ax = phiOf( cts.x[0:-1:10,s.com_x] )
  t = cts.x[-1,s.t]
  l = mean(sqrt(cts.x[:,s.com_z]**2+cts.x[:,s.com_x]**2))
  print "X natf:", sqrt(y0.par.tqKth/(y0.par.mass*l*l))/(2*pi)
  print "X freq:", (ax[-1]-ax[0])/(t*2*pi)
  print "Z natf:", sqrt(y0.par.stK/y0.par.mass)/(2*pi)
  print "Z freq:", (az[-1]-az[0])/(t*2*pi)

def test_sideToSide( y0 ):
  y0.set(
    com_z = 4,
    com_x = 0,
    com_vz = 0,
    com_vx = 0,
    clk = pi/2,
    leg_x_0 = 3,
    leg_z_0 = -4,
    leg_x_1 = -4.5,
    leg_z_1 = -4
  ).par.set(
    stK = 100,
    stMu = 0.1,
    stNu = 0,
    tqKth = 0,
    tqKxi = 0,
    xi0 = 0,
    flKp = 1000,
    flKd = 1,
    omega = 0,
    stHalfDuty = 0.7*pi,
    stHalfSweep = 0.3*pi,       
    stOfs = -pi/2,
    gravityZ = -9.81,
  )
  cts = y0.mdl
  cts.prepare(100000)
  cts.setDomain(DomName['both'])
  cts.domDt=[0.1,0.05,0.05,0.05,0.1]
  subplot(121)
  cts.setICS(y0).reset().integrate()
  plot(cts.x[:,s.com_x]+cts.x[:,s.leg_x_0],
       cts.x[:,s.com_z]+cts.x[:,s.leg_z_0],'.-g')
  plot(cts.x[:,s.com_x]+cts.x[:,s.leg_x_1],
       cts.x[:,s.com_z]+cts.x[:,s.leg_z_1],'.-c')
  subplot(122)
  visXE(cts)
