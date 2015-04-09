from numpy import *
import numpy.linalg as linalg

def fixedPointRPM( func, ics, args=(), 
      xtol=1e-6, threshRatio = 1e-3, sRat=100, sMin=1e-6, slowRat = 0.9, disp=None 
  ):
    ics = asarray(ics).flatten()
    dim = ics.size
    eye = identity(dim)
    Z = zeros((dim,0))
    eyez = 1
    isNewDim = True    
    isSlow = False
    q=[ics]
    z=[0]
    u=[ics]
    Q = eye
    while True:
      # If isSlow convergence --> update projection
      if isSlow:
        # equation (4.4) pp. 1106
        dq = c_[q[-1]-q[-2],q[-2]-q[-3]]
        D,T = linalg.qr(dq)  
        print T
        if abs(T[1,1]/T[0,0]) > threshRatio:
          Z,_ = linalg.qr(c_[Z,D[:,1]])
          if disp:
            disp("Increasing basis by 1D")
        else:
          Z,_ = linalg.qr(c_[Z,D[:,[0,1]]])
          if disp:
            disp("Increasing basis by 2D")
        eyez = identity(Z.shape[1])
        # update z as for eqn. 5.5 pp. 1109
        z[-1] = dot(Z.T, u[-1])
        isNewDim = True
      # If isNewDim, i.e. new P, Q, matrices --> must run two steps
      #   to initialize the projection method
      if isNewDim:
        # Prepare projector
        Q = eye - dot(Z,Z.T)
        # Iterate once
        nu = func( u[-1], *args )
        u.append(nu)
        q.append( dot(Q,nu) )
        z.append( dot(Z.T,nu) )
        # Iterate once
        nu = func( u[-1], *args )
        u.append(nu)
        q.append( dot(Q,nu) )
        z.append( dot(Z.T,nu) )
        # done
        if disp:
          disp("Startup iterations for dimension %d" % Z.shape[1] )
        isNewDim = False
      # Degenerate case with no stabilization
      if Z.shape[1]==0:        
        u.append(func(u[-1],*args))
        q.append(u[-1])
        z.append(0)
      else: # non-degenerate case
        ## Numericaly compute the Jacobian applied to Z
        dz = z[-1]-z[-2]
        scl = abs(dz)*sRat+sMin
        # Compute dot(jacobian(func)(u),Z):
        DFZ = array([
          (func(u[-1]+zj*sj,*args)-func(u[-1]-zj*sj,*args))/(2*sj)
          for zj,sj in zip(Z.T,scl) 
        ]).T 
        # Function value
        Fu = func(u[-1],*args)
        # Compute newton update in z coord (eqn. 5.6 pp 1109)
        newt = inv(eyez - dot(Z.T,DFZ))
        z.append( z[-1] + dot(newt, dot(Z.T,Fu)-z[-1]) )
        # Q F + P F = F --> q = Q F = F - P F = u - p = u - Z z
        q.append( dot(Q,Fu) )
        u.append( q[-1] + dot(Z,z[-1]) )
      # simple "slowness" test
      n0 = norm(u[-1]-u[-2])
      n1 = norm(u[-2]-u[-3])
      if n0<xtol:
        break
      if n0>slowRat*n1:
        isSlow = True
      if disp:
        disp("Size is %g" % n0)  
    return u[-1],(u,q)

def pseudoNewton( func, ics, args=(), 
      rate=1.0, xtol=1e-6, sRat=1e-3, sMin=1e-6, maxIter = 100 
  ):
    ics = asarray(ics).flatten()
    dim = ics.size
    eye = identity(dim)
    u=[ics, func(ics,*args)]
    n = []
    for it in xrange(maxIter):
      ## Numericaly compute the Jacobian 
      du = u[-1]-u[-2]
      nn = norm(du)
      n.append(nn)
      if nn<xtol:
        break
      scl = abs(du)*sRat+sMin
      DFu = array([
        (func(u[-1]+ej*sj,*args)-func(u[-1]-ej*sj,*args))/(2*sj)
        for ej,sj in zip(eye,scl) 
      ]).T
      # Function value
      f = func(u[-1],*args)
      # Compute newton update
      nu = u[-1] - rate * dot( inv(DFu), f )
      print u[-1],repr(f),repr(DFu)
      u.append(nu)
    return u[-1],(u,n)
  
  
if 0:#__name__=="__main__":
  def showit( msg ):
    print msg
  def func( x ):
    return array([x[0]**2 + 0.7 * x[0], x[1] * 0.6, x[2] * 1.1])
  res = fixedPointRPM( func, [-0.5,-0.5,-0.5], disp=showit )
  
