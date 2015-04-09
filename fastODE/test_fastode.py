import os, time
from numpy import *
from pylab import *

from fastode import FastODE

class Rossler( FastODE ):
  def __init__(self):
    FastODE.__init__(self,"rossler_payload")
    
  def vis( self, ros ):
    subplot(221)
    plot(ros[:,1],ros[:,2])
    xlabel('x')
    ylabel('y')
    subplot(222)
    plot(ros[:,3],ros[:,2])
    xlabel('z')
    ylabel('y')
    subplot(223)
    plot(ros[:,1],ros[:,3])
    xlabel('x')
    ylabel('z')
    # Compute a "3D" view
    vwSkw = mat(((0,1,-1),(-1,0,1),(1,-1,0)))
    u,s,v = svd(vwSkw)
    vw = dot(u,dot(diag(exp(s/2)),v)).real
    vu = dot(ros[:,1:],vw[:,:2])
    subplot(224)
    plot(vu[:,0],vu[:,1])

  def test( self ):
    y = zeros((10000,self.WIDTH))    
    N = self.odeOnce( y, 800, pars = ( 0.2,  0.2,  5.7 ) )
    print "N=",N
    self.vis( y[:N,:].copy() )
    N = self.odeOnce( y, 800, pars = ( 0.2,  0.1,  5.7) )
    print "N=",N
    self.vis( y[:N,:].copy() )
    N = self.odeOnce( y, 800, pars = ( 0.2,  0.3,  5.7) )
    print "N=",N
    self.vis( y[:N,:].copy() )    

class Circle( FastODE ):
  def __init__(self):
    FastODE.__init__(self,"circle")
  
  def test( self ):
    y = zeros((10000,self.WIDTH))    
    y[0,:] = (0,1,0)
    N = self.odeOnce( y, 800 )
    print "N=",N
    t = y[:N+1,0]
    y = y[:N+1,1:]    
    def exact(t):
      return column_stack((cos(t),sin(t)))    
    plot(t,y-exact(t),'.-')
    legend(['x-cos(t)','y-sin(t)'],loc=0)
    title("Event time error %.2e" % (t[-1]-2*pi+arccos(-0.1)))
    self.t = t
    self.y = y
  
def runTest( Class ):
  print "Test: ",Class.__name__
  t0,ct0 = time.time(),time.clock()
  T = Class()
  T.test()
  t1,ct1 = time.time(),time.clock()
  print "Time: ",(t1-t0)," cpu ",ct1-ct0

if __name__=="__main__":
  figure()
  runTest( Circle )
  figure()
  runTest( Rossler )
