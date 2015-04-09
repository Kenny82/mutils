import fastode
import numpy
import numpy.random
import time
p = numpy.array([0.2,0.2,5.7,10])
y = numpy.zeros((10000,4),numpy.double)
y0 = numpy.array([0,5.0,-5,2])
n = 0
sl = []
ode = fastode.FastODE('rossler_payload')
while n < y.shape[0]-5:
  y[0,:] = y0
  l = ode.odeOnce( y, 10, dt=1,  pars=p )
  print "--> ",n,l
  sl.append(slice(n,n+l,1))
  y0 = numpy.array([0,5.0,-5,2]) + numpy.random.randn(4)/10
  n = n + l

from pylab import *
subplot(221)
[ plot(y[s,1],y[s,2]) for s in sl ]
subplot(222);
[ plot(y[s,1],y[s,3]) for s in sl ]
subplot(223);
[ plot(y[s,2],y[s,3]) for s in sl ]
show()

