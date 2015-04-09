import numpy as np
from numpy import ( 
  array, asarray, dot, sum,  concatenate, argmax, empty_like, newaxis, dtype
)
import pdb as PDB

try:
  from scipy.cluster.vq import kmeans, whiten, vq
except ImportError:
  kmeans = None
  
class RBF( object ):
  """
  Radial Basis Function interpolator
  
  Concrete class implementing an RBF interpolator. An RBF interpolator computes
  the value of the interpolant by taking distances to known "knot-points",
  then using a (quickly decaying) function of that distance to generate weights
  for the knot points. The weights are used to combine known function values
  """
  def __init__( self ):
    """
    ATTRIBUTES:
      .c -- M x 1 x D -- M centers (knot points) for a D dimensional function
      .v -- M x ..S.. -- M values for the M points in .c
      .p -- M x ..V.. -- M parameters associated with the knot points 
    """
    self.clear()
      
  def clear(self):
    """
    Clear all fitting data
    """
    self.c = None
    self.v = None
    self.p = None
  
  def rbf( self, d ):
    """
    Radial basis function -- the importance function use for weight
    computations. 
    
    INPUT:
      d -- M x N x D -- pairwaise differences
    OUTPUT: 
      return an array of importances given by:
        rbf(d) := 1/(1+norm(d)**2/p)
      where p is the self.p parameter corresponding to the knot point
      
    Subclasses may override this method with different RBF-s
    """
    return 1/(1+sum(d * d.conj(),axis=-1)/self.p)
  
  def combine( self, d ):
    """
    Combiner function -- given importance matrix d and knot-point values self.v,
    compute the value of the interpolation function.
    
    INPUT:
      d -- N x M -- importance matrix for N points on M knot-points 

    Subclasses may override this method with different interpolants
    """
    return dot( self.weights(d), self.v )
    
  def weights( self, d ):
    """
    Convert the importance matrix d to a weight matrix wgt such that
    the interpolation values are: dot( wgt, self.v )

    Subclasses may override this method with different interpolants
    """
    return d / sum( d, axis=0 )[newaxis,...]  
    
  def fit( self, x, y, p=None ):
    """
    Fit the array valued function x --> y
    
    INPUT:
      x -- ..S.. x D -- an array of D dimensional points
      y -- ..S.. x ..V.. -- an array of V-shaped values corresponding to the
        points in x
      p -- ..S.. x ..W.. -- an array of parameters for the knot-point x
        by default, the value of p will be set automatically to 1,
        i.e. W = (1,), and all(p==1)
        
        In the RBF base class, p is the unit of distance squared for the RBF
        of that specific point, i.e. p=4 makes the radius of effect half the
        default value.
    
    WARNING:
      The first call to fit() on a newly clear()ed RBF will use the x,y
      parameters without copying, if they are arrays. This is INTENTIONAL, but
      may cause strange results if you change their contents after calling fit()
    """
    x,y = asarray(x),asarray(y)    
    rnk = x.ndim    
    # Make sure shapes match
    if y.shape[:rnk-1] != x.shape[:-1]:
      raise ValueError("x.shape %s does not match y.shape %s"
        % (repr(x.shape[:-1]),repr(y.shape[:rnk-1])))
    # Flatten the ..S.. part
    x = x.reshape(x.size/x.shape[-1],x.shape[-1])
    y = y.reshape((x.shape[0],)+y.shape[rnk-1:])
    if y.ndim == 1:
      y.shape = (y.size,1)
    # Reshape p to match, broadcasting as needed
    if p is None:
      # Create all ones
      p = empty_like(x[...,0])[:,newaxis]
      p[:] = 1
    else:      
      p = asarray(p)
      # If was a scalar --> broadcast to full size
      if not p.shape: 
        tmp = empty_like(x[...,0])
        tmp[:] = p
        p = tmp
      else: # else --> reshape, flattening the ..S.. part
        p = p.reshape((x.shape[0],)+p.shape[rnk-1:])
      if p.ndim == 1:
        p.shape = (p.size,1)
    # If these are the first points added --> set .c,.v
    if self.c is None:
      self.c = x[:,newaxis,:]
      self.v = y
      self.p = p
    else: # else make sure dimensions match, then append
      if x.shape[-1] != self.dim:
        raise ValueError("Points of dimension %d don't match domain dimension %d"
          % (x.shape[-1],self.dim))
      if y.shape[1:] != self.v.shape[1:]:
        raise ValueError("Values of shape %s don't match co-domain shape %s" 
          % (y.shape[1:],self.v.shape[1:]))
      if p.shape[1:] != self.p.shape[1:]:
        raise ValueError("Parameters of shape %s don't match previous shape %s" 
          % (p.shape[1:],self.p.shape[1:]))
      self.c = concatenate( (self.c,x[:,newaxis,:]),axis=0)
      self.v = concatenate( (self.v,y),axis=0)
      self.p = concatenate( (self.p,p),axis=0)    
    # Expose "sensible" array rank information
    self.dim = x.shape[-1]
    self.shape = self.v.shape[1:] 
  
  def val( self, x ):
    """
    Interpolate function at points x
    
    INPUT:
      x -- ..S.. x D -- D dimensional points at which to evaluate function
      
    OUTPUT:
      ..S.. x ..V.. -- array of values at corresponding positions
    """
    x = asarray(x)
    sz = x.shape
    if x.shape[-1] != self.dim:
      raise ValueError("Dimension mismatch: must have .shape[-1]==%d" 
        % self.dim)
    # All-pairs differences
    ofs = x.reshape( (1, x.size/self.dim, self.dim) ) - self.c
    # Radii for all pairs
    r = self.rbf( ofs )
    # Values
    v = self.combine( r.T )
    return v.reshape( sz[:-1] + v.shape[1:] )
  
  if kmeans:
    def mergeParams( self, idx ):
      """
      Merge the parameters for the knot points listed in idx
      
      For the default RBF implementation, this adds the cluster radius to the 
      scale of the radial functions, and takes the maximal value thereof.
      
      OUTPUT:
        p -- 1 x self.p.shape[1] --  new parameter vector for
          the merged knot points.
      """
      idx = asarray(idx).flatten()
      if idx.dtype is dtype('bool'):
        idx = idx.nonzero()[0]      
      c = self.c[idx,...]
      c -= sum( c, axis=0 )/float(idx.size)
      r2 = sum( c*c, axis=-1 )
      p2 = r2 + self.p[idx,...]
      print "N,r",len(idx), r2.max(axis=0) 
      return p2.max(axis=0)
      
    def kmeans( self, N ):
      """
      Use the k-means clustering algorithm to reduce the graph of the RBF 
      function down to N knot points.
      
      INPUT:
        N -- int -- number of knot points to keep
        
      NOTE:
        Parameters for the new RBF-s are computed using self.mergParams()
      """
      # Build graph
      gr = concatenate( (self.c[:,0,:], self.v), axis=1 )
      # Whiten
      scl = std(gr,axis=0)[newaxis,:]
      gr /= scl
      # Compute codebook
      km,_ = kmeans( gr, N )
      # New centroids
      nc = km[:,:self.dim] * scl[:,:self.dim]
      # Eval at new centroids to get new values
      nv = self.val( nc )
      # Identify which knot points go to each new centroid
      cid,_ = vq( gr, km )
      # Loop over new centroids to obtain their parameter values
      np = array([
          self.mergeParams( cid == n ) 
          for n in xrange(N)
      ])      
      self.c = nc[:,newaxis,:]
      self.v = nv
      self.p = np
            
  def __call__(self, x):
    return self.val(x)
    
class VornoiBF( RBF ):
  """
  Use a Vornoi-diagram like interpolant -- map each point to the value at the
  nearest knot point. 
  """
  def __init__(self):
    RBF.__init__(self)
  
  def combine( self, wgt ):
    mx = argmax(wgt,axis=-1)
    return self.v[mx,...]

def test_VornoiBF(n=20):
  v = VornoiBF()
  pxy = rand(n,2)
  v.fit( pxy, linspace(0,1,n) )
  x,y = meshgrid( linspace(0,1,n*4), linspace(0,1,n*4) )
  xy = concatenate((x[...,newaxis],y[...,newaxis]), axis=2)
  z = v(xy)
  z.shape = x.shape
  imshow(z)
  ax = gca().axis()
  plot(pxy[:,0]*(ax[1]-ax[0])+ax[0],(1-pxy[:,1])*(ax[3]-ax[2])+ax[2],'ow')

def test_vecRBF():
  r = RBF()
  p = []
  v = []
  x,y = meshgrid( linspace(0,1,35), linspace(0,1,35) )
  xy = concatenate((x[...,newaxis],y[...,newaxis]), axis=2)
  
  for k in xrange(9):
    p.append( rand(2,2) )
    v.append( randn(2,2) )
    r.fit( p[-1], v[-1], [0.01]*2 )
    subplot(3,3,k+1)
    cp = concatenate(p)
    cv = concatenate(v)
    plot(cp[:,0],cp[:,1],'yo')
    quiver( cp[:,0], cp[:,1], cv[:,0], cv[:,1])
    z = r(xy)
    quiver( x, y, z[...,0], z[...,1], color=[0.5]*3 )
    a = gca()
    a.set_xticks([])
    a.set_yticks([])
    text(0.5,0.5,"[%d]" % (k+1),backgroundcolor = [1,1,1], size='large')
    
    
