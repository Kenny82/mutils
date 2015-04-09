from numpy import *
from util import *
from scipy import signal
import warnings
import pylab
import util as u

class ZScore:
  """
  Class for finding z scores of given measurements with given or computed
  covarance matrix.
  
  This class implements equation (7) of [Revzen08]
  
  Properties:
    y0 -- Dx1 -- measurement mean
    M -- DxD -- measurement covariance matrix
    S -- DxD -- scoring matrix
  """
  
  def __init__( self, y = None, M = None ):
    """Computes the mean and scoring matrix of measurements
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
      M -- DxD (optional) -- measurement error covariance for y
        -- If M is missing, it is assumed to be diagonal with variances
	-- given by 1/2 variance of the second order differences of y
    """
    
    # if M given --> use fromCovAndMean
    # elif we got y --> use fromData
    # else --> create empty object with None in members 
    if M is not None:
      self.fromCovAndMean( mean(y, 1), M)
    elif y is not None:
      self.fromData( y )
    else:
      self.y0 = None
      self.M = None
      self.S = None


  
  def fromCovAndMean( self, y0, M ):
    """
    Compute scoring matrix based on square root of M through svd
    INPUT:
      y0 -- Dx1 -- mean of data
      M -- DxD -- measurement error covariance of data
    """
    self.y0 = y0
    self.M = M
    (D, V) = linalg.eig( M )
    self.S = dot( V.transpose(), diag( 1/sqrt( D ) ) )
  
  def fromData( self, y ):
    """
    Compute scoring matrix based on estimated covariance matrix of y
    Estimated covariance matrix is geiven by 1/2 variance of the second order
    differences of y
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    """
    self.y0 = mean( y, 1 )
    self.M = diag( std( diff( y, n=2, axis=1 ), axis=1 ) )
    self.S = diag( 1/sqrt( diag( self.M ) ) )
  
  def __call__( self, y ):
    """
    Callable wrapper for the class
    Calls self.zScore internally
    """
    return self.zScore( y )
  
  def zScore( self, y ):
    """Computes the z score of measurement y using stored mean and scoring
    matrix
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    OUTPUT:
      zscores for y -- DxN
    """
    return dot( self.S, y - self.y0.reshape( len( self.y0 ), 1 ) )



def _default_psf(self, x):
  """Default Poincare section function
     by rights, this should be inside the Phaser class, but pickle
     would barf on Phaser objects if they contained functions that
     aren't defined in the module top-level.
  """
  return signal.lfilter( 
    array([0.02008336556421, 0.04016673112842,0.02008336556421] ), 
    array([1.00000000000000,-1.56101807580072,0.64135153805756] ),
  x[0,:] )
  
  
class DataMismatchError(Exception):
     def __init__(self, value = None):
         if value is None:     
             self.value = 'Data do not match'
         else:
             self.value = value
     def __str__(self):
         return repr(self.value)
  
  
class Phaser2:
  """
  Class to train and evaluate a phaser
  
  Properties:
    sc -- ZScore object for converting y to z-scores
    P_k -- Dx1 list of FourierSeries object -- series correction for correcting proto-phases
    prj -- Dx1 complex -- projector on combined proto-phase
    P -- 1x1 FourierSeries object -- series correction for combined phase
    psf -- 1x1 Method -- callback to psecfun

    d_ordp -- 1x1 Int -- default ordP
    
    Adaption from Moritz Maus (compared to Phaser):
        changes in the internal data structure such that the data used as 
        poincare-section does not have to be included 
  """
  
  
  
  def __init__( self, y = None, psecData = None, C = None, ordP = None, psecfunc = None ):
    """
    Initilizing/training a phaser object
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
      ordP -- 1x1 (optional) -- Orders of series to use in series correction
      psecfunc -- 1x1 (optional) -- Poincare section function
    """
    
    # if psecfunc given -> use given
    if psecfunc is not None:
      self.psf = psecfunc
    else:
      self.psf = _default_psf
    
    self.cycl = None
    # if y given -> calls self.phaserTrain
    if y is not None:
        if len(y) is not len(psecData):            
            raise DataMismatchError('length of Trials and Poincare-section-data do not match')
        else:            
            self.phaserTrain( y, psecData, C, ordP )
  
  def __call__( self, dat ):
    """
    Callable wrapper for the class. Calls phaserEval internally
    """
    return self.phaserEval( dat )
  
    
  def phaserEval( self, dat, trialPsecData ):
    """
    Computes the phase of testing data
    INPUT:
      dat -- DxN -- Testing data whose phase is to be determined
    OUTPUT:
      Returns the complex phase of input data
    """
    
    # compute z score    
    z = self.sc.zScore( dat )    
    
    # compute Poincare section
    p0 = trialPsecData
    
    # compute protophase using Hilbert transform    
    zeta = self.mangle * u.hilbert(z) #array([signal.hilbert( zs ) for zs in z])

    """
    figure(9)
    print('z shape:')
    print(z.shape)
    print('zeta shape:')
    print(zeta.shape)
    plot(diff(angle(zeta)).T)
    """
    
    z0, ido0 = self.sliceN( zeta, p0 )
    
    # Compute phase offsets for proto-phases
    ofs = exp(-1j * angle(mean(z0, axis = 1)).T)
    
    # series correction for each dimision using self.P_k
    th = self.angleUp( zeta * ofs[:,newaxis] ) 
    
    # evaluable weights based on sample length
    p = 1j * zeros( th.shape )
    print('th-shape: %s' % str(th.shape))
    for k in range( th.shape[0] ):        
        p[k,:] = self.P_k[k].val( th[k,:] ).T + th[k,:]
    
    rho = mean( abs( zeta ), 1 ).reshape(( zeta.shape[0], 1 ))
    # compute phase projected onto first principal components using self.prj
    ph = self.angleUp( dot( self.prj.T, vstack( [cos( p ) * rho, sin( p ) * rho] ) ))
    
    # return series correction of combined phase using self.P
    phi = real( ph + self.P.val( ph ).T )
    pOfs2 = (p0[ido0+1] * exp(1j * phi.T[ido0+1]) - p0[ido0] * exp(1j * phi.T[ido0] )) / (p0[ido0+1] - p0[ido0])
    return phi - angle(sum(pOfs2))
  
  def phaserTrain( self, y, allpsecData, C = None, ordP = None ):
    """
    Trains the phaser object with given data.
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
    """
    
    # if given one sample -> treat it as an ensemble with one element
    if y.__class__ is ndarray:
      y = [y]
    
    # check dimension agreement in ensemble
    if len( set( [ ele.shape[0] for ele in y ] ) ) is not 1:
      raise( Exception( 'newPhaser:dims','All datasets \
	      in the ensemble must have the same dimension' ) )
    D = y[0].shape[0]
    
    
    # train ZScore object based on the entire ensemble
    self.sc = ZScore( hstack( y ), C )
    
    
    # initializing proto phase variable
    zetas = []
    cycl = zeros( len( y ))
    svm = 1j*zeros( (D, len( y )) )
    svv = zeros( (D, len( y )) )
    
    # compute protophases for each sample in the ensemble
    for k in range( len( y ) ):
      # hilbert transform the sample's z score
      #print('k = %i' % k)
      
      zsc = self.sc.zScore( y[k] )       
      zetas.append( u.hilbert(zsc) ) # array([signal.hilbert( zs ) for zs in zsc]))
      
      #print ('(y[%i].shape = %s' % (k,str(y[k].shape)))
      # trim beginning and end cycles, and check for cycle freq and quantity      
      cycl[k], zetas[k], y[k] = self.trimCycle( zetas[k], y[k] )
      print 'aps-shape:', allpsecData[k].shape, '\n'
      allpsecData[k] = allpsecData[k][cycl[k]:-cycl[k]]
      print 'aps-shape(2):', allpsecData[k].shape, '\n'
      #print ('(y[%i].shape = %s' % (k,str(y[k].shape)))
      #print ('(cycl[%i].shape = %2.3f' % (k,cycl[k])  )
      print ('(zetas[%i].shape = %s' % (k,str(zetas[k].shape)) )


      # Computing the Poincare section
      # sk = self.psf( y[k] )
      sk = allpsecData[k]
      print 'sk.shape: ', sk.shape, 'zetas.shape:', zetas[k].shape, 'cycl: ', cycl[k]
      (sv, idx) = self.sliceN( zetas[k], sk )      
      if idx.shape[-1] == 0:
	raise Exception( 'newPhaser:emptySection', 'Poincare section is empty -- bailing out' )
      
      svm[:,k] = mean( sv, 1 )
      svv[:,k] = var( sv, 1 ) * sv.shape[1] / (sv.shape[1] - 1)
      
    
    # computing phase offset based on psecfunc
    self.mangle, ofs = self.computeOffset( svm, svv )
    
    # correcting phase offset for proto phase and compute weights
    wgt = zeros( len( y ) )
    rho_i = zeros(( len( y ), y[0].shape[0] ))
    for k in range( len( y ) ):
      zetas[k] = self.mangle * exp( -1j * ofs[k] ) * zetas[k]
      wgt[k] = zetas[k].shape[0]
      rho_i[k,:] = mean( abs( zetas[k] ), 1 )
    
    # compute normalized weight for each dimension using weights from all samples
    wgt = wgt.reshape(( 1, len( y )))
    rho = ( dot( wgt, rho_i ) / sum( wgt ) ).T
    # if ordP is None -> use high enough order to reach Nyquist/2
    if ordP is None:
      ordP = ceil( max( cycl ) / 4 )
    
    # correct protophase using seriesCorrection
    self.P_k = self.seriesCorrection( zetas, ordP )
    
    
    # loop over all samples of the ensemble
    q = []
    for k in range( len( zetas ) ):
      # compute protophase angle
      th = self.angleUp( zetas[k] )
      
      phi_k = 1j * ones( th.shape )
      
      # loop over all dimensions
      for ki in range( th.shape[0] ):
        # compute corrected phase based on protophase
	phi_k[ki,:] = self.P_k[ki].val( th[ki,:] ).T + th[ki,:]
      
      # computer vectorized phase
      q.append( vstack( [cos( phi_k ) * rho, sin( phi_k ) * rho] ) )
    
    # project phase vectors using first two principal components
    W = hstack( q[:] )
    W = W - mean( W, 1 )[:,newaxis]
    pc = svd( W, False )[0]
    self.prj = reshape( pc[:,0] + 1j * pc[:,1], ( pc.shape[0], 1 ) )
    
    # Series correction of combined phase
    qz = []
    for k in range( len( q ) ):
      qz.append( dot( self.prj.T, q[k] ) )
    
    # store object members for the phase estimator
    self.P = self.seriesCorrection( qz, ordP )[0]
    self.cycl = cycl
  
  
  def computeOffset(self, svm, svv ):
    """
    """
    # convert variances into weights
    svv = svv / sum( svv, 1 ).reshape( svv.shape[0], 1 )
    
    # compute variance weighted average of phasors on cross section to give the phase offset of each protophase
    mangle = sum( svm * svv, 1)
    if any( abs( mangle ) ) < .1:
      b = find( abs( mangle ) < .1 )
      raise Exception( 'computeOffset:badmeasureOfs', len( b ) + ' measuremt(s), including ' + b[0] + ' are too noisy on Poincare section' )
    
    # compute phase offsets for trials
    mangle = conj( mangle ) / abs( mangle )
    mangle = mangle.reshape(( len( mangle ), 1))
    svm = mangle * svm
    ofs = mean( svm, 0 )
    if any( abs( ofs ) < .1 ):
      b = find( abs( ofs ) < .1 )
      raise Exception( 'computeOffset:badTrialOfs', len( b ) + ' trial(s), including ' + b[0] + ' are too noisy on Poincare section' )
    
    return mangle, angle( ofs )
  
  #computeOffset = staticmethod( computeOffset )
  
  def sliceN(self, x, s, h = None ):
    """
    Slices a D-dimensional time series at a surface
    INPUT:
      x -- DxN -- data with colums being points in the time series
      s -- N, array -- values of function that is zero and increasing on surface
      h -- 1x1 (optional) -- threshold for transitions, transitions>h are ignored
    OUPUT:
      slc -- DxM -- positions at estimated value of s==0
      idx -- M -- indices into columns of x indicating the last point before crossing the surface
    """
    
    # checking for dimension agreement    
    #print('%s - %s' % (str(x.shape),str(s.shape)))
    if x.shape[1] != s.shape[0]:
      raise Exception( 'sliceN:mismatch', 'Slice series must have matching columns with data' )
    
    idx = find(( s[1:] > 0 ) & ( s[0:-1] <= 0 ))
    idx = idx[idx < x.shape[1]]
    
    if h is not None:
      idx = idx( abs( s[idx] ) < h & abs( s[idx+1] ) < h );
    
    N = x.shape[0]
    
    if len( idx ) is 0:
      return zeros(( N, 0 )), idx
    
    wBfr = abs( s[idx] )
    wBfr = wBfr.reshape((1, len( wBfr )))
    wAfr = abs( s[idx+1] )
    wAfr = wAfr.reshape((1, len( wAfr )))
    slc = ( x[:,idx]*wAfr + x[:,idx+1]*wBfr ) / ( wBfr + wAfr )
    
    return slc, idx
  
  #sliceN = staticmethod( sliceN )
  
  def angleUp(self, zeta ):
    """
    Convert complex data to increasing phase angles
    INPUT:
      zeta -- DxN complex
    OUPUT:
      returns DxN phase angle of zeta
    """
    # unwind angles
    th = unwrap( angle ( zeta ) )
    
    # reverse decreasing sequences
    bad = th[:,0] > th[:,-1]
    if any( bad ):
      th[bad,:] = -th[bad,:]
    return th
  
  #angleUp = staticmethod( angleUp )
  
  def trimCycle(self, zeta, y ):
    """
    """
    # compute wrapped angle for hilbert transform
    ph = self.angleUp( zeta )
    
    # estimate nCyc in each dimension
    nCyc = abs( ph[:,-1] - ph[:,0] ) / 2 / pi
    cycl = ceil( zeta.shape[1] / max( nCyc ) )
    
    # if nCyc < 7 -> warning
    # elif range(nCyc) > 2 -> warning
    # else truncate beginning and ending cycles
    if any( nCyc < 7 ):
      warnings.warn( "PhaserForSample:tooShort" )
    elif max( nCyc ) - min( nCyc ) > 2:
      warnings.warn( "PhaserForSample:nCycMismatch" )
    else:
      zeta = zeta[:,cycl:-cycl]
      y = y[:,cycl:-cycl]
    
    return cycl, zeta, y
  
  #trimCycle = staticmethod( trimCycle )
  
  def seriesCorrection(self, zetas, ordP ):
    """
    Fourier series correction for data zetas up to order ordP
    INPUT:
      zetas -- [DxN_1, DxN_2, ...] -- list of D dimensional data to be corrected using Fourier series
      ordP -- 1x1 -- Number of Fourier modes to be used
    OUPUT:
      Returns a list of FourierSeries object fitted to zetas
    """
    
    # initialize proto phase series 2D list
    proto = []
    
    # loop over all samples of the ensemble
    wgt = zeros( len( zetas ) )
    for k in range( len( zetas ) ):
      proto.append([])
      # compute protophase angle (theta)
      zeta = zetas[k]
      N = zeta.shape[1]
      theta = self.angleUp( zeta )
      
      # generate time variable
      t = linspace( 0, 1, N )      
      # compute d_theta
      dTheta = diff( theta, 1 )
      # compute d_t
      dt = diff( t )
      # mid-sampling of protophase angle
      th = ( theta[:,1:] + theta[:,:-1] ) / 2.0
      
      # loop over all dimensions
      for ki in range( zeta.shape[0] ):
        # evaluate Fourier series for (d_theta/d_t)(theta)
	# normalize Fourier coefficients to a mean of 1
	fdThdt = FourierSeries().fit( ordP * 2, th[ki,:].reshape(( 1, th.shape[1])), dTheta[ki,:].reshape(( 1, dTheta.shape[1])) / dt )
	fdThdt.coef = fdThdt.coef / fdThdt.m
	fdThdt.m = array([1])
	
	# evaluate Fourier series for (d_t/d_theta)(theta) based on Fourier
	# approx of (d_theta/d_t)
	# normalize Fourier coefficients to a mean of 1
	fdtdTh = FourierSeries().fit( ordP, th[ki,:].reshape(( 1, th.shape[1])), 1 / fdThdt.val( th[ki,:].reshape(( 1, th.shape[1] )) ).T )
	fdtdTh.coef = fdtdTh.coef / fdtdTh.m
	fdtdTh.m = array([1])
	
	# evaluate corrected phsae phi(theta) series as symbolic integration of 
	# (d_t/d_theta), this is off by a constant
	proto[k].append(fdtdTh.integrate())
        del fdThdt
        del fdtdTh
      
      # compute sample weight based on sample length
      wgt[k] = zeta.shape[0]
      
    wgt = wgt / sum( wgt )
    
    # return phase estimation as weighted average of phase estimation of all samples
    proto_k = []
    for ki in range( zetas[0].shape[0] ):
      proto_k.append( FourierSeries.bigSum( [p[ki] for p in proto], wgt ))
      
    return proto_k
    
  #seriesCorrection = staticmethod( seriesCorrection )
