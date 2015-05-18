# -*- coding : utf8 -*-
"""

.. module:: mutils.misc
    :synopsis: This module contains some convenient functions for the linear return map
        analysis and other miscellanelous routines

.. moduleauthor:: Moritz Maus <mmaus@sport.tu-darmstadt.de>


"""


import numpy
from pylab import (vstack, hstack, zeros, arange, array, randn, dot, sum, eig,
                   logspace,sqrt, exp, log, find, sin, diag, randint, svd, 
                   inv, mod, mean, cumsum, pi, zeros_like, linspace, interp,
                   polyval, ceil, eye, isreal, isnan, linspace, concatenate,
                   convolve, polyfit, polyval, roots, median)
from scipy.signal import medfilt
from scipy.special import gamma, hyp2f1
from scipy.linalg.matfuncs import sqrtm
from scipy.integrate import odeint
import time
import scipy.fftpack as fft

import re
#from IPython.nbformat import current
from IPython import nbformat   # new in IPython3

#import mutils.fourier as mfou

# keep for compatibility: these functions were previously in this module
from mutils.fourier import diff_f, diff_f0, int_f, int_f0, interp_f

def fBM_nd(dims, H, return_mat = False, use_eig_ev = True):
    """
    creates fractional Brownian motion
    parameters: dims is a tuple of the shape of the sample path (nxd); 
                H: Hurst exponent
    this is the slow version of fBM. It might, however, be more precise than
    fBM, however - sometimes, the matrix square root has a problem, which might
    induce inaccuracy    
    use_eig_ev: use eigenvalue decomposition for matrix square root computation
    (faster)
    """
    n = dims[0]
    d = dims[1]
    Gamma = zeros((n,n))
    print ('building ...\n')
    for t in arange(n):
        for s in arange(n):
            Gamma[t,s] = .5*((s+1)**(2.*H) + (t+1)**(2.*H) - abs(t-s)**(2.*H))
    print('rooting ...\n')    
    if use_eig_ev:
        ev,ew = eig(Gamma.real)
        Sigma = dot(ew, dot(diag(sqrt(ev)),ew.T) )
    else:
        Sigma = sqrtm(Gamma)
    if return_mat:
        return Sigma
    v = randn(n,d)
    return dot(Sigma,v)

def fill_nan(data, max_len=None, fill_ends=True):
    """
    Fills the "nan" fields of a 1D array with linear interpolated values.
    At the edges, constant values are assumed.
    
    :args:
       data (1d array): the input data
       max_len (int or None): maximal length of gaps to fill
       fill_ends (bool): whether or not to fill the ends
    
    :returns:
        data' (1d array): a copy of the input data, where `nan`-values are
        replaced by a linear interpolation between adjacent values
    """
    res = data.copy()
    if all(isnan(data)):
        return res
    missing_idx = find(isnan(data))
    
    # group to missing segments
    missing_segs = []
    gap_lengths = []
    lastidx = -2 # some invalid index: idx == lastidx + 1 cannot be true for this!
    startidx = -2 # some invalid index
    gaplen = 0
    for idx in missing_idx:
        if idx == lastidx + 1:
            # all right, the segment continues
            lastidx = idx            
            gaplen += 1
        else:
            # a new segment has started            
            # first: "close" old segment if exists
            if startidx >= 0:
                missing_segs.append([startidx, lastidx])
                gap_lengths.append(gaplen)
            # now: initialize new segment
            gaplen = 1
            startidx = idx
            lastidx = idx
    
    # manually close the last segment if exists
    if startidx >= 0:
        if lastidx < len(data) - 1 or fill_ends: # skip edge if not fill_ends
            missing_segs.append([startidx, lastidx])
    
    # fill missing segments
    for seg in missing_segs:
        start_idx, stop_idx = seg
        if max_len is not None:
            if stop_idx - start_idx > max_len:
                continue
        # if startpoint is missing: constant value
        if start_idx == 0 and fill_ends:
            res[:stop_idx + 1] = res[stop_idx + 1]
        # if endpoint is missing: use constant value
        elif stop_idx == len(data)-1 and fill_ends:
            res[start_idx:] = res[start_idx - 1]
        # else: linear interpolation
        else:

            res[start_idx: stop_idx+1] = interp(range(start_idx, stop_idx + 1), 
                [start_idx - 1, stop_idx + 1], data[[start_idx - 1, stop_idx + 1]])
        
    return res

def kern(mat, threshold = 1e-8):
    """
    returns the kernel of the matrix
    parameter:
        mat: matrix to be analyzed
        threshold: min. singular value to be considered as "different from 0"
    """
    u,s,v = svd(mat,full_matrices = True)
    dim_k = len(find(s < threshold)) + v.shape[0] - len(s)
    if dim_k > 0:
        return v[-dim_k:,:].T
    else:
        return None


def sim_Lorenz(t,x0 = [1.,0.,0.],params = [28.,10.,8./3.]):
    """
    simulates the Lorez attractor
    parameters rho, sigma, beta
    parameters: x0: initial value
    """     
    def dLorenz(x,t0,par):
        """
        the Lorenz equations
        """
        return array([par[1]*(x[1]-x[0]),
                      x[0]*(par[0]-x[2]) - x[1],
                      x[0]*x[1] - par[2]*x[2]])
                      
    return odeint(dLorenz,array(x0),t,args=(array(params),),rtol=1e-9,atol=1e-9)
    

def sim_Roessler(t, x0 = [5.,0.,5.], params = [0.2, 0.2, 5.7]):
    """
    simulates the Roessler attractor

    @param t (N-by-1 array): the output times of the simulation
    @param x0 (3-by-1 array or list): initial state
    @param params (3-by-1 array or list): parameters

    @return Returns a N-by-3 array, containing the computed trajectory.
    """     
    def dRoessler(x,t0,par):
        """
        the Roessler equations
        """
        return array([-x[1]-x[2],
                      x[0] + par[0] * x[1],
                      par[1] + x[2] * (x[0] - par[2])])
                      
    return odeint(dRoessler, array(x0), t, args=(array(params),),
            rtol=1e-9, atol=1e-9)


def calcSpatScaling(data,idx,nRadii = None):
    """
    calculates the spatial scaling behavior
    computes the number of points in a neighbourhood of radius r for each point
    whose index is given in "indices".    
    data must be an array of NxD - format, N: number of observations, 
    D: system dimension
    
    returns: [(radius, N), ... ]
    to estimate dimension from that:
        D ~ lim(radius -> 0){ d log(N) / d log(radius) }
    """
    nRadii = data.shape[0]/100 if nRadii is None else nRadii
    
    #for idx in indices:
    sq_dists = sqrt(sum((data - data[idx,:])**2,axis=1))
        
    radii = logspace(-5,log(max(sq_dists))+.1,    
                  nRadii,base=exp(1))
    sq_dists.sort()
    # this is rather slow; a "manual walk-through would be faster -> implement
    # if required
    return vstack([(r,len(find(sq_dists<r))) for r in radii])
    
    
def sinDistort(data,twisting=1.):
    """
    a distortion of the data:
        x will be mapped to sin(x/max(abs(x))), for every coordinate
    this is to distort a lower dimensional system so that it is not
    restricted to a lower-dimensional subspace any longer
    data must be given in NxD - Format
    the optional twisting factor increases the twisting strength
    """
    return vstack([sin( data[:,x]*twisting/max(abs(data[:,x])))*max(abs(data[:,x]))
                   for x in range(data.shape[1])]).T
    
    

def fBM(n,H):
    """
    creates fractional Brownian motion
    parameters: length of sample path; 
                Hurst exponent
    this method uses another computational approach than fBM
    http://en.wikipedia.org/wiki/Fractional_Brownian_motion#Method_2_of_simulation
    I should somewhen look that up with proper references
    - proper scaling is not implemented. Look up the article to implement it!
    """
    gammaH = gamma(H + .5)
    def KH(t,s):
        """
        accordint to the article
        """
        return (t - s)**(H - .5)/gammaH*hyp2f1(H-.5, .5-H, H+.5, 1.-float(t)/float(s))
        
    incs = randn(n+1)
    path = zeros(n+1)
    for pos in arange(n)+1:        
        path[pos] = sum([KH(pos,x)*incs[x] for x in arange(pos)+1] )
    
    return path[1:]


def ulam(niter,x0=0):
    """
    creates iterates of the ulam map: x(n+1) = 1 - 2*x(n)^2
    the ulam map is a chaotic 1d mapping
    """
    res = zeros(niter)
    res[0] = array(x0)
    for x in arange(1,niter):
        res[x] = 1.-2*res[x-1]**2
    return res


def henon(niter,x0=[0,0],alpha=1.4,beta=.3):
    """
    creates iterates of the henon map: 
        x(n+1) = 1 - alpha*x(n)^2 + y(n)
        y(n+1) = beta*x(n)
    """
    res = zeros((niter,2))
    res[0,:] = array(x0)
    for x in arange(1,niter):
        res[x,0] = 1.-alpha*res[x-1,0]**2 + res[x-1,1]
        res[x,1] = beta*res[x-1,0]
    return res

    

def createBigMap(sectionMappings):
    """
    creates a permutation matrix out of section mappings in the form
    | 0    ....     an |
    | a1 0   ...     0 |
    | 0  a2   ...    0 |
    | ...      ...     |
    | 0 0 ... a(n-1) 0 |
    , where A = an*a(n-1)*...*a2*a1 is the full-stride map
    """
    dim0 = sectionMappings[0].shape[0]
    matSize = len(sectionMappings)*dim0    
    mat0 = zeros( (matSize, matSize))
    for n, mapping in enumerate(sectionMappings[::-1]):
        row0 = ((n+1)*dim0) % matSize
        col0 = n*dim0        
        mat0[ row0:row0+dim0, col0:col0+dim0] = mapping.copy()
    
    return mat0

def twoD_oneD(data2D,nps):
    """
    transforms the 2D format into the 1D-format used here
    1D-format: 
        a single row represents one stride; 
        the first (nps) frames represent coordinate 1, the second (nps)
        frames represent coordinate 2, ...
    2D-format: 
        a single row represents one coordinate.
        The k-th stride is represented by the subsection [:,k*(nps):(k+1)*nps]
    """    
    data1D = vstack([hstack(data2D[:,x*nps:(x+1)*nps]) for x in range(data2D.shape[1]/nps)])
    return data1D


def oneD_twoD(data1D,nps):
    """
    transforms the 1D-format used here into the 2D-format
    1D-format: 
        a single row represents one stride; 
        the first (nps) frames represent coordinate 1, the second (nps)
        frames represent coordinate 2, ...
    2D-format: 
        a single row represents one coordinate.
        The k-th stride is represented by the subsection [:,k*(nps):(k+1)*nps]    
    """   
    ncoords = data1D.shape[1]/nps
    data2D = vstack([hstack(data1D[:,nps*x:nps*(x+1)]) for x in range(ncoords)])
    return data2D





def a2dat(data, filename, sep = '\t'):
    """
    writes a numpy array into a csv file
    """
    myfile = open(filename,'w')
    for a in data:
        if type(a) == numpy.ndarray:                
            mystring = numpy.array2string(a, separator = sep,max_line_width = numpy.inf)[1:-1]                
        else:
            mystring = str(a)
        mystring = mystring.strip()
        myfile.write('%s\n' % mystring)
    myfile.close()
        

    
def dat2a(filename, sep = None):
    """ reads a csv file into a numpy array
    any line that does not only contain numbers.
    with sep, a separator can be given (default: None, i.e. all whitespaces)
    """
    myfile = open(filename,'r')
    c = []
    counter = 0
    while True:            
        mystring = myfile.readline()    
        if mystring == '':
            break
        counter += 1
        try:                
            vals = [ float(x) for x in mystring.split(sep)]
            c.append(vals)
        except ValueError:
            print('skipping line %s: %s\n' % (str(counter), mystring))
        
        
    myfile.close()
    c = numpy.array(c)        
    return c

def matDist(mat1, mat2,nidx = 100):
    """
    returns the distance of two lists of matrices mat1 and mat2.
    output: [d(mat1,mat1),d(mat2,mat2),d(mat1,mat2),d(mat2,mat1)]
    d(mat1,mat2) and d(mat2,mat1) should be the same
    up to random variance (when d(mat1,mat1) and d(mat2,mat2) have the same
    width in "FWHM sense")
    
    nidx: n matrices are compared to n matrices each, that is the result has
    length n**2
    """
    
    # pick up a random matrix from mat1
    # compute distances from out-of-sample mat1
    # compute distances from sample of same size in mat2
    # repeat; for random matrix from mat2
    d_11 = []
    d_22 = []
    d_12 = []
    d_21 = []
    
    nidx1 = nidx
    nidx2 = nidx
    # for d_11 and d_12
    for nmat in randint(0,len(mat1),nidx1):
        refmat = mat1[nmat]
        for nmat_x in randint(0,len(mat1),nidx1):
            if nmat_x == nmat:
                nmat_x = (nmat - 1) if nmat > 0 else (nmat + 1)
            d_11.append(svd(mat1[nmat_x] - refmat,False,False)[0])
        # ... I could use a []-statement, but I do not want to reformat a list
        # of lists ...
        for nmat_x in randint(0,len(mat2),nidx1):
            d_12.append(svd(mat2[nmat_x] - refmat,False,False)[0])

    for nmat in randint(0,len(mat2),nidx2):
        refmat = mat2[nmat]
        for nmat_x in randint(0,len(mat2),nidx2):
            if nmat_x == nmat:
                nmat_x = (nmat - 1) if nmat > 0 else (nmat + 1)
            d_22.append(svd(mat2[nmat_x] - refmat,False,False)[0])
        # ... I could use a []-statement, but I do not want to reformat a list
        # of lists ...
        for nmat_x in randint(0,len(mat1),nidx2):
            d_21.append(svd(mat1[nmat_x] - refmat,False,False)[0])

    return (d_11,d_22,d_21,d_12)


def kalman(F,G,H,u,y,x0,Px0,Px,Py):
    """
    computes the discrete-time Kalman-Filter
    parameters:
        F - system matrix [m x m]
        G - input matrix   x(k+1) = F*x(k) + G*u(k) [m x d_u]
        H - observation matrix  y(k) = H*x(k)
        u: input vector [d_u x n]
        y: observations [d_y x n]
        x0: initial estimate [d_x,]
        Px: (initial) covariance matrix of x
        Py: covariance matrix of y
        Q: covariance of dynamical noise (can be 0)
    returns:
        (state, state_cov) the state and the state covariance matrices
    
    Implementation according to Geering: Regelungstechnik, Springer (2004),
    p. 259f.
    """
    n = y.shape[-1]
    if y.shape[-1] != u.shape[-1]:
        raise ValueError, 'y and u do not have same length!'
    
    # reserve space for results
    Px_minus = zeros((Px0.shape[0],Px0.shape[1],n))
    Px_plus = zeros((Px0.shape[0],Px0.shape[1],n))
    x_minus = zeros((x0.shape[0],n))
    x_plus = zeros((x0.shape[0],n))
    
    Px_minus[:,:,0] = Px0
    x_minus[:,0:1] = x0
    
    # compute for each time frame
    for k in xrange(n):
        Px_plus[:,:,k] = Px_minus[:,:,k] - reduce(dot,
                         [Px_minus[:,:,k], H.T, 
                          inv(Py + dot(H,dot(Px_minus[:,:,k],H.T))), H,
                          Px_minus[:,:,k]])
        L = dot(Px_minus[:,:,k], 
                dot(H.T, inv( Py + dot(H, dot(Px_minus[:,:,k],H.T)) )))
        x_plus[:,k:k+1] = x_minus[:,k:k+1] + dot(L,
                          (y[:,k:k+1] - dot(H,x_minus[:,k:k+1]) ))
        if k < n-1:
            #print 'k = ', k, 'x-plus: ',x_plus.shape,'u',u.shape, 'x_minus:', x_minus.shape         
            Px_minus[:,:,k+1] = (dot(F, dot(Px_plus[:,:,k],F.T)) +
                                 dot(G, dot(Px,G.T)))
            x_minus[:,k+1:k+2] = dot(F,x_plus[:,k:k+1]) + dot(G,u[:,k:k+1])
        
    
    return x_plus, Px_plus



def dHausdorff(A,B,d):
    """
    computes the Hausdorff distance of two sets A,B, whose element-wise distance
    is given by the function d(a,b)
    A: set (e.g. list) with elements a
    B: set with elements b
    d: (a,b) -> R+ the distance between a and b
    
    the metric is given by max{ sup_A[ inf_B( d(a,b) )], 
                                sup_B[ inf_A( d(a,b) )] }
    """
    
    distances = zeros((len(A),len(B)))
    for na,a in enumerate(A):
        for nb,b in enumerate(B):
            distances[na,nb] = d(a,b)
            
    d1 = max(distances.min(axis=0))
    d2 = max(distances.min(axis=1))
    return max([d1,d2])
    

def rHausdorff(A,d):
    """
    returns min_a1( max_a2 (d(a1,a2))) for all elements a1,a2 in A
    this is something like the "radius" (check ?)
    """
    distances = zeros((len(A),len(A)))
    for na1,a1 in enumerate(A):
        for na2,a2 in enumerate(A):
            distances[na1,na2] = d(a1,a2)
    
    # because of symmetry of "distances", it does not matter which axis to 
    # look first at -> can skip one axis, look only once
    return min(distances.max(axis=0))


def rHausdorff2(A,d):
    """
    returns max d(a1,a2) for all elements in A
    """
    distances = zeros((len(A),len(A)))
    for na1,a1 in enumerate(A):
        for na2,a2 in enumerate(A):
            distances[na1,na2] = d(a1,a2)
            
    return max(distances.flat)
    

def magicsq(N):    
    """
    copied from internet (lost the source...)
    Creates an N x N magic square.

    **Input:** 
        *N* -- an integer in some form, may be float or quotted.

    **Output:** 
        an ``'int32'`` *N* by *N* array -- the same magic square as in
        Matlab and Octave ``magic(N)`` commands.  In particular, the 
        Siamese method is used for odd *N* (but with a different 
        implementation.)

    """
    from pylab import tile,arange
    global _constant
    n = int(N)
    if n < 0 or n == 2:                    # consistent with Octave
        raise TypeError("No such magic squares exist.")
    elif n%2 == 1:
        m = n>>1
        b = n*n + 1
        _constant = n*b>>1
        return (tile(arange(1,b,n),n+2)[m:-m-1].reshape(n,n+1)[...,1:]+
              tile(arange(n),n+2).reshape(n,n+2)[...,1:-1]).transpose()
    elif n%4 == 0:
        b = n*n + 1
        _constant = n*b>>1
        d=arange(1, b).reshape(n, n)
        d[0:n:4, 0:n:4] = b - d[0:n:4, 0:n:4]
        d[0:n:4, 3:n:4] = b - d[0:n:4, 3:n:4]
        d[3:n:4, 0:n:4] = b - d[3:n:4, 0:n:4]
        d[3:n:4, 3:n:4] = b - d[3:n:4, 3:n:4]
        d[1:n:4, 1:n:4] = b - d[1:n:4, 1:n:4]
        d[1:n:4, 2:n:4] = b - d[1:n:4, 2:n:4]
        d[2:n:4, 1:n:4] = b - d[2:n:4, 1:n:4]
        d[2:n:4, 2:n:4] = b - d[2:n:4, 2:n:4]
        return d
    else:
        m = n>>1
        k = m>>1
        b = m*m
        d = tile(magicsq(m), (2,2))          # that changes the _constant
        _constant = _constant*8 - n - m     
        d[:m, :k] += 3*b
        d[m:,k:m] += 3*b
        d[ k,  k] += 3*b
        d[ k,  0] -= 3*b
        d[m+k, 0] += 3*b
        d[m+k, k] -= 3*b
        d[:m,m:n-k+1] += b+b
        d[m:,m:n-k+1] += b
        d[:m, n-k+1:] += b
        d[m:, n-k+1:] += b+b
        return d

def pseudoSpect(A, npts=200, s=2., gridPointSelect=100, verbose=True,
                lstSqSolve=True):
    """ 
    original code from http://www.cs.ox.ac.uk/projects/pseudospectra/psa.m
    % psa.m - Simple code for 2-norm pseudospectra of given matrix A.
    %         Typically about N/4 times faster than the obvious SVD method.
    %         Comes with no guarantees!   - L. N. Trefethen, March 1999.
    
    parameter: A: the matrix to analyze
               npts: number of points at the grid
               s: axis limits (-s ... +s)
               gridPointSelect: ???
               verbose: prints progress messages
               lstSqSolve: if true, use least squares in algorithm where
                  solve could be used (probably) instead. (replacement for
                  ldivide in MatLab)
    """
    
    from scipy.linalg import schur, triu
    from pylab import (meshgrid, norm, dot, zeros, eye, diag, find,  linspace,                       
                       arange, isreal, inf, ones, lstsq, solve, sqrt, randn,
                       eig, all)

    ldiv = lambda M1,M2 :lstsq(M1,M2)[0] if lstSqSolve else lambda M1,M2: solve(M1,M2)

    def planerot(x):
        '''
        return (G,y)
        with a matrix G such that y = G*x with y[1] = 0    
        '''
        G = zeros((2,2))
        xn = x / norm(x)
        G[0,0] = xn[0]
        G[1,0] = -xn[1]
        G[0,1] = xn[1]
        G[1,1] = xn[0]
        return G, dot(G,x)

    xmin = -s
    xmax = s
    ymin = -s
    ymax = s;  
    x = linspace(xmin,xmax,npts,endpoint=False)
    y = linspace(ymin,ymax,npts,endpoint=False)
    xx,yy = meshgrid(x,y)
    zz = xx + 1j*yy
     
    #% Compute Schur form and plot eigenvalues:
    T,Z = schur(A,output='complex');
        
    T = triu(T)
    eigA = diag(T)
    
    # Reorder Schur decomposition and compress to interesting subspace:
    select = find( eigA.real > -250)           # % <- ALTER SUBSPACE SELECTION
    n = len(select)
    for i in arange(n):
        for k in arange(select[i]-1,i,-1): #:-1:i
            G = planerot([T[k,k+1],T[k,k]-T[k+1,k+1]] )[0].T[::-1,::-1]
            J = slice(k,k+2)
            T[:,J] = dot(T[:,J],G)
            T[J,:] = dot(G.T,T[J,:])
          
    T = triu(T[:n,:n])
    I = eye(n);
    
    # Compute resolvent norms by inverse Lanczos iteration and plot contours:
    sigmin = inf*ones((len(y),len(x)));
    #A = eye(5)
    niter = 0
    for i in arange(len(y)): # 1:length(y)        
        if all(isreal(A)) and (ymax == -ymin) and (i > len(y)/2):
            sigmin[i,:] = sigmin[len(y) - i,:]
        else:
            for jj in arange(len(x)):
                z = zz[i,jj]
                T1 = z * I - T 
                T2 = T1.conj().T
                if z.real < gridPointSelect:    # <- ALTER GRID POINT SELECTION
                    sigold = 0
                    qold = zeros((n,1))
                    beta = 0
                    H = zeros((100,100))                
                    q = randn(n,1) + 1j*randn(n,1)                
                    while norm(q) < 1e-8:
                        q = randn(n,1) + 1j*randn(n,1)                
                    q = q/norm(q)
                    for k in arange(99):
                        v = ldiv(T1,(ldiv(T2,q))) - dot(beta,qold)
                        #stop
                        alpha = dot(q.conj().T, v).real
                        v = v - alpha*q
                        beta = norm(v)
                        qold = q
                        q = v/beta
                        H[k+1,k] = beta
                        H[k,k+1] = beta
                        H[k,k] = alpha
                        if (alpha > 1e100):
                            sig = alpha 
                        else:
                            sig = max(abs(eig(H[:k+1,:k+1])[0]))
                        if (abs(sigold/sig-1) < .001) or (sig < 3 and k > 2):
                            break
                        sigold = sig
                        niter += 1
                        #print 'niter = ', niter
                
                  #%text(x(jj),y(i),num2str(k))         % <- SHOW ITERATION COUNTS
                    sigmin[i,jj] = 1./sqrt(sig);
                #end
                #  end
        if verbose:
            print 'finished line ', str(i), ' out of ', str(len(y))
    
    return x,y,sigmin


def recolor(boxplt, set_color, lw=1):
    """
    recolors a boxplot (unicolor)
    """
    for key in boxplt.keys():
        for elem in boxplt[key]:
            elem.set_color(set_color)
            elem.set_linewidth(lw)
  

class data_container(object):
    """
    a simple data container object
    """
    def __init__(self,):
        """
        create some defauls data
        """
        # appendix f,cs, as indicate <f>ull(3D) or <s>agittal-plane(2D) data
        # aslip and cslip refer to acausal and causal SLIP states, which is
        # state[apex_i] + params[subsequent step] (acausal)  or
        # state[apex_i] + params[previous step] (causal), respectively
        self.vred_state_f = []
        self.vred_aslip_f = []
        self.vred_cslip_f = []
        self.vred_state_s = []
        self.vred_slip_s = []        
        self.vred_aslip_s = []
        self.vred_cslip_s = []
        self.std_vred_state_f = []
        self.std_vred_aslip_f = []
        self.std_vred_cslip_f = []
        self.std_vred_state_s = []
        self.std_vred_slip_s = []        
        self.std_vred_aslip_s = []
        self.std_vred_cslip_s = []        
        self.tot_var_f = []
        self.tot_var_s = []
        self.grf = []
        self.std_grf = []


class logger(object):
    """
    This class provides a logging object. It further supports the 
    __call__ - method.
    
    Methods overview
    ================
    
    See the methods' individual docstrings for more detailled documentation    
    
    .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Name        
        - Description
      * - **\__init__**         
        - initalizes the object, is called when the object is created
      * - **\__call__**        
        - is executed when the object is called directly
      * - **log**
        - displays the logging message
      * - **reset**        
        - resets both the internally stored log and timer
      * - **store**
        - stores the log in a file
      
      
    Example
    =======
    
    This class can be used as follows::
    
        LOG = logger()            # initializes the logger
        LOG('doing something...') # output: [time] *doing something*
        LOG.store(filename)       # stores the log in the file named *filename*
    
    """
    
    def __init__(self, ):
        self.time0 = time.time()
        self.logstrings = []
    
    def reset(self, ):
        """
        This function resets the LOG
        """
        self.__init__()
    
    def __call__(self, message):
        """
        This __call__ method redirects to the log method
        """
        self.log(message)
    
    def log(self, message):
        """
        This method displays the message including a timestamp.
        It further stores the log message, including the timestamp,
        in the internal storage.
        
        ===========
        Parameters:
        ===========
        message : `string`
           The log message to be displayed
           
        ========
        Returns:
        ========
        (nothing)
        """
        
        curtime = int(time.time() - self.time0)
        # compute h, m, s
        curh = curtime // 3600
        curm = mod(curtime // 60, 60)
        curs = mod(curtime, 60)
        
        timestring = 'time: %02i:%02i:%02i' % (curh, curm, curs)
        logstring = ' '.join([timestring, message])
        self.logstrings.append(logstring)
        print logstring
        
    def store(self, filename):
        """
        This method stores all logmessages in the file specified by
        *filename*        
        
        ===========
        Parameters:
        ===========
        filename: `string`
           the name of the file in which the log should be written
           
        ========
        Returns:
        ========
        (nothing)
        """
        try:
            with open(filename, 'w') as f:                
                f.writelines([x + '\n' for x in self.logstrings])
        except IOError as err:
            print 'Error: log could not be written!'
            pass
        
        

def int_f1(x, fs=1.):
    """
    A fourier-based integrator (simple version)

    ===========
    Parameters:
    ===========
    a : *array* (1D)
        The array which should be integrated
    fs : *float*
        sampling time of the data

    ========
    Returns:
    ========
    y : *array* (1D)
        The integrated array

    """  
    baseline = mean(x) * arange(len(x)) / float(fs)
    int_fluc = int_f0(x, float(fs))
    return int_fluc + baseline - int_fluc[0]

def expandLabels(markers):
    """
    expands a short list of labels (just the marker names) to the long list,
    i.e. a format that can be used as subjData.selection

    ===========
    Parameters:
    ===========
    markers : *list* of strings
        each element is the name of a marker, without the 'l' or 'r' denoting
        the side

    ========
    Returns:
    ========
    flist : *list* of strings
        each marker is splitted into left and right side and its position is
        relative to CoM. e.g. ['anl', ] -> ['r_anl_x - com_x', 'r_anl_y -
        com_y', ...]


    """
    selection = []
    for elem in markers:
        selection.append('l_' + elem + '_x - com_x')
        selection.append('r_' + elem + '_x - com_x')
        selection.append('l_' + elem + '_y - com_y')
        selection.append('r_' + elem + '_y - com_y')
        selection.append('l_' + elem + '_z - com_z')
        selection.append('r_' + elem + '_z - com_z')
    return selection




#def interp_f(x, n_new):
#    """
#    makes a fourier-based interpolation.
#    missing frequencies will be replaced with 0
#
#    ===========
#    Parameters:
#    ===========
#    x : *array* (1D)
#        the time series to be interpolated
#        **NOTE** it is assumed that x are evenly spaced in time
#    n_new : *integer*
#        the number of new (evenly spaced) sampling points
#
#    """
#    print '!!! WARNING !!! \n currently, this is incomplete.'
#    print '\nit *only* works reliably if len(x) is even and n_new > len(x)!!'
#    spec = fft.fft(x)
#    spec2 = zeros(n_new, dtype=complex)
#    if mod(len(x), 2) == 0:
#        # there is a single "central" frequency
#        cmax = min( len(x) / 2,  n_new // 2)
#        print 'cmax:', cmax
#        spec2[:cmax + 1] = spec[:cmax + 1]
#        spec2[-cmax:] = spec[-cmax:]
#        # one frequency is now on two points -> half weight
#        spec2[cmax] *= .5
#        spec2[-cmax] *= .5
#
#    else:
#        cmax = min( (len(x)-1) / 2 ,  n_new // 2)
#        spec2[:cmax] = spec[:cmax]
##        spec2[-cmax:] = spec[-cmax:]
#
#    return fft.ifft(spec2).real * float(n_new) / float(len(x))

def getApices(y):
    """ 
    returns the time (in frames) and position of initial and final apex
    height from a given trajectory y, which are obtained by fitting a cubic
    spline 
    
    ==========
    parameter:
    ==========
    y : *array* (1D)
        the trajectory. Should ideally start ~1 frame before an apex and end ~1
        frame behind an apex

    ========
    returns:
    ========

    [x0, xF], [y0, yF] : location (in frames) and value of the first and final
    apices

    """
    # the math behind here is: fitting a 2nd order polynomial and finding 
    # the root of its derivative. Here, only the results are applied, that's
    # why it appears like "magic" numbers
    c = dot(array([[.5, -1, .5], [-1.5, 2., -.5], [1., 0., 0.]]), y[:3])
    x0 = -1. * c[1] / (2. * c[0])
    y0 = polyval(c, x0)

    c = dot(array([[.5, -1, .5], [-1.5, 2., -.5], [1., 0., 0.]]), y[-3:])
    xF = -1. * c[1] / (2. * c[0])
    yF = polyval(c, xF)
    xF += len(y) - 3

    return [x0, xF], [y0, yF]

def map_eq(t, fs=250., eps=1e-8):
    """
    returns the longest array of time which has support points every 1 / fs ms
    that is not smaller than min[t] and does not exceed max[t]

    ==========
    Parameter:
    ==========
    t : *array*
        vector that contains a smallest and largest element.
    fs : *float*
        sampling time of new time vector
    eps : *float* (internally used!)
        

    ========
    Returns:
    ========
    teq : *array*
        an equally sampled array with a sampling rate of fs
    """
    t_min = ceil(min(t) * fs - eps)
    t_max = ceil(max(t) * fs + eps)
    return 1. / fs * arange(t_min, t_max + 1)
    
 
def sigdiff(t, y, t_ref, y_ref):
    """
    returns the difference y - y_ref at the corresponding times of t.  y can be
    sampled at different times t_ref.    

    ==========
    Parameter:
    ==========
    t : *array* (1-D)
        time vector of the signal y (must have same length as y)
    y : *array* (1-D)
        input signal from which y_ref should be subtracted.

    ========
    Returns:
    ========    
        y' : y - y_ref at times t
    """
    y_1 = interp(t, t_ref, y_ref)
    return y - y_1


def mpow(A, n):
    """ 
    Returns the n-th power of A.
    If n is no integer, the next smaller integer will be assumed.

    ==========
    Parameter:
    ==========
    A : *array*
        the square matrix from which the n-th power should be returned
    n : *integer*
        the power

    ========
    Returns:
    ========
    B : *array*
        B = A^n

    """
    return reduce(dot, [eye(A.shape[0]), ] * 2 + [A, ] * int(n))


def mpow2(A, n):
    """ 
    Returns the n-th power of A.
    Here, this is computed using eigenvalue decomposition.
    
    ==========
    Parameter:
    ==========
    A : *array*
        the square matrix from which the n-th power should be returned
    n : *integer*
        the power

    ========
    Returns:
    ========
    B : *array*
        B = A^n

    """
     
    D, L = eig(A)
    if isreal(A).all():
        return reduce(dot, [L, diag(D**n), inv(L)]).real
    else:
        return reduce(dot, [L, diag(D**n), inv(L)])


def cov_ar(A, n=200):
    """ 
    returns the covariance matrix of an AR(1)-process of the following form:
    x_(n+1) = A * x_n + eta,
    where eta is iid noise. 
    The result can be computed by sum_k=0^infinity [A^k A.T^k]
    
    ==========
    Parameter:
    ==========
    A : *array*
        the system matrix of the AR(1)-process
    n : *integer*
        number of matrix powers to compute before convergence is assumed.

    ========
    Returns:
    ========
    P : *array*
        the positive-semidefinite symmetric covariance matrix of the process
        output

    """

    d = ([dot(mpow(A, n), mpow(A.T, n)) for n in arange(190)])  
    return reduce(lambda x, y: x + y, d)

class Struct(object):
    """ 
    This class provides struct-like access to a dictionary.

    Usage Example::

        mydict = {'a' : 34, 'key2' : 'string' } # define a sample dictionary
        mystruct = Struct(mydict) # transform into struct-like object
        print 'a:', mystruct.a   # prints 'a: 34'
        print 'key2' : mystruct.key2 # prints 'string'


    """

    def __init__(self, dictionary):
        """

        Parameters:
        ===========
        dictionary : *dict*
            the dictionary you want access to

        """
        self.__dict__.update(dictionary)
    # other variant: use **dictionary in parameter definition. Then, to use the
    # code, use: s = Struct(**mydict) 
    

def sphinx_test(arg):
    """
    This function is just a documentation test for sphinx (auto-doc generation)

    .. warning::

      This code has no use!

    .. note::

        well, you can test notes

    .. seealso::

      another resource to see would be the web, indeed





    """
    pass



def upper_phases(phases, sep=pi, return_indices=False):
    """
    returns only phases in the "upper half": x if pi <= x mod (2*pi) < 2 *pi

    *NOTE* by choosing "sep=0", this function returns the _lower_ phases instead!

    :args:
        phases: an iterable object, interpreted as phases to sort
        sep (float, default: pi): the separation between "low" and "high". Note that 
             the intervals will be "wrapped around 2 pi", and will always be of width 
             pi.
             This is equivalently to shifting every element of phases by  -x (-pi)
        return_indices (bool): return indices instead of values
    """
    if not return_indices:
        up = [x for x in phases if mod(x - sep + pi, 2.*pi) >= pi]
    else:
        up = [idx for idx, x in enumerate(phases) if mod(x - sep + pi, 2.*pi) >= pi]
    return up

def calcJacobian(fun, x0, h=.0001):
    """
    calculates the jacobian of a given function fun with respect to its
    parameters at the point (array or list) x0.

    :args:
        fun (function): the function to calcualte the jacobian from
        x0 (iterable, e.g. array): position to evaluate the jacobian at
        h (float): step size 

    :returns:
        J (n-by-n array): the jacobian of f at x0
    """
    J = []
    x = array(x0)
    for elem, val in enumerate(x0):
        
        ICp = x.copy()
        ICp[elem] += h
        resp = fun(ICp)
        ICn = x.copy()
        ICn[elem] -= h
        resn = fun(ICn)
        J.append((resp - resn)  / (2. * h))
        
    J = vstack(J).T    
    return J

def dt_movingavg(data, tailLength, use_median=False):
    """
    removes a trend in the data using a moving average.
    the parameter tailLength determines how many datapoints are taken for
    averaging in positive and negative direction each (total: 2*tailLength
    + 1 datapoints)
    operates along the first axis

    :args:
        data (n-by-d array): the data to be detrended.
        tailLength (int): Length of tail of a half of detrending window
        use_median (bool): Use median instead of mean (slow!!)

    :returns:
        dt_data (n-by-d array): data after the moving average has been 
            removed (a copy of original data)
    
    """

    def mov_avg(data, tailLength):
        """
        returns the moving average for a 1D array data
        """
        data1 = concatenate((data[tailLength:0:-1],data,data[-tailLength:]))
        #print "mov avg idat shape:", data1.shape, "tailLength:", tailLength
        avgFilter = array([1./(2.*tailLength+1.),]*(2*tailLength + 1))
        #print "avgFilter:", avgFilter.shape
        res = convolve(data1,avgFilter)[2*tailLength:-2*tailLength]
        #print "mov avg shape:", res.shape
        return res

    
    def my_medfilt(data, tailLength):
        """
        returns the median-filtered data; edges are "extrapolated" (constant)
        """
        data1 = hstack([data[tailLength:0:-1], data, data[-tailLength:]])
        out = medfilt(data1, 2 * tailLength + 1)
        return out[tailLength:-tailLength]



    if data.ndim > 2:
        raise ValueError, 'Error: detrending only for 1- and 2dim-data'        
    if data.ndim == 2:
        res = []
        if use_median:
            res = [data[:,dim] - my_medfilt(data[:,dim], tailLength) for
                    dim in arange(data.shape[1])]
            #res = [data[:,dim] - medfilt(data[:,dim], 2 * tailLength + 1) for
            #        dim in arange(data.shape[1])]
        else:
            #print "data shape:", data.shape
            for dim in arange(data.shape[1]):
                res.append(data[:,dim]-mov_avg(data[:,dim],tailLength))
        res = vstack(res).T
    else:
        if use_median:
            res = data - my_medfilt(data, tailLength)
        else:
            res = data - mov_avg(data,tailLength)
        
    return res        

def get_minmax(data, deg=2):
    """
    returns the interpolated extremum and position (in fractions of frames)

    :args:
        data (iterable): data to be fitted with a <deg> degree polynomial
        deg (int): degree of polynomial to fit

    :returns:
        val, pos: a tuple of floats indicating the position
    """

    x = arange(len(data))
    p = polyfit(x, data, deg)
    d = (arange(len(p))[::-1] * p)[:-1]
    r = roots(d)
    cnt = 0
    idx = None
    for nr, rx in enumerate(r):
        if isreal(r):
            idx = nr
            cnt +=1
            if cnt > 1:
                raise ValueError("Too many real roots found." + 
                        "Reduce order of polynomial!")
    x0 = r[nr].real
    return x0, polyval(p, x0)

def run_nbcells(nbfile, cell_ids, run_other=False, fileformat=3):
    """
    Runs specific cells from another notebook. In the specified notebook, a cell label can be defined
    in a comment like this:
    
    # cell_ID 1c
    
    This give the cell the label 1c
    
    :args:
        nbfile (str): name of the file to open (typically an ipython notebook)
        cell_ids (list): a list of cell id's (strings) to be looked for
        run_other (bool): run cells if not all required cells have been found
        fileformat(int): default=3, the file format of the notebook


        OLD: fileformat (str): 'json' (default), 'ipynb' or 'py' specifies the format of the file.
        
    :returns:
        None
    
    :raises:
        NameError if not all cells could be found and run_other is set to False
    
    """
    
    # how to change
    with open(nbfile) as f:
        nb = nbformat.read(f, fileformat)
    
    codecells = []
    found_ids = []
    ws = nb['worksheets'][0] # the "worksheet" (what's the difference between worksheet and notebook?)
    for cell in ws.cells:
        # cell has cell_type, input ("code"), output
        # get cell ID for every cell
        if cell.cell_type == 'code':
            for line_ in cell.input.splitlines():
                id_found = False
                if line_.startswith('#') and 'cell_ID' in line_:
                    m = re.search('cell_ID\s*([0-9A-Za-z][0-9A-Za-z._]*)', line_)
                    if m is not None:
                        if len(m.groups()) > 0:                        
                            cell_id = m.groups(0)[0].strip()
                            id_found = True                            
                            if cell_id in cell_ids:
                                codecells.append(cell.input)
                                found_ids.append(cell_id)
    
    allfound = set(found_ids) == set(cell_ids)
    if allfound:    
        ip = get_ipython()
        for code in codecells:        
            ip.run_cell(code)
    elif run_other:
        print "The following labels have not been found:"
        print ', '.join(set(cell_ids) - set(found_ids))
        ip = get_ipython()
        for code in codecells:        
            ip.run_cell(code)
    else:
        print "The following labels have not been found:"
        print ', '.join(set(cell_ids) - set(found_ids))
        raise NameError("some cells could not be identified")
    
            
            
