# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:57:25 2011

@author: moritz

This module contains convenience functions for data analysis of Floquet data
"""

import fmalibs.phaser2 as phaser2
from pylab import (vstack, pi, interp, ceil, floor, linspace, zeros, hstack,
                  rand, lstsq, svd, diag, dot, var, randint, randn, find,
                  iscomplex, isreal, inv, log10,
                  log, exp, logspace, sort, cumsum, mean, polyval, polyfit,
                  std, tanh, gradient,
                  permutation, newaxis,
                  figure, plot, xlabel, ylabel, title, legend)
from numpy import arange, array, concatenate, diff
import scipy.fftpack as fftpack
from scipy import convolve
from scipy.integrate import cumtrapz
from numpy import ndarray
import numpy as np
import mutils.io as mio
import mutils.misc as mi
# this function was formerly in this module here... so keep it available!
from mutils.misc import dt_movingavg 
from mutils.statistics import DFA, visDFA, vRedPartial, otheridx, fitData

   
import warnings

def RecalcPhase(data):    
    """
    this function re-computes the phase of a given artificial Floquet system
    analog to the way the phases are computed in the original systen.
    
    Data must be given in a D x N-format, 
        D: number of Dimensions, 
        N: number of samples
    """
    
    p1 = -1.*data[0,:]              # R_Kne_y - L_kne_y
    p2 = data[5,:] - data[2,:]      # R_Trc_y - R_Anl_y
    p3 = data[6,:] - data[1,:]      # L_trc_y - L_Anl_y
    p4 = data[2,:] - data[1,:] - data[0,:]   # R_anl_y - L_anl_y   
    
    allPhrIn_TDR = [vstack((p1,p2,p3,p4)),]
    psec_TDR = [p4.copy(),]    
    phrIn = vstack((p1,p2,p3,p4))
    psecIn = p4.copy()  

    print 'building phaser ...\n'
    Phaser = phaser2.Phaser2(y = allPhrIn_TDR, psecData = psec_TDR)
    print 'computing phases ...\n'
    return Phaser.phaserEval(phrIn,psecIn).squeeze()

def createSimilarAR(data):
    """
    creates an AR-process that is similar to a given data set.
    data must be given in n x d-format
    """
    # step 1: get "average" fit matrix
    l_A = []
    for rep in arange(100):
        idx = randint(0,data.shape[0]-1,data.shape[0]-1)
        idat = data[idx,:]
        odat = data[idx+1,:]
        l_A.append(lstsq(idat,odat)[0])

    sysmat = meanMat(l_A).T
    # idea: get "dynamic noise" from input data as difference of
    # expected vs. predicted data:
    # eta_i = (sysmat*(data[:,i-1]).T - data[:,i])
    # however, in order to destroy any possible correlations in the
    # input noise (they would also occur in the output), the
    # noise per section has to be permuted.
    prediction = dot(sysmat,data[:-1,:].T)
    dynNoise = data[1:,:].T - prediction
    res = [zeros((dynNoise.shape[0],1)), ]
    for nidx in permutation(dynNoise.shape[1]):
        res.append( dot(sysmat,res[-1]) + dynNoise[:,nidx][:,newaxis] )
    
    return hstack(res).T


    


def ReSample(data,phase,nps):
    """
    re-samples the data with nps frames per stride.
    phase gives the phase information at a given cycle.
    the output will be truncated towards an integer number of strides.
    
    The last stride will usually be removed.
    
    Data must be given in a D x N-format, 
        D: number of Dimensions, 
        N: number of samples
    """ 
    minPhi = ceil(phase[0]/(2.*pi))*2.*pi
    maxPhi = floor(phase[-1]/(2.*pi))*2.*pi
    nFrames = round((maxPhi-minPhi)/(2.*pi)*nps)
    phi_new = linspace(minPhi, maxPhi, nFrames, endpoint=False)
    return vstack([interp(phi_new,phase,data[row,:]) for row in range(data.shape[0])])
    
 
def varRed(idat,odat,A,bootstrap = None):
    """
    computed the variance reduction when using A*idat[:,x].T as predictor for odat[:,x].T
    if bootstrap is an integer > 1, a bootstrap with the given number of iterations
    will be performed.
    returns
    tVred, sVred: the total relative variance after prediction (all coordinates)
       and the variance reduction for each coordinate separately. These data are
       scalar and array or lists of scalars and arrays when a bootstrap is performed.
       
    Note: in the bootstrapped results, the first element refers to the "full" 
    data variance reduction.
    """
    
    nBoot = bootstrap if type(bootstrap) is int else 0
    if nBoot < 2:
        nBoot = 0        
    
    odat_pred = dot(A,idat.T)
    rdiff = odat_pred - odat.T # remaining difference
    rvar = var(rdiff,axis=1)/var(odat.T,axis=1) # relative variance
    trvar = var(rdiff.flat)/var(odat.T.flat)    # total relative variance
    
    if nBoot > 0:
        rvar = [rvar,]
        trvar = [trvar,]
    for rep in range(nBoot-1):
        indices = randint(0,odat.T.shape[1],odat.T.shape[1])
        odat_pred = dot(A,idat[indices,:].T)
        rdiff = odat_pred - odat[indices,:].T # remaining difference
        rvar.append( var(rdiff,axis=1)/var(odat.T,axis=1) ) # relative variance
        trvar.append (var(rdiff.flat)/var(odat.T.flat) )    # total relative variance
        
    return trvar, rvar
    

    

def reduceDim(fullmat,n=1):
    """
    reduces the dimension of a d x d - matrix to a (d-n)x(d-n) matrix, 
    keeping the largest eigenvalues unchanged.
    """
    u,s,v = svd(fullmat)
    return dot(u[:-n,:-n],dot(diag(s[:-n]),v[:-n,:-n]))
    
def lroll(mylist, n):
    """
    returns a list that is "rolled" by n items
    """
    if type(mylist) is not list:
        raise ValueError, 'a list is required'
    return [x for x in mylist[n:] + mylist[:n]]
    

def reduceDimDat(fulldat,n=1):
    """
    reduces the dimension of a given data set by removing the 
    lowest principal component.
    data must be given in D X N - format 
      (D: dimension, N: number of measurements)
    """
    raise NotImplementedError, \
          'Wait a minute - this function in raw form does not make much sense here ...'
    u,s,v = svd(fulldat, full_matrices = False)
    

#def dt_movingavg(data,tailLength):
#    """
#    maps to mutils.misc.dt_movingavg
#    """
#    return mi.dt_movingavg(data, tailLength)

 
def createBigMap(sectionMappings):
    """
    input: list of mappings [a1,a2, ..., an]
    creates a permutation matrix out of section mappings in the form
    | 0    ....     an |
    | a1 0   ...     0 |
    | 0  a2   ...    0 |
    | ...      ...     |
    | 0 0 ... a(n-1) 0 |
    , where A = an*a(n-1)*...*a2*a1 is the full-stride map

    NOTE: the full stride map is *NOT* a1*a2*...*an, but the product in reversed order!
    """
    dim0 = sectionMappings[0].shape[0]
    matSize = len(sectionMappings)*dim0    
    mat0 = zeros( (matSize, matSize))
    for n, mapping in enumerate(sectionMappings):
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
    

def meanMat(list_of_matrices):
    """
    returns the element-wise mean of the given list of matrices
    """
    return reduce(lambda x,y: x+y, list_of_matrices)/float(len(list_of_matrices))
    
    
    
    
def fitData_2s(idat, odat, nps, nidx = None, nrep = 500, rcond = 1e-3):
    """
    fits the data using two strides ahead
    
    - TODO: rewrite this docstring!
    performs a bootstrapped fit of the data, i.e. gives a matrix X that minimizes
    || odat.T - X * idat.T ||.
    This matrix X is the least squares estimate that maps the column vector 
    representing stride k to the column vector of k+1, at a / the given
    Poincare section(s).
    
    idat, odat: input-data in 1d-format (see e.g. twoD_oneD for description)
       each row in idat and odat must represent a stride (idat) and the subsequent 
       stride (odat), i.e. if you want to use all strides, odat = idat[1:,:]
       However, odat must not be shorter (fewer rows) than idat.
       
    nps: numbers of samples per stride
    
    nidx: number of strides that should be used for a fit in each bootstrap 
       iteration. If omitted, idat.shape[0]*2/3 is used
       
    nrep: how many bootstrap iterations should be performed
    
    sections: list of integers that indicate which sections should be used as
       intermediate mappings. If only a single section is given, a "stride-map" 
       is computed
       
    rcond: the numerical accuracy which is used for the fit (this value is
       passed through to lstsq). Too high values will cause loss of detection of
       lower eigenvalues, too low values will corrupt the return maps.
       
       
    returns a triple: 
       (1) a list of a list of matrices. Each list containts the matrices that 
       map from the first given section to the next.
       *NOTE* if len(sections) is 1, then this is left empty (equivalen matrices
       are then in (2))
       (2) a list of matrices, which represent the "full stride" fits for
       the given set of indices.       
       (3) a list of a list of indices. Each list lists the indices
       (rows of idat) that were used for the regression.
    """    
    if nidx is None:
        nidx = int(2./3.*idat.shape[0] )
        
    #if any(diff(sections)) < 0:
    #    raise ValueError, 'sections must be given in increasing order!'
    
    # 1st: create bootstrap indices
    indices = [randint(1,idat.shape[0],nidx) for x in range(nrep)]
    #[(idat.shape[0]*rand(nidx-1)+1).astype(int) for x in range(nrep)]

    # 2nd: create section fits (if sections)    
    # part A: do fits from one section to the next (within same stride)    
   # sectMaps = [ 
   #               [
   #                  lstsq( idat[idcs,sections[sct]::nps],
   #                         idat[idcs,sections[sct+1]::nps], rcond=rcond)[0].T
   #               for idcs in indices ]
   #            for sct in range(len(sections)-1) ]
                         
    # part B: do fits from last section to first section of next stride
    #if len(sections) > 1:
    #    sectMaps.append( [
    #                        lstsq( idat[idcs,sections[-1]::nps],
    #                        odat[idcs,sections[0]::nps], rcond=rcond)[0].T
    #              for idcs in indices ]  )

    # 3rd: create stride fits
    strideMaps = [ lstsq(hstack([idat[idcs-1,0::nps],idat[idcs,0::nps]]), 
                         odat[idcs,0::nps], rcond=rcond)[0].T
                         for idcs in indices ]
    
    return strideMaps, indices


def blockdiag(blocks):
    """
    returns a block-diagonal matrix with the given blocks
    """
    # 1st: compute required dimensions
    dim = 0
    for elem in blocks:
        # elem is block
        if type(elem) is ndarray:
            if elem.shape[0] != elem.shape[1]:
                raise ValueError('blocks must be numbers or square matrices')
            dim += elem.shape[0]
        # elem is number
        else: 
            dim += 1
    
    block = zeros((dim,dim),dtype=complex)
    nRow = 0
    for elem in blocks:
        # elem is block
        if type(elem) is ndarray:
            block[nRow:nRow+elem.shape[0],nRow:nRow+elem.shape[0]] = elem.copy()
            nRow += elem.shape[0]
        # elem is number
        else: 
            block[nRow,nRow] = elem
            nRow += 1
            
    if max(abs(block.flatten().imag)) == 0:
        block = block.real
    return block
    


def create_cm(dim,eigList1 = None, eigList2 = None):
    """
    returns two real-valued commuting matrices of dimension dim x dim
    the eigenvalues of each matrix can be given; single complex numbers will be
    interpreted as pair of complex conjuates.
    With this restriction, the (internally augmented) lists must have the length
    of dim
    """
    if eigList1 is None:
        eigList1 = rand(dim)
    if eigList2 is None:
        eigList2 = rand(dim)
        
    # order 1st array such that complex numbers are first
    EL1 = array(eigList1)    
    imPos1 = find(iscomplex(EL1))
    rePos1 = find(isreal(EL1)) # shorter than set comparisons :D
    EL1 = hstack([EL1[imPos1],EL1[rePos1]])
    # order 2nd array such that complex numbers are last
    EL2 = array(eigList2)    
    imPos2 = find(iscomplex(EL2))
    rePos2 = find(isreal(EL2)) # shorter than set comparisons :D
    EL2 = hstack([EL2[rePos2],EL2[imPos2]])
    # now: make eigenvalues of list #2, where a block is in list #1, 
    # pairwise equal, and other way round
    EL2[1:2*len(imPos1):2] = EL2[0:2*len(imPos1):2]    
    EL1[-2*len(imPos2)+1::2] = EL1[-2*len(imPos2)::2]
    
    if len(imPos2)*2 + len(imPos1)*2 > dim:
        raise ValueError(
           'too many complex eigenvalues - cannot create commuting matrices')

    # augment lists
    ev1 = []       
    nev1 = 0
    for elem in EL1:
        if elem.imag != 0.:
            ev1.append( array( [[elem.real, -elem.imag],
                               [elem.imag,  elem.real]]))
            nev1 += 2
        else:
            ev1.append(elem)
            nev1 += 1
            
    if nev1 != dim:
        raise ValueError(
          'number of given eigenvalues #1 (complex: x2) does not match dim!')
            
    
    ev2 = []
    nev2 = 0
    for elem in EL2:
        if elem.imag != 0.:
            ev2.append( array( [[elem.real, -elem.imag],
                               [elem.imag,  elem.real]]))
            nev2 += 2
        else:
            ev2.append(elem)
            nev2 += 1
    
    if nev2 != dim:
        raise ValueError(
          'number of given eigenvalues #2 (complex: x2) does not match dim!')
        


    
    u,s,v = svd(randn(dim,dim))
    # create a coordinate system v that is not orthogonal but not too skew
    v = v + .2*rand(dim,dim) - .1
    
    cm1 = dot(inv(v),dot(blockdiag(ev1),v))
    cm2 = dot(inv(v),dot(blockdiag(ev2),v))
    # create block diagonal matrices
    
    return cm1, cm2



def ComputeCoM(kinData, forceData, kin_est=None, mass=None, f_thresh=1., steepness=10.,
        use_Fint=False, return_mass=False, adapt_kin_mean=True):
    """
    Computes the CoM motion using a complementary filter as described in Maus
    et al, J Exp. Biol. (2011).

    :args:
        kinData: dict or mutils.io.saveable object with kinematic data. Should
            contain "fs" field
        forceData: dict or mutils.io.saveable object with force data. Should
            contain "fs" field. 
        kin_est (optional, d-by-1 array): if present, the kinematic estimate of
            the CoM (x,y,z direction). Overrides missing kinData (can be empty,
            i.e. {}, then). Data will be (Fourier-)interpolated, so can have a
            different sampling frequency than the force data.
        mass (float): the subject's mass. If omitted, it is automatically determined.
        f_thresh (float): threshold frequency (recommended: slightly below
            dominant frequency)
        steepness (float): steepness of the frequency threshold (1/Hz).
        use_Fint (bool, default=False): whether or not to use a fourier-based
            integration scheme (zero-lag)
        return_mass(bool): whether or not to additionally return the mass
	adapt_kin_mean (bool): wether or not to adapt the combined mean to the
            kinematic  mean

    :returns:
        Force, CoM: Arrays which contain physically consistent GRF and CoM
            data.

    """

    if type(kinData) == dict:
        kd = mio.saveable(kinData)
    elif type(kinData) == mio.saveable:
        kd = kinData
        
    if type(forceData) == dict:
        fd = mio.saveable(forceData)
    elif type(forceData) == mio.saveable:
        fd = forceData

    if fd.fs < 1:
        warnings.warn(
            "warning: fs is the sampling FREQUENCY, not the sampling time" +
            "\nAre you sure your sampling frequency is really that low?")


    def weighting_function(freq_vec, f_changeover, sharpness):
        """ another weighting function, using tanh to prevent exp. overflow """
        weight = .5 - .5*tanh((freq_vec.squeeze()[1:] - f_changeover) * sharpness)
        weight = (weight + weight[::-1]).squeeze()
        weight = np.hstack([1., weight])
        return weight

    def kin_estimate(selectedData):
        """
        calculates the kinematic CoM estimate from "selectedData"
        """
        # anthropometry from Dempster
        # format: [prox.marker, dist. marker,  rel. weight (%),
        # segment CoM pos rel. to prox. marker (%) ]
        aData = [
            ('R_Hea','L_Hea',8.26,50.), 
            ('L_Acr','R_Trc',46.84/2.,63.),
            ('R_Acr','L_Trc',46.84/2.,63.),
            ('R_Acr','R_Elb',3.25,43.6), 
            ('R_Elb','R_WrL',1.87 + 0.65,43. + 25.),
            ('L_Acr','L_Elb',3.25,43.6),
            ('L_Elb','L_WrL',1.87 + 0.65,43. + 25.),
            ('R_Trc','R_Kne',10.5,43.3),
            ('R_Kne','R_AnL',4.75,43.4),
            ('R_Hee','R_Mtv',1.43,50. + 5.),
            ('L_Trc','L_Kne',10.5,43.3),
            ('L_Kne','L_AnL',4.75,43.4),
            ('L_Hee','L_Mtv',1.43,50. + 5.)
            ]
    
        # adaptation to dataformat when extracted from database: lowercase
        aData = [(x[0].lower(), x[1].lower(), x[2], x[3]) for x in aData]
        CoM = np.zeros((len(kd.sacr[:,0]),3))
        for segment in aData:
            CoM += segment[2]/100.* (
                (getattr(selectedData, segment[1]) - getattr(selectedData,
                    segment[0])) * segment[3]/100. + getattr(selectedData,
                        segment[0]) )

        return CoM


    elems = dir(fd)
    # get convenient names for forces
    if 'fx' in elems:
        Fx = fd.fx.squeeze()
    elif 'Fx' in elems:
        Fx = fd.Fx.squeeze()
    elif 'fx1' in elems and 'fx2' in elems:
        Fx = (fd.fx1 + fd.fx2).squeeze()
    else:
        raise ValueError("Error: Fx field not in forces (fx or fx1 and fx2)")

    if 'fy' in elems:
        Fy = fd.fy.squeeze()
    elif 'Fy' in elems:
        Fy = fd.Fy.squeeze()
    elif 'fy1' in elems and 'fy2' in elems:
        Fy = (fd.fy1 + fd.fy2).squeeze()
    else:
        raise ValueError("Error: Fy field not in forces (fy or fy1 and fy2)")

    if 'fz' in elems:
        Fz = fd.fz.squeeze()
    elif 'Fz' in elems:
        Fz = fd.Fz.squeeze()
    elif all([x in elems for x in ['fz1', 'fz2', 'fz3', 'fz4']]):
        Fz = (fd.fz1 + fd.fz2 + fd.fz3 + fd.fz4).squeeze()
    elif all([x in elems for x in ['fzr1', 'fzr2', 'fzr3', 'fzr4',
        'fzl1', 'fzl2', 'fzl3', 'fzl4',]]):
        Fz = (fd.fzr1 + fd.fzr2 + fd.fzr3 + fd.fzr4 + 
            fd.fzl1 + fd.fzl2 + fd.fzl3 + fd.fzl4).squeeze() # remove bodyweight later
    else:
        raise ValueError(
        "Error: Fz field not in forces (fz or fz1..4 or fzr1..4 and fzl1...4)"
            )



    if kin_est is None:
        kin_est = kin_estimate(kd)
    kin_est_i = vstack([mi.interp_f(kin_est[:, col].copy(), len(Fz), detrend=True) for
        col in range(3)]).T

    if use_Fint:
        diff_op = lambda x: mi.diff_f(x.squeeze(), fs=fd.fs)
        int_op = lambda x, x0: mi.int_f(x.squeeze(), fs=fd.fs) + x0
    else:
        diff_op = lambda x: gradient(x.squeeze()) * fd.fs
        int_op = lambda x, x0: cumtrapz(x.squeeze(), initial=0) / fd.fs + x0
    
    vd = vstack([diff_op(kin_est_i[:,col]) for col in range(3)]).T
    
    # correct mean forces, pt 1
    n = 2 # for higher reliably use multiple points
    T = len(Fz) / fd.fs
    dvx = ((kin_est_i[-1,0] - kin_est_i[-(n+1),0]) - (kin_est_i[n,0] -
        kin_est_i[0,0])) * (fd.fs / n)
    dvy = ((kin_est_i[-1,1] - kin_est_i[-(n+1),1]) - (kin_est_i[n,1] -
        kin_est_i[0,1])) * (fd.fs / n)
    dvz = ((kin_est_i[-1,2] - kin_est_i[-(n+1),2]) - (kin_est_i[n,2] -
        kin_est_i[0,2])) * (fd.fs / n)

    ax = dvx/T
    ay = dvy/T
    az = dvz/T + 9.81

    #print "dvz = ", dvz
    #print "az = ", az
    

    if mass == None: # determine mass
        mass = array(mean(Fz)/az).squeeze()
        Fz -= mass*9.81
    else:
        Fz -= mean(Fz) 
        Fz += (az - 9.81)*mass
    Fx = Fx - mean(Fx) + ax*mass
    Fy = Fy - mean(Fy) + ay*mass
        #print "estimated mass:", mass
    

    vi = vstack([int_op(F / mass, v0 ) for F, v0 in zip([Fx, Fy, Fz],
        vd[0,:])]).T

    spect_diff = vstack([fftpack.fft(vd[:, col]) for col in range(3)]).T
    spect_int = vstack([fftpack.fft(vi[:, col]) for col in range(3)]).T
    
    freq_vec = linspace(0, fd.fs, len(Fz), endpoint=False)
    wv = weighting_function(freq_vec, f_thresh, steepness)

    spect_combine = vstack([wv*spect_diff[:, col] + (1.-wv)*spect_int[:, col]
        for col in range(3)]).T
    
    v_combine = vstack([fftpack.ifft(spect_combine[:, col]).real for col in
        range(3)]).T

    x_combine = vstack([int_op(v_combine[:, col], kin_est[0, col]) for col in
        range(3)]).T

    if adapt_kin_mean:
        x_combine = x_combine - mean(x_combine, axis=0)
        for dim in range(x_combine.shape[1]):
             x_combine[:,dim] += mean(kin_est[:,dim])
    f_combine = vstack([diff_op(v_combine[:, col])*mass for col in
        range(3)]).T
    f_combine[:, 2] += mass*9.81

    if return_mass:
        return f_combine, x_combine, mass
    return f_combine, x_combine


