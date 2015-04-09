# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:13:53 2011

.. module:: statistics
    :synopsis: Several functions for data fitting (regression) and analysis,
    e.g. DFA

.. moduleauthor: Moritz Maus <moritz@hm10.net>




"""

import numpy as np
import numpy.random as rnd
from pylab import (svd, pinv, dot, diag, sum, cumsum, std, vstack, mean, sqrt,
        array, sort, logspace, exp, arange, polyfit, polyval, var, median,
        find)
from pylab import (figure, plot, log10, log, xlabel, ylabel, title,
    legend, isnan, diff, rand, randn, lstsq)
#from mutils.FDatAn import fitData,

def ar_process(n,alpha):
    """
    creates a AR(1)-Process
    alpha: can be float(1D) or array (NxN)
    """
    if type(alpha) is not np.ndarray:
        psi = np.array([alpha])
    else:
        psi = alpha
    if psi.ndim == 2:
        if psi.shape[0] != psi.shape[1]:
            raise ValueError, 'shape mismatch of sytem matrix alpha'
    #print psi, psi.shape
    res = np.zeros((psi.shape[0],n+1))
    xi = rnd.randn(psi.shape[0],n)
    for x in xrange(n):
        res[:,x+1:x+2]=np.dot(psi,res[:,x:x+1]) + xi[:,x:x+1]
    return res[:,1:]

def ar_similar(x):
    """
    returns an ar(1)-process that has the same autocorrelation(1), same
    amplitude (std) and mean as the given 1D vector x

    parameters
    ----------
    x : *array*
        The data to which a similar ar(1)-process is sought

    returns
    -------
    y : *array*
        Result of a random ar(1)-process with same correlation coefficient as x


    """
    dat = x - mean(x)
    alpha = dot(dat[1:], dat[:-1]) / sqrt(dot(dat[1:], dat[1:]) * dot(dat[:-1],
        dat[:-1]))
    res = ar_process(len(dat), alpha).squeeze()
    res = res - mean(res)
    return res / std(res) * std(dat) + mean(x)



def find_factors(idat, odat, k = None):
    """
    A routine to compute the main predictors (linear combinations of
    coordinates) in idat to predict odat.

    *Parameters*
        idat: d x n data matrix,
            with n measurements, each of dimension d
        odat: q x n data matrix
             with n measurements, each of dimension q

    *Returns*
        **Depending on whether or not** *k* **is provided, the returned
        value is different**

      * if k is given, compute the first k regressors and return an orthogonal
        matrix that contains the regressors in its colums,
        i.e. reg[0,:] is the first regressor

      * if k is not given or None, return a d-dimensional vector v(k)
        explaining which fraction of the total predictable variance can be
        explained using only k regressors.

    **NOTE**


    #. idat and odat must have zero mean
    #. To interpret the regressors, it is advisable to have the
       for columns of idat having the same variance
    """
    # transform into z-scores
    u, s, v = svd(idat, full_matrices = False)
    su = dot(diag(1./s), u.T)

    z = dot(su,idat)
    # ! Note that the covariance of z is *NOT* 1, but 1/n; z*z.T = 1 !

    # least-squares regression:
    A = dot(odat, pinv(z))

    uA, sigma_A, vA = svd(A, full_matrices = False)
    if k is None:
        vk = cumsum(sigma_A**2) / sum(sigma_A**2)
        return vk

    else:
    # choose k predictors
        sigma_A1 = sigma_A.copy()
        sigma_A1[k:] = 0
        A1 = reduce(dot, [uA, diag(sigma_A1), vA])
        B = dot(A1, su)
        uB, sigma_B, vB = svd(B, full_matrices = False)
        regs = vB[:k,:].T
        return regs




def yule_walker(x):
    """
    computes the yule-walker-prediction for an 1D-AR(1)-process
    """
    idat = x[1:].copy()
    odat = x[:-1].copy()
    #idat -= np.mean(idat)
    #odat -= np.mean(odat)
    #pred1 = (np.sum((idat*odat))/(len(odat)-1.)) /(np.sum(x**2)/(float(len(x)-1.)))
    pred = (np.sum((idat*odat))/(len(odat)-1.)) /(np.sum(odat**2)/(float(len(odat)-1.)))
    return pred

def yule_walker_old(x):
    """
    computes the yule-walker-prediction for an 1D-AR(1)-process
    """
    pred = np.mean(x[1:]*x[:-1])/np.mean(x**2)
    return pred

def vred(x,alpha,asProcess = True):
    """
    estimates the quality of a prediciton by computing the reduction of
    the variance
    if asProcess is True, then compute the "prediction", i.e. omit first
    element for variance computation
    """
    pred = x*alpha
    rdiff = x[1:] - pred[:-1]
    if asProcess:
        var0 = np.sum(x[1:]**2)
        varp = np.sum(rdiff**2)
    else:
        var0 = np.sum(x**2)
        varp = np.sum(rdiff**2)

    return varp/var0

def vred2(x,alpha):
    """
    estimates the quality of a prediciton by computing the reduction of
    the variance
    """
    pred = np.dot(alpha,x)
    rdiff = x[:,1:] - pred[:,:-1]
    varp = np.diag(np.cov(rdiff))
    var0 = np.diag(np.cov(x))

    return varp/var0





def expect_dVar(n, sigmaSquare = 1., mu = 0.):
    """
    computes the expected variance of the variance for a gaussian
    distributed random number as a function of the length
    """
    nf = float(n)
    mu_4 = mu**4. + 6.*mu*sigmaSquare + 3.*sigmaSquare**2
    sigma_4 = sigmaSquare**2
    res = 1./nf*(mu_4 - (nf-3.)/(nf-1.)*sigma_4)
    return res



def emp_dVar(length,nSamples):
    """
    numerical experiment: compute random numbers (trials of length "length"),
    compute individual variances, and compute variance of variances
    """
    all_v = []
    for a in range(nSamples):
        r = rnd.randn(length)
        all_v.append(np.var(r))
    return np.var(all_v)


def pc_pm_std(data, ndim):
    """
    This is a helper function.
    It returns the value of +1 * std(x), where x is the ndim-th principal
    component of the data

    Parameters:
    -----------
    data: `array` (*n*-by-*d*)
        the data on which the principal component analysis is performed.
    ndim: `integer`
        the number of the principal axis on which the analysis is performed.
        **NOTE** this is zero-based, i.e. to compute the first principal
        component, ndim=0

    Returns:
    --------
    std_pc: `array` (1-by-*d*)
        the vector that points in the direction of the *ndim*th principal
        axis, and has the length of the standard deviation of the scores
        along this axis.

    """

    u,s,v = svd(data.T, full_matrices = False)
    direction = u[:, ndim : ndim + 1]
    scale = std(dot(direction.T, data.T))
    return scale * direction.T


def predTest(idat, odat, out_of_sample=True, nboot=50, totvar=False, rcond=1e-7):
    """
    .. note::

      computes how well odat can be predicted (in terms of variance reduction)
      using idat, using the bootstrap method

    Some formatting test: :py:func:`mutils.statistics.vred`

    Parameters
    ----------
    idat : array_like
        format: n x d , d-dimensional data in n rows
        used to predict odat
    odat : array_like
        format: n x q, q-dimensional data in n rows
        to be predicted from idat
    out_of_sample : bool
        if True, perform an out-of-sample prediction
    nboot : int
        the number of bootstrap repetitions to be performed for prediction
    totvar : bool
        if `True`, the total relative remaining variance will be computed,
        otherwise the relative remaining variance for each coordinate
        to be predict will be computed

    Returns
    -------
        Return value and format depends on whether or not **totvar** is *True*

        * if **totvar** is *True*:
                returns an array of dimension nboot x 1, which contains
                the relative remaining variance after prediciton
                (in nboot bootstrap repetitions)
        * if **totvar** is *False*:
                returns an array of dimension nboot x q, which contains the
                relative remaining variance after  prediction for each
                coordinate (in nboot bootstrap repetitions)




    """

    _, mapsT, idcs = fitData(idat, odat, nps=1, nrep=nboot, rcond=rcond)
    maps = [x.T for x in mapsT]
    # indices will be "swapped" to "NOT(indices)" in vRedPartial
    indices = idcs if out_of_sample else [otheridx(x, idat.shape[0]) for x in idcs]
    vaxis = None if totvar else 0
    res = vRedPartial(idat, odat, maps, indices, vaxis)
    return vstack(res)



def DFA(data, npoints=None, degree=1, use_median=False):
    """
    computes the detrended fluctuation analysis
    returns the fluctuation F and the corresponding window length L

    :args:
        data (n-by-1 array): the data from which to compute the DFA
        npoints (int): the number of points to evaluate; if omitted the log(n)
            will be used
        degree (int): degree of the polynomial to use for detrending
        use_median (bool): use median instead of mean fluctuation

    :returns:
        F, L: the fluctuation F as function of the window length L

    """
    # max window length: n/4

    #0th: compute integral
    integral = cumsum(data - mean(data))

    #1st: compute different window lengths
    n_samples = npoints if npoints is not None else int(log(len(data)))
    lengths = sort(array(list(set(
            logspace(2,log(len(data)/4.),n_samples,base=exp(1)).astype(int)
             ))))

    #print lengths
    all_flucs = []
    used_lengths = []
    for wlen in lengths:
        # compute the fluctuation of residuals from a linear fit
        # according to Kantz&Schreiber, ddof must be the degree of polynomial,
        # i.e. 1 (or 2, if mean also counts? -> see in book)
        curr_fluc = []
#        rrt = 0
        for startIdx in arange(0,len(integral),wlen):
            pt = integral[startIdx:startIdx+wlen]
            if len(pt) > 3*(degree+1):
                resids = pt - polyval(polyfit(arange(len(pt)),pt,degree),
                                  arange(len(pt)))
#                if abs(wlen - lengths[0]) < -1:
#                    print resids[:20]
#                elif rrt == 0:
#                    print "wlen", wlen, "l0", lengths[0]
#                    rrt += 1
                curr_fluc.append(std(resids, ddof=degree+1))
        if len(curr_fluc) > 0:
            if use_median:
                all_flucs.append(median(curr_fluc))
            else:
                all_flucs.append(mean(curr_fluc))
            used_lengths.append(wlen)
    return array(all_flucs), array(used_lengths)


def visDFA(F, L, nFit=None):
    """
    plots and fits a linear slope to the given DFA results.
    nFit: fit only first n datapoints
    """
    p = polyfit(log10(L[0:nFit]),log10(F[0:nFit]),1)
    fig = figure()
    plot(log10(L),log10(F),'b',label='data')
    plot(log10(L),polyval(p,log10(L)),'r',
         label= ('linear fit\n($\\alpha$ = %1.3f' % p[0]) )
    xlabel('log (window size)')
    ylabel('log (fluctuation)')
    title('DFA')
    legend()
    return fig


def vRedPartial(idat, odat, maps, idcs, vaxis=0):
    """
    computes the variance reduction for each coordinate after prediction,
    given input data idat, output data odat, the maps that map idat to odat,
    and the indices that were used to compute the regression.
    An out-of-sample prediction is performed.
    idat, odat: shape n x d, n: number of measurement, d: number of dimensions
    maps: a list of matrices A where idat*A predicts odat
    """
    rvar = []
    for A, idx in zip(maps,idcs):
        tidx = otheridx(idx,idat.shape[0])
        pred = dot(idat[tidx,:],A)
        rvar.append(var(odat[tidx,:] - pred, axis=vaxis) /
                        var(odat[tidx,:], axis=vaxis))
    return rvar


def otheridx(idx,max_idx):
    """
    returns a sorted, integer array containing all numbers
    in a range from 0 to max_idx-1 that are not in idx
    """
    return sort(list(set(arange(max_idx))-set(idx)))



def fitData(idat, odat, nps=1, nidx = None, nrep = 500, sections = [0,], rcond = 1e-6):
    """
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
       iteration. If omitted, idat.shape[0] is used

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
        nidx = idat.shape[0]

    if any(diff(sections)) < 0:
        raise ValueError, 'sections must be given in increasing order!'

    # 1st: create bootstrap indices
    indices = [(idat.shape[0]*rand(nidx)).astype(int) for x in range(nrep)]

    # 2nd: create section fits (if sections)
    # part A: do fits from one section to the next (within same stride)
    sectMaps = [
                  [
                     lstsq( idat[idcs,sections[sct]::nps],
                            idat[idcs,sections[sct+1]::nps], rcond=rcond)[0].T
                  for idcs in indices ]
               for sct in range(len(sections)-1) ]

    # part B: do fits from last section to first section of next stride
    if len(sections) > 1:
        sectMaps.append( [
                            lstsq( idat[idcs,sections[-1]::nps],
                            odat[idcs,sections[0]::nps], rcond=rcond)[0].T
                  for idcs in indices ]  )

    # 3rd: create stride fits
    strideMaps = [ lstsq(idat[idcs,sections[0]::nps],
        odat[idcs,sections[0]::nps], rcond=rcond)[0].T for idcs in indices ]

    return sectMaps, strideMaps, indices


class RCLS(object):
    """
    A recursive least squares estimator.

    The general structure of the model to be estimated is:
        y = A x + e
        y: output data (m-by-do, m: # of measurements, do: output dimension)
        A: model to be estimated (do-by-di)
        x: input data (m-by-di, di: input dimension)
        e: error (to be minimized)

    Usage:

        r = RCLS(di, do, d=50, lambda_=1)
        # where di, do as above, d is an initialization factor (the lower, the
        # slower (but more robust) the convergence is, and lambda_ is a
        # "forgetting factor" with values between 0 and 1, with:
        #    0: "use only last measurement", 1: "use all measurements"
        Aest = RCLS.digest(x, y) # do this for every input/output pair
        # with x,y  being the current observations, and Aest beeing the updated
        # estimation of A

    """
    def __init__(self, di, do, d=50, lambda_ = 1):
        self.lambda_ = lambda_
        self.d = d
        self.P = d * np.eye(di)
        self.A_est = np.zeros((do, di))


    def digest(self, x, y):
        """
        processes the input-output pair x,y

        @param
            x (array of len di) current input data
            y (array of len do) current output data

        @return Returns the current estimate of A

        Thanks to de.wikipedia.org/wiki/RLS-Algorithmus
        """
        # bring x and y to "correct" shape
        x_ = np.array(x).squeeze()[:, np.newaxis]
        y_ = np.array(y).squeeze()[:, np.newaxis]


        # weighting factor for updates
        gamma = np.dot(x_.T, self.P) / (self.lambda_ +
                                        np.dot(x_.T, np.dot(self.P, x_)))


        # ct: correction term ("prediction error")
        ct = y_ - np.dot(self.A_est, x_)

        # update prediction
        self.A_est = self.A_est + np.dot(ct, gamma)

        # update P
        self.P = 1./self.lambda_ * (self.P - np.dot(self.P, dot(x_, gamma)))
        return self.A_est

def pdf_from_cv(cv, origin, roundoff=1e-9):
    """
    Returns a gaussian pdf from a given covariance matrix
    
    @param cv (n-by-n): covariance matrix
    @param origin (n-by-1): origin
    @param roundoff (float): ratio of largest to smallest eigenvalue 
        s.th. smallest eigenvalue is not considered 0 (degenerate case)
        
    @return Returns a pdf function        
    """
    
    if not np.allclose(cv - cv.T, 0):
        raise ValueError("argument must be a positive definite symmetric "
                "covariance matrix")
        
    dim = cv.shape[0]
    x_ref = np.array(origin)
    if len(x_ref) != dim:
        raise ValueError("x_ref dimension must match covariance dimension")
    
    # this is numerically stable here since cv is symmetric
    # -> eigenvectors are normal
    eigvals, eigvecs = np.linalg.eig(cv) 
    if not all(eigvals > 0):
        raise ValueError("argument must be a positive definite "
                "symmetric covariance matrix")
        
    max_ev = max(eigvals)
    small_ev = find(eigvals < roundoff*max_ev)
    large_ev = find(eigvals >= roundoff*max_ev)
    eigvals_i = eigvals.copy()
    eigvals_i[small_ev] = 0
    eigvals_i[large_ev] = 1./eigvals_i[large_ev]
    
    # compute the pseudo-inverse using eigenvalues
    # (numerically stable here, see above)
    cv_i = np.dot(eigvecs, 
               np.dot(np.diag(eigvals_i), eigvecs.T))
    
    p_det = reduce(lambda x, y: x*y, eigvals[large_ev]) # pseudo-determinant
    
    scale = (1. / np.sqrt((2*np.pi)**dim * p_det))
    projector = np.dot(cv, cv_i)
    def fun(xp, threshold=1e-10, debug=False):
        x = np.dot(projector, xp)
        if np.linalg.norm(x - xp) > threshold:
            if debug:
                print "debug:", svd(projector, 0, 0)
                print "threshold exceeded:", x, xp
                print ("(norm of", x - xp, ":", np.linalg.norm(x - xp), " > ",
                        threshold, ") ")

            return 0
        return scale * np.exp(-0.5 * np.dot(x - x_ref, np.dot(cv_i, x - x_ref)))
    return fun
