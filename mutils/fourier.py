# -*- coding : utf8 -*-
"""

.. module:: mutils.fourier
    :synopsis: This module contains some Fourier-transform based functions:
        - a zero-lag integrator and differentiator
        - a spectrum attenuation notch filter
        - a Fourier model builder and computer

.. moduleauthor:: Moritz Maus <mmaus@sport.tu-darmstadt.de>


"""


from pylab import (vstack, hstack, zeros, arange, array, randn, dot, sum, eig,
        logspace,sqrt, exp, log, find, sin, diag, randint, svd, ones, inv, mod,
        mean, cumsum, pi, zeros_like, linspace, interp, polyval, ceil, eye,
        isreal, isnan, linspace, concatenate, convolve, polyfit, polyval,
        roots, median) 

import scipy.fftpack as fft

def f_nfilt(sig, fs, fstop, wstop):
    """    
    fourier-spectrum-manipulation based "notch" filter. The spectrum is 
    "manually" attenuated with a gaussian at the selected frequency.
    
    :args:
        sig (1-by-n array): original signal
        fs (int or float): sampling frequency [Hz]
        fstop (float): notch frequency [Hz]
        wstop (float): width of the gaussian [Hz]
        
    :returns:
        sig_p (1-by-n array): the filtered signal
    """
    freq_vec = arange(len(sig)) * float(fs) / float(len(sig))
    att = ones(len(sig))
    f0 = fstop
    fE = fs - fstop
    att -= exp( -((freq_vec - f0)/wstop)**2) + exp( -((freq_vec - fE)/wstop)**2)
    spec = fft.fft(sig)
    return fft.ifft(spec*att).real



def int_f(a, fs=1.):
    """
    A fourier-based integrator.

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

    if False:
    # version with "mirrored" code
        xp = hstack([a, a[::-1]])
        int_fluc = int_f0(xp, float(fs))[:len(a)]
        baseline = mean(a) * arange(len(a)) / float(fs)
        return int_fluc + baseline - int_fluc[0]
    
    # old version
    baseline = mean(a) * arange(len(a)) / float(fs)
    int_fluc = int_f0(a, float(fs))
    return int_fluc + baseline - int_fluc[0]

    # old code - remove eventually (comment on 02/2014)
    # periodify
    if False:
        baseline = linspace(a[0], a[-1], len(a))
        a0 = a - baseline
        m = a0[-1] - a0[-2]
        b2 = linspace(0, -.5 * m, len(a))
        baseline -= b2
        a0 += b2
        a2 = hstack([a0, -1. * a0[1:][::-1]]) # "smooth" periodic signal  

        dbase = baseline[1] - baseline[0]
        t_vec = arange(len(a)) / float(fs)
        baseint = baseline[0] * t_vec + .5 * dbase * t_vec ** 2
        
        # define frequencies
        T = len(a2) / float(fs)
        freqs = 1. / T * arange(len(a2))
        freqs[len(freqs) // 2 + 1 :] -= float(fs)

        spec = fft.fft(a2)
        spec_i = zeros_like(spec, dtype=complex)
        spec_i[1:] = spec[1:] / (2j * pi* freqs[1:])
        res_int = fft.ifft(spec_i).real[:len(a0)] + baseint
        return res_int - res_int[0]



def diff_f0(x, fs=1.):
    """
    returns the 'basic' fourier derivative of a signal
    """
    # define frequencies
    T = len(x) / float(fs)
    freqs = 1. / T * arange(len(x))
    freqs[len(freqs) // 2 + 1:] -= float(fs)

    spec = fft.fft(x)
    spec_i = spec * (2j * pi* freqs)
    # if an even number of frames is recorded, an alternating signal (the 
    # highest single frequency) cannot be 'sampled' -> set to 0.
    # for example: consider x = [0 1], making this a periodic function will
    # give [0 1 0 1 0 1], and at each point its derivative is 0 because of
    # symmetry arguments
    if mod(len(x), 2) == 0:
        spec_i[len(x) // 2] = 0.
    sig_d = fft.ifft(spec_i) 
    return sig_d.real



def int_f0(x, fs=1.):
    """
    returns the 'basic' fourier integration of a signal
    
    """
    # define frequencies
    T = len(x) / float(fs)
    freqs = 1. / T * arange(len(x))
    freqs[len(freqs) // 2 + 1:] -= float(fs)

    spec = fft.fft(x)
    spec_i = zeros_like(spec, dtype=complex)
    # exclude frequency 0 - it cannot be integrated
    spec_i[1:] = spec[1:] / (2j * pi* freqs[1:])
    if mod(len(x), 2) == 0:
        spec_i[len(x) // 2] = 0.
    sig_d = fft.ifft(spec_i) 
    return sig_d.real


def diff_f(x, fs=1.):
    """
    A Fourier-based differentiator

    *changed 02 / 2014* The signal is "periodified", that is it is reflected
    and concatenated with the original signal. The resulting signal has no
    discontinuities and no trend. (Code added but disabled; not functional;
    comment below is still up to date)

    *HINT* A Fourier series implicitly assumes that the data is a window of a
    periodicit signal with the window length as period. This implies that if
    the first and the last point of the data are not 'close', a jump is
    assumed, which will be reflected in a 'jump' of the derivative at the
    beginning and the end of the signal.  To circumvent this, the baseline is
    separated and derived separately.

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

    #xp = hstack([x, x[::-1]])
    #return diff_f0(xp)[:len(x)]

    if True:
        baseline0 = linspace(x[0], x[-1], len(x), endpoint=True)
        x0 = x - baseline0
        baseline1 = 0 * baseline0
        if mod(len(x), 2) == 0:
            baseline1 = -1.* (2. * fft.fft(x0)[len(x0) / 2].real /
                    float(len(x0)) * arange(len(x0)))
        x0 = x0 - baseline1
# x0 can now be integrated and differentiated using diff_f0, int_f0
        baseline = baseline0 + baseline1
        return diff_f0(x0, float(fs)) + (baseline[1] - baseline[0]) * fs

def interp_f(arr_orig, newlen, detrend=False):
    """
    Uses a Fourier based interpolation to create an array with newlen points based on the data of arr.
    
    Notes:
    * The endpoints of arr_orig and the interpolated data do *NOT* refer to the same point in time! Rather,
      it is assumed that the total measurement duration is unchanged (e.g.: if you measure 10 seconds,
      the last datapoints will be at t=9.9s and t=9.99s for 10Hz and 100Hz sampling rate).
      Practically, you can compute the corresponding time vectors using linspace:
      t_orig = linspace(0, T, len(arr_orig), endpoint=False)
      t_orig = linspace(0, T, newlen, endpoint=False)
    * Implicitly, continuity at the egde is assumed. If this is not the case, oscillations will occur. These
      can be diminished a little bit by setting "detrend" to true.

    :args:
        arr_orig (array(1d)): the original data
        newlen (int): number of points of the new array
        detrend (bool): whether or not to manually detrend the data

    :returns:
        intp (1d array): an array with newlen elements which are the interpolated data of arr
    
    """
    arr = arr_orig.squeeze().copy()
    if detrend and len(arr) < 3:
        raise ValueError('Cannot detrend arrays shorter than 3!')
    if detrend:
        d0 = arr[1] - arr[0]
        dE = arr[-1] - arr[-2]
        dm = .5 * (d0 + dE)
        endval = arr[-1] + dm
        trend = linspace(arr[0], endval, len(arr), endpoint=False)
        trend_i = linspace(arr[0], endval, newlen, endpoint=False)
        arr -= trend
        
    spec = fft.fft(arr)
    f0 = spec[0]
    n2 = (len(arr)) // 2
    n2n = (newlen) // 2
    snew = zeros(newlen, dtype=complex)
    snew[0] = f0
    nm = min(n2, n2n)

    snew[1:nm+1] = spec[1:nm+1]
    snew[-nm:] = spec[-nm:]
            
    # adapt scaling
    snew *= float(newlen) / float(len(arr))
    if detrend:
        return fft.ifft(snew).real + trend_i
    return fft.ifft(snew).real

def f_mdl(mdl, phi, maxord):
    """
    given a periodic function "mdl" consisting of n data points (ranging from  [0,2pi)),
    a fourier model of order maxord is computed for all phases in phi

    :args:
        mdl (1-by-k array): datapoints of the reference model, ranging from
           [0, 2pi). Length in datapoints is arbitrary.
        phi (1-by-n array): phases at which to evaluate the model
        maxord (int): order of the Fourier model to compute

    :returns:
       mdl_val (1-by-n array): value of the Fourier model obtained from mdl for
           the given phases phi
    
    """
    spec_fy = fft.fft(mdl)
    as_, ac_ = spec_fy.imag, spec_fy.real
    sigout = zeros(len(phi))
    for order in range(maxord):
        sigout -= sin(order * phi) * as_[order]
        sigout += cos(order * phi) * ac_[order]    
        sigout += cos(order * phi) * ac_[-order]
        sigout += sin(order * phi) * as_[-order]
    sigout /= len(mdl)
    return sigout
    

