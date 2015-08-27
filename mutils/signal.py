# -*- coding : utf8 -*-
"""

.. module:: mutils.signal
    :synopsis: This module contains some convenient signal processing functions

.. moduleauthor:: Moritz Maus <h.maus@imperial.ac.uk>
"""

import scipy.signal as _sig
import numpy as _np

def filtfilt2(b, a, sig):
    """
    extends the signal in both directions by an anti-symmetric copy of the
    original signal, and then applies filtfilt. This improves signal quality at
    the edges (assuming the signal is roughly smooth at the edges)

    :param b: B of filter (e.g. from scipy.signal.butter)
    :param a: A of filter (e.g. from scipy.signal.butter)
    :param sig: Signal to be filtered (array)

    :returns: the filtered signal
    """
    sig_ = sig.flatten()
    x0 = _np.hstack([2 * sig_[0] - sig_[1:][::-1], sig_, 2.0 * sig_[-1] -
        sig_[-2::-1]])
    return _sig.filtfilt(b, a, x0)[len(sig_) - 1:2*len(sig_)-1]


def interp_2d(t_out, t_data, data):
    """
    Returns the data, interpolated at t_out.
    Data is assumed to be in [dim, time] format

    :param t_out: output time samples
    :type t_out: 1-by-n array
    :param t_data: data time samples
    :type t_data: 1-by-m array
    :param data: input data to be interpolated
    :type data: d-by-m array

    :returns: The interpolated data
    :rtype: d-by-n array

    """
    return _np.vstack([_np.interp(t_out, t_data, data[dim, :]) for dim in
        range(data.shape[0])])

def mat_pr(data):
    """
    Returns the "derivative" of data in a matrix along the last axis.

    e.g., if data are in the format [dim, time], then the derivative w.r.t time
    is computed.

    :param data: array of at least 2 dimensions
    :type data: numpy array
    
    :returns: the two-point derivative of data w.r.t last dimension (like
        numpy gradient does)
    :rtype: numpy array

    """
    res = _np.zeros_like(data)
    d1 = _np.diff(data, axis=-1)
    res[..., :-1] = d1
    res[..., -1] = d1[..., -1]
    res[..., 1:-1] = 0.5 * (d1[..., 1:] + d1[..., :-1])
    return res


