"""
This module implements clustering functionality.

:author:
    moritz.maus@hm10.net

:date:
    July 15, 2014

:synopsis:
    This module currently implements a variant of the Rodriguez-Laio
    clustering algorithm, described in
      Alex Rodriguez and Alessandro Laio: 
      *Clustering by fast search and find of density peaks*
      **Science** 344 (6191), 1492-1496, **2014**
    The differences concern the "local density function", which has now a
    parameter-free variant, and the visualization of the rho-d_i plot which can
    be now semi-logarithmic.

"""

import numpy as np
import numpy.linalg as li
import pylab

def _l2_distance(a,b):
    """
    returns |a-b|
    """
    return li.norm(a-b)

def _get_distances(data, distance_func = _l2_distance):
    """
    internal use, computes the distance matrix between data points
    data is in N-by-d format

    norm_func (function(x)): the norm of an element x

    returns N-by-N distance matrix
    """
    islist = False
    if isinstance(data, list):
        npoints = len(data)
        islist = True
    elif isinstance(data, np.ndarray):
        npoints = data.shape[0]
    else:
        raise TypeError('Argument "data" must be list or numpy array!')
    d_ij = np.zeros((npoints, npoints))
    for elem_i in range(npoints):
        for elem_j in range(elem_i, npoints):
            if islist:
                # this is only because I don't trust the data[row] ==
                # data[row,:] convention for arrays
                d_ij[elem_i, elem_j] = distance_func(data[elem_i], data[elem_j])
            else:
                d_ij[elem_i, elem_j] = distance_func(data[elem_i,:],
                        data[elem_j,:])

            d_ij[elem_j, elem_i] = d_ij[elem_i, elem_j]
            # only set distance to 1 if distance smaller than dc

    return d_ij

def _get_density(distances, dc=None):
    """
    gets the "local density" for each point.
    :args:
        distances (N-by-N): the distance matrix
        dc (float or None): the cut-off threshold. If omitted, use 1/d_ij as
           distance measure.
    :returns:
        N-by-1 array with local densities for each point ("rho" in the paper)

    """
    if dc is None:
        # the element-wise exponential
        d_ij = np.exp(-distances)
        # d_ij = 1. / (np.eye(distances.shape[0]) + distances)
        # alternatively, 1./ (...)**.5 works also well
    else:
        d_ij = (distances < dc).astype(int)
    rho = np.sum(d_ij, axis=0)
    return rho



def find_cluster_centers_fast(data, return_distances=True, dc=None):
    """
    similar to find_cluster_centers, with two main differences:
    (1) A distance function cannot be specified - Euklidean distance is
        computed
    (2) This implementation is orders of magnitudes faster
    """
    distances = np.zeros((data.shape[0], data.shape[0]))

    # compute mutual distances - quite fast
    for idx in range(data.shape[0]):
        dsts_raw = data - data[idx, :]
        distances[idx,:] = np.sqrt(np.sum(dsts_raw**2, axis=1))

    # compute local density (same as old version)
    rho = _get_density(distances, dc=dc) 
    max_dst = np.max(distances)

    # compute for every point: minimal distance to point with higher density
    # (for point with highest density: set distance = max_distance)
    d_i = np.zeros(len(rho))
    for row in range(distances.shape[0]):
        cand_idcs = pylab.find(rho > rho[row])
        if len(cand_idcs):
            d_i[row] = min(distances[row, cand_idcs])
        else:
            d_i[row] = max_dst
        
    if return_distances:
        return rho, d_i, distances
    return rho, d_i

def find_cluster_centers(data, return_distances = True, dc=None,
        distance_function=_l2_distance, distances=None):
    """
    Runs the clustering algorithm which identifies the cluster centers.

    *example*: run 
        >rho, d_i, distances = find_cluster_centers(data)
        >figure()
        >semilogy(rho, d_i, '+')
    and find "(upper right) outliers", for example:
        >centers = find( (rho > THRESHOLD_RHO) * (d_i > THRESHOLD_DI))

    :args:
        data (N-by-d array): N samples of data of length d
        return_distances (bool): if True, return additionally the distance
            matrix
        dc (float or None): dc parameter; if omitted use sum exp(-d_ij) as
            local density
        norm_func (function data -> R^+_0): a function that returns the norm of
            the difference between two data points
        distances(N-by-N array): (optional) the pre-computed distances

    :returns:
        rho, d_i *or* rho, d_i, distances:
            the rho (local density for each point), d_i (distance of each point
                to a point with higher density) parameters of the paper, and
                (optionally) the distance matrix.
    """
    if distances is None:
        distances = _get_distances(data, distance_function) 
    rho = _get_density(distances, dc=dc) 
    
    max_dst = max(distances.flat)
    
    if isinstance(data, list):
        npoints = len(data)
    elif isinstance(data, np.ndarray):
        npoints=data.shape[0]

    d_i = np.inf * np.ones(npoints)
    for idx_i in range(npoints):
        # go through all j
        # for all j with rho_j > rho_i:
        #   compute distance i-j
        #   if distance i-j < min_dst(i):
        #      min_dst(i) = distance i-j
        for idx_j in range(npoints):
            if idx_i == idx_j:
                continue
    
            if rho[idx_j] > rho[idx_i]:
                if d_i[idx_i] > distances[idx_i, idx_j]:
                    d_i[idx_i] = distances[idx_i, idx_j]
        # the element with the highest density -> set distance to set diameter
        if d_i[idx_i] == np.inf:
            d_i[idx_i] = max_dst

    if return_distances:
        return rho, d_i, distances
    return rho, d_i


def join_near(rho, d_i, max_d, centers, distances,
        rho_threshold = 0.05, d_i_threshold=0.05):
    """
    for a given list of cluster centers (as indices of the data), points will
        be removed that are close to another center with same rho, similar d_i,

    :args:
        rho (n-by-1 array or list): the densities for each data point
        d_i (n-by-1 array or list): distances to points with higher density
        max_d (float): distance for which two centers will be considered
           "close"
        centers (k-by-1 array or list): indices of cluster center points
        distances (n-by-n array): the mutual distances
        rho_threshold (float): the measure if rho[j] is close to rho[k], if
           (rho[j] - rho[k])/(rho[j] + rho[k])  < 2*rho_threshold 
        d_i_threshold (float): same as rho_threshold, only for d_i

    :returns:
        new_centers (m-by-1 array): the remaining indices of cluster centers

    """
# for every center: 
# check if another center further in the list is "close" on decision plot
# if yes, check if distance is closer than max_d
# if yes, do *not* include cluster center in the data
    new_centers = []
    for cidx in range(len(centers)):
        use_point = True
        RHO_THRESHOLD = rho[centers[cidx]] * rho_threshold
        D_I_THRESHOLD = d_i[centers[cidx]] * d_i_threshold
        for ocidx  in range(cidx+1, len(centers)):
            # introduce abbreviations
            c1 = centers[cidx]
            c2 = centers[ocidx]
            if (rho[c1] - rho[c2])/(rho[c1] + rho[c2]) < 2*rho_threshold:
                if (d_i[c1] -d_i[c2])/(d_i[c1] + d_i[c2]) < 2*d_i_threshold:
                    if distances[c1, c2] < max_d:
                        #print "skipping idx", centers[cidx], "because", 
                        #print centers[ocidx], "has distance", 
                        #print distances[centers[cidx], centers[ocidx]] 
                        use_point = False
                        break
        if use_point:
            new_centers.append(centers[cidx])

    return np.array(new_centers)


    


def assign_points(rho, centers, distances, noise_centers = []):
    """
    assigns points to the cluster centers. The rule is:

       'After the cluster centers have been found, each remaining point is
       assigned to the same cluster as its nearest neighbor of higher density.'

    :args:
        rho (N-by-1 array): the local density parameter obtained before
        centres (array or list of k elements): each element gives the index of
            a presumed center in the data.
        distances (N-by-N array): the distance matrix
        noise_centers (array or list of k' elements): centers of noise clusters
            (i.e. putative centers) with low density (rho) but medium to high
            distance (d_i).
            assignments to noise centers will all have the value "0", which is
            not used if no noise_centers are given.
    
    :returns:
        assignment (N-by-1 array (int)): a number that labels the cluster for
        each point
    """
    
    all_centers = np.hstack([centers, noise_centers])
    n_centers = len(centers)
    assignment = np.zeros_like(rho, dtype=int)
    
    # assign numbers to each center
    for nr, c_idx in enumerate(all_centers):
        assignment[c_idx] = nr + 1
 
    # create a sorted N-by-2 array which has (local_density, index) in each row
    # and is ordered by decreasing local_density.
    
    ordered = sorted(zip(rho, range(len(rho))),
            lambda x, y: 1 if x[0] < y[0] else -1)
    
    idcs_sorted = np.array([elem[1] for elem in ordered])
    
    for nr, (rho_i, idx_i) in enumerate(ordered):
        # skip identified cluster centers
        if assignment[idx_i]:
            continue
        # compute distances to points with higher densities:
        idcs_higher = idcs_sorted[:nr]
        if any(rho[idcs_higher] < rho[idx_i]):
            # if there is no error in the implementation, this cannot happen
            raise ValueError("This must not happen!")
        idx_closest = np.argmin(distances[idx_i, idcs_higher])
        assignment[idx_i] = assignment[idcs_higher[idx_closest]]
    # assign "0" to noise clusters if present
    assignment[assignment > n_centers] = 0
    return assignment


def split_in_clusters(data, assignment):
    """
    splits the data into clusters

    :args:
        data (N-by-d array or list of N elements): the data points
        assignment (N-by-1 array): the assignments

    :returns:
        [data, ...] a list containing the data points, each element containing
            only points belonging to one cluster.

    """

    splitted = []
    if isinstance(data, list):
        #ncluster = len(set(assignment))
        splitted_d = {assig: [] for assig in assignment}# elem in range(ncluster)]
        for idx, assig in enumerate(assignment):
            splitted_d[assig].append(data[idx])
        splitted = [val for key, val in splitted_d.iteritems()]
    elif isinstance(data, np.ndarray):
        for cluster in set(assignment):
            splitted.append(data[assignment == cluster, :])
    else:
        raise TypeError("data must be of type list or numpy array!")

    return splitted


