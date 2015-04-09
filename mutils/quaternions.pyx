"""

This is a cython module. Use cython to import:

import pyximport
pyximport.install()

import mutils.cy_quaternions as quart

"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, cos, acos # much, much faster than np.sqrt
from scipy.linalg import norm

def q_mult(np.ndarray[np.float_t, ndim=1] p, np.ndarray[np.float_t, ndim=1] q):
    """
    quaternion-like multiplication of 4-by-1 vectors q1, q2 (treated as quaternions)
    
    :args:
        p, q: 1D vectors of 4 elements, treated as quaternions
    :returns:
        p*q (4-dim vector), quaternion multiplication of p*q
    
    """
    res = np.zeros(4, dtype=np.float64)
    
    res[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    res[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    res[2] = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    res[3] = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return res

def q_conj(np.ndarray[np.float_t, ndim=1] q):
    """
    the conjugated quaternion
    """
    # ugly code, but fast
    res = np.zeros(4, dtype=np.float64)
    res[0] = q[0]
    res[1] = -q[1]
    res[2] = -q[2]
    res[3] = -q[3]
    return res
    
def q_inv(np.ndarray[np.float_t, ndim=1] q):
    """
    the inverse of a quaternion q
    """
    # ugly code, but fast
    res = np.zeros(4, dtype=np.float64)
    nrm2 = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    res[0] = q[0]/nrm2
    res[1] = -q[1]/nrm2
    res[2] = -q[2]/nrm2
    res[3] = -q[3]/nrm2
    return res


def RotationBetweenVectors(start, dest):
    """
    returns the quaternion required to rotate start into dest

    *NOTE* this is pure python code -> slow!

    """
    
    start = np.array(start) / norm(np.array(start))
    dest = np.array(dest) / norm(np.array(dest))
 
    cosTheta = np.dot(start, dest)
 
    if (abs(cosTheta) > 1 - 0.0001):
        # special case when vectors in opposite directions:
        # there is no "ideal" rotation axis
        # So guess one; any will do as long as it's perpendicular to start
        rotationAxis = np.cross([0, 0, 1.], start);
        if (norm(rotationAxis) < 0.01 ):  # bad luck, they were parallel, try again!
            rotationAxis = np.cross([1., 0, 0], start)
        if cosTheta < 0: # 180 deg rotation:
            q_rot = np.hstack([0, rotationAxis])
            return q_rot
        print "oops - check returned quaternion... and delete warning"
    else:
        rotationAxis = np.cross(start, dest)
 
    rotationAxis = rotationAxis / norm(rotationAxis)

    # this could be sped up by using trigonometry theorems ...
    # simple version for development
    Theta = acos(cosTheta)
    cTheta2 = cos(Theta / 2.)
    sTheta2 = sin(Theta / 2.)
    
    
    
#    //axis is a unit vector
    q_rot = np.hstack([cTheta2,] + [elem for elem in sTheta2*rotationAxis])
    return q_rot

def makeQ(vec):
    """
    returns a quarternion representing vec in w,x,y,z form (i.e. [0, vec])
    """
    return np.hstack([0., vec])


def rot_by(vec, q):
    """
    applies the rotation of quaternion q (must be normalized) the 3D vector vec
    
    :args:
        q (4-by-1 array): [theta, x, y, z] - form rotation quaternion
        vec (3-by-1 array): the vector to be rotated
    
    :returns:
        vec (3-by-1 array): the rotated vector
    
    """
    
    res = reduce(q_mult, [q, makeQ(vec), q_inv(q)])
    
    return res[1:] #.array[1:]

def QfromR(R):
    """
    
    returns a quaternion representation of the rotation matrix R
    
    algorithm from wikipedia 
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    
    :args:
        R (3x3 array): the rotation matrix
        
    :returns:
        q (4-by-1 array): a corresponding quaternion in [w, x,y,z] order
    
    """
        
    t = np.trace(R)
    r_ = sqrt(1. + t)
    w = 0.5 * r_
    x = np.copysign(0.5 * sqrt(1. + R[0,0] - R[1,1] - R[2,2]), R[2,1] - R[1,2])
    y = np.copysign(0.5 * sqrt(1. - R[0,0] + R[1,1] - R[2,2]), R[0,2] - R[2,0])
    z = np.copysign(0.5 * sqrt(1. - R[0,0] - R[1,1] + R[2,2]), R[1,0] - R[0,1])
    
    return np.array([w, x, y, z], dtype=np.float64)

def getRotMat(q):
    """
    returns the 3x3 rotation matrix from a quaternio q in [w, x,y,z] format
    """
    w, x, y, z = q

    rotmat = np.array([
    [1. - 2.*y**2 - 2*z**2, 2.*x*y - 2*w*z, 2.*x*z + 2*w*y],
    [2.*x*y + 2*w*z, 1. -2.*x**2 - 2*z**2, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1. - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return rotmat



def _getFrame(np.ndarray[np.float64_t, ndim=1] m1,
        np.ndarray[np.float64_t, ndim=1] m2,
        np.ndarray[np.float64_t, ndim=1] m3, ):
    """
    undocumented - unused 
    """

    cdef np.ndarray[np.float64_t, ndim=1] d1
    cdef np.ndarray[np.float64_t, ndim=1] d2
    cdef np.ndarray[np.float64_t, ndim=1] xax
    cdef np.ndarray[np.float64_t, ndim=1] yax
    cdef float ynorm
    d1 = (m2 - m1)/np.linalg.norm(m2 - m1)
    d2 = (m3 - m1)/np.linalg.norm(m3 - m1)
    xax = d1
    yax = d2 - d1 * np.dot(d1, d2)
    ynorm = np.linalg.norm(yax)
    if np.linalg.norm(yax) < 1e-3: # distance below 1 mm -> measurement error assumed
        raise ValueError("oops - norm too small")
    yax = yax / ynorm
    zax = np.cross(xax, yax)
    return np.vstack([xax, yax, zax]).T


