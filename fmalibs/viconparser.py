#!/usr/bin/python

#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Shai Revzen, U Penn, 2010

"""
viconparser provides tools that interface the output of a viconreader with
numpy. The main class is ViconParser, instances of which can parse an 
Info packet (return value from ViconReader.connect or .dcr file), and use this
information to parse data emitted from the Vicon system (sequence items 
returned from ViconReader.stream or .dat file). 

For conveniece, the VinconParser has a parseStream method that can also 
construct a sliding window from the stream data.
"""

import re
from copy import copy
from numpy import asarray,nan,fromfile,frombuffer,nonzero,float64,newaxis,all
from numpy.linalg import norm

MRK_XYZ = re.compile('(\w+):(\w+) <P-[XYZ]>')
OCC = re.compile('(\w+):(\w+) <P-O>')
FPS = re.compile('Time (\d+) fps <F>')

def find(ar):
    "find indices where ar is python true"
    res, = nonzero(asarray(ar,bool).flat)
    return res

def iwindow( N, p, mapping=copy):
  """Iterator returning a list consisting of a sliding window 
  of N consecutive items from iterator p. 
  By default, a copy of each sliding window is returned. 
  Setting mapping to some other callable that expects a list
  allows other functions of the sliding window to be computed 
  without an unnecessary copy operation.
  
  >>> list(VP.iwindow(2,xrange(5),mutable=False))
  [[0, 1], [1, 2], [2, 3], [3, 4]]  
  """
  W = []
  for item in p:
    W.append(item)
    if len(W)==N:
      if mutable:
        yield W
      else:
        yield copy(W)
      W.pop(0)
        
class ViconParser(object):
  """ViconParser parses the output of ViconReader-s into numpy data
  
  Typical usage reading from a file:
  >>> vp = ViconParse()
  >>> vp.load("run1")
  >>> print abs( vp.x + 1j* vp.y )
  
  Typical use for a live stream:
  >>> vp = ViconParse()
  >>> vr = ViconReader()
  >>> vp.useInfo( vr.connect() )
  >>> for now in vp.parseStream( vr.stream(), 10 ):
  ...   print "%6.2f %g,%g" % (now, mean(vp.x,0), mean(vp.y,0) )
  
  The ViconParse creates the following self.attributes:
    fps -- the fps reating of the mocap
    t -- N -- timestamps (seconds)
    x,y,z -- N x M -- x, y and z of M markers over N times
    occ -- N x M -- boolean indicating occlusion of marker M at sample N
    
  NOTE: occluded markers have their coordinates set to nan
  """
  def __init__( self, src = None ):
    self.plan = None
  
  def useInfo( self, dcr ):
    """parse the descriptors from info packet into an indexing scheme"""
    # Number of columns
    self.cols = len(dcr)
    # Find time data
    fm = [ FPS.match(x) for x in dcr ]
    i_t = find(fm)
    # Extract fps
    self.fps = int(fm[i_t].group(1))
    #
    # Find xyz data
    i_xyz = find([ MRK_XYZ.match(x) for x in dcr ] )
    #
    # Find occlusion data
    om = [ OCC.match(x) for x in dcr ]
    i_o = find(om)
    #
    self.plan = dict(
      t = i_t,
      xyz = i_xyz,
      occ = i_o
      )

  def parseStream( self, idat, window=1, mutable=False ):
    """Parse data from a stream (python generator). 
    Window is the length of a sliding window to be taken from the data, e.g.
    setting window=10 gives values from the last 10 frames.
    vp.parseStream() updates the values in vp IN PLACE and yields timestamp.
    mutable may be set to True to allow vp to use idat values in place (no copy)
    
    WARNING: untested
    """
    # If a sliding window was requested --> construct it
    if window>1:
      assert window == int(window)
      idat = iwindow( window, idat, lambda x : "".join(x) )
      # Since we own the buffer created by join, it's safe to mutate
      mut = True
    else: # else --> using idat returns values
      mut = mutable
    # loop over input and use sliding window data for updates
    for dat in idat:
      self.parseData( dat, mutable=mut )
      yield self.t[-1]
      
  def parseData( self, dat, mutable=False ):
    """Parse a data blob into useful data
       dat -- data block as array of floats or some buffer with float64-s
       mutable -- if true, parsing will modify input array in place
    """
    # If a string --> parse into floating point numbers
    if type(dat)==str:
      d = frombuffer( dat, float64 )
      d.shape = (d.size / self.cols, self.cols)
    else: # else --> must be something that becomes an array
      d = asarray(dat,float64)
      # If rank 1 array --> reshape it
      if d.ndim != 2:
        # If reference to non-mutable input --> copy it
        if (d is dat) and (not mutable):
          d = d.copy()
        # Reshape
        d.shape = (d.size / self.cols, self.cols)
    # Create all columns
    for key,val in self.plan.iteritems():
      col = dat[:,val]
      if not mutable:
        col = col.copy()
      setattr( self, key, col )
    # Observation columns specify nans for x,y,z
    self.occ = asarray(self.occ,bool)
    self.xyz.shape = (self.xyz.shape[0],self.xyz.shape[1]/3,3)
    self.xyz[self.occ] = nan
    self.x = self.xyz[...,0]
    self.y = self.xyz[...,1]
    self.z = self.xyz[...,2]

    
  def load( self, filename ):
    """Load ViconReader data from filename.dcr and filename.dat"""
    # Load columns descriptors
    f = open(filename+".dcr",'r')
    dcr = f.readlines()
    f.close()
    # Load data
    f = open(filename+".dat",'rb')
    dat = fromfile( f, float64 )
    dat.shape = ( dat.size / len(dcr), len(dcr) )
    self.useInfo( dcr )
    self.parseData( dat, mutable=True )
  
  def clipFor(self, cols):
    """clip to a region where cols are no occluded"""
    ok = find(all(~self.occ[:,cols],axis=1))
    if not ok.size:
      raise IndexError("All entries are occluded")
    slc = slice(ok[0],ok[-1]+1)
    return self.clip(slc)
    
  def clip( self, slc ):
    self.xyz = self.xyz[slc,...]
    self.occ = self.occ[slc,...]
    self.t = self.t[slc,...]
    self.x = self.xyz[...,0]
    self.y = self.xyz[...,1]
    self.z = self.xyz[...,2]
    return self
