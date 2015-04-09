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

import struct
import socket 
import sys

"""
viconreader is a simple library for connecting to the real-time feed emitted
by a vicon motion capture system. 

It provides two functions:

(1) As a library, it provides the ViconReader class which gives an interface for
connect()-ing to and reading data from a vicon system. ViconReader gives its
user the list of variables emitted by the vicon, and a python generator object
which yields consecutive samples from the motion capture system. stop() and
close() methods allow the user to stop streaming data and to close a TCP
connection to the vicon.

(2) As a commandline tool, viconreader.py will connect to the DEFAULT_HOST, 
currently configured for the GRASP lab vicon (*NOT* high-bay), and save two 
files: a .dcr file containing the variable names, one per line, and a .dat file
containing samples as a continuous stream of 64 bit floats. Capture is stopped 
with a ctrl-c (or ctrl-break, if you use such systems).
"""

DEFAULT_HOST = "10.66.68.1"  
DEFAULT_PORT = 800
  
class ViconReader(object):
  """ViconReader instances provide access to the real-time data stream 
  made available by vicon motion tracking systems.
  
  Instances of ViconReader can send Query, Start and Stop messages and parse 
  responses into lists of attributes and blocks of doubles (encoded as strings).
  
  ViconReader is used from the commandline to log tracking data to a file.
  
  Typical use is:
  >>> V = ViconReader()
  >>> names = V.connect() # names gets list of names from Query Response
  >>> # V.stream() is a generator returning packet payload each .next()
  >>> for pkt,_ in zip(V.stream(),xrange(100)): 
  >>>   dat = struct.unpack("%dd" % (len(pkt)/8),pkt)
  >>>   print
  >>>   for nm,val in zip(name,dat):
  >>>     print nm,val
  >>> V.stop()
  """
  QUERY = (1,0)
  INFO = (1,1)
  START = (3,0)
  STOP = (4,0)
  DATA = (2,1)
  def __init__(self):
    self.sock = None
    self.push = ''

  def _get( self, fmt ):
    "Read data from socket based on format string, and parse it accordingly"
    N = struct.calcsize(fmt)
    # Start reading from push-back buffer
    buf = self.push[:min(len(self.push),N)]
    self.push = self.push[:len(buf)]
    while len(buf)<N:
      buf += self.sock.recv(N-len(buf))
    return struct.unpack(fmt,buf)

  def _parseInfo( self ):
    "Parse an INFO packet, starting with byte after the header"
    N = self._get("1L")[0]
    lst = []
    for _ in xrange(N):
      L = self._get("1L")[0]
      lst.append(self._get("%ds" % L)[0])
    return lst
  
  def _parseData( self ):
    "Parse a DATA packet, starting with byte after the header"
    N = self._get("1L")[0]
    return self._get("%ds" % (N*8))[0]
  
  def _parse( self ):
    "Parse an incoming packet"    
    hdr = self._get("2L")
    if hdr==self.__class__.DATA:
      return (hdr, self._parseData())
    elif hdr==self.__class__.INFO:
      return (hdr, self._parseInfo())
    # Failed -- need to resync
    self.push = struct.pack("2L",*hdr)
    self._resync()
  
  def _resync(self):
    raise ValueError,"Lost synchronization on socket"

  def _cmd( self, hdr ):
    "Command encoded as a 2-tuple header"
    self.sock.send(struct.pack( "2L", *hdr ))
  
  def connect( self, host = DEFAULT_HOST, port = DEFAULT_PORT ):
    # Connect the socket
    self.sock = socket.socket( socket.AF_INET, socket.SOCK_STREAM )
    self.sock.connect((host,port))
    # Send a query
    self._cmd( self.__class__.QUERY )
    # Loop until a query response is received
    while True:
      hdr,names = self._parse()
      if hdr == self.__class__.INFO:
        return names
        
  
  def stream( self ):
    "Generator producing a stream of data packet payloads"
    self._cmd( self.__class__.START )
    while True:
      hdr,data = self._parse()
      if hdr == self.__class__.DATA:
        yield data
  
  def stop( self ):
    "Tell Vicon to stop streaming"
    self._cmd( self.__class__.STOP )
  
  def close( self ):
    "Close connection to Vicon"
    self.sock.close()
    self.sock = None

if __name__=="__main__":
  # Simple commandline viconreader tool
  if len(sys.argv) != 2:
    sys.stderr.write("""
Usage: %s filename
  Creates filename.dcr with the field names and filename.dat with the raw
  data stored as doubles.
  """ % sys.argv[0])
    sys.exit(5)
  # Filename user requested
  fn = sys.argv[1]
  V = ViconReader()
  names = V.connect()
  # Write descriptor
  dcr = open(fn+".dcr", "w")
  dcr.write("\n".join(names)+"\n")
  dcr.close()
  # Write data
  L = 0
  N = 0
  #P dat = open(fn+".dat", "w")
  dat = []  
  try: # for catching ctrl-c termination
    for pkt in V.stream():
      if not L: # First packet sets the length we'll expect 
        L = len(pkt)
      elif L != len(pkt):
        print "\nUnexpected packet of length %d instead of %d" % (len(pkt),L)
        continue
      #P dat.write(pkt)
      dat.append(pkt)
      N += 1
      if (N%50)==0:
        print "\r%5d" % N,
    # ---
  except KeyboardInterrupt, ex:
    #P dat.close()
    f = open(fn+".dat", "w")
    f.write("".join(dat))
    f.close()
    V.stop()
    print "\nStop."
    


