"""
qualisys.py

OVERVIEW

  The Qualisys module provide support for a subset of the Version 1.9
  protocol for real-time interaction with QTM, the Qualisys motion 
  system base-station software.

  This code has been tested with QTM version 2.6 (build 682)

Interface

  The top-level class most users of this library will use is the QTM
  class. QTM provides the communication interface, which generates and
  consumes instances of the class hierarchy under AbstractQTM_Data.
  
  The QTM data classes all follow the same overall structural scheme:
  they use the numpy array classes generated via numpy.frombuffer to 
  provide zero-copy access to the data pulled from the sockets.
  
  Subclasses represent specific packet types, and specific component 
  types of the Data (type 3) packets. 

  Parsing is accomplished using several generators which abstract away
  the differences between UDP and TCP sockets. The code makes heavy use 
  of socket timeouts. All the socket parsing iterators yield None when
  sockets time-out without returning a new packet.
  
  For an overview of the QTM Data classes, see class AbstractQTM_Data.
  For an overview of the iterator based parsing interface, see the
  packetIter function.

LICENSE GPL 3.0 (c) Shai Revzen, Royal Vet College UK and U. Pennsylvania
"""
from socket import (
  socket, AF_INET, SOCK_STREAM, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST,
  timeout as SocketTimeout,
  error as SocketError
)

from xml.etree import ElementTree 

from numpy import (
  asarray, frombuffer, newaxis, 
  float32, float64, uint64, uint32, uint32, uint16, uint8
)

def unpack( dat, fmt ):
  """Similar to struct.unpack, but unpacks into numpy objects
  fmt is a list of one of the following types of elements:
   - numpy dtype -- parses a one-element array of that type
   - tuple (n, numpy dtype) -- parses an n-element array of that type
   - as above, but dtype is None -- skips bytes
  
  returns list of arrays, len, dat[len:]
  
  if dat is None, first and last elements will be None
  """
  res = []
  pos = 0
  try:
    for f in fmt:
      if type(f) is tuple:
        n,f = f
      else:
        n = 1
      if f is None:
        pos += n
        continue
      if dat is not None:
        el = frombuffer( dat, f, count=n, offset=pos )
        res.append( el )
      pos += el.nbytes
  except ValueError:
    raise ValueError("Could not parse %d %s-s at position %d" % (n,str(f),pos))
  if dat is None:
    return None, pos, None
  return res, pos, dat[pos:]


def blobFrameIter( blob, trailing=False ):
  """
  Iterator yielding frames from a blob of data in the form:
  
  INPUT:
    blob -- buffer type
    trailing -- boolean -- if true, last value yielded is the 
        trailing data after last frame (possibly empty)
        
  YIELDS:
    type, header, body
    -or-
    array of trailing data dtype uint8
  """
  if not len(blob):
    return
  blob = frombuffer(blob,uint8)
  while blob.nbytes>=8:
    (sz,t),pos,tail = unpack( blob[:8], (uint32, uint32) )
    yield t, blob[:pos], blob[pos:sz]
    blob = blob[sz:]
  if trailing:
    yield blob

def udpFrameIter( sock, blocksize = 1<<16 ):
  """
  Iterator yielding frames from a UDP socket
  
  INPUT:
    blocksize -- the maximal frame size that will be returned.
      Longer frames will be truncated.
  """
  while True:
    try:
      buf = sock.recv(blocksize)
      seq = list( blobFrameIter( buf, True ) )
      while len(seq)>1:
        yield seq.pop(0)
    except SocketTimeout:
      yield None

def tcpRecvIter( sock, n ):
  """
  Iterate until n bytes received, yielding None for every timeout
  Finally, yields a buffer with n bytes.
  
  INPUT:
    sock -- an open TCP socket
    n -- a non-negative integer
  """
  l = 0
  buf = []
  while True:
    try:
      blk = sock.recv(n-l)
      buf.append( blk )
      l += len(blk)
    except SocketTimeout:
      pass
    if l<n:
      yield None
    else:
      break
  res = "".join(buf)
  assert len(res)==n
  yield res
  
def tcpFrameIter( sock ):
  """
  Iterator yielding QTM protocol frames from a TCP socket
  
  QTM frames start with a length and type, both 32-bit integers.
  
  INPUT:
    sock -- TCP socket
  YIELDS: 
    t, hdr, body -- type (uint32 array with one element), 
      header (of length 8) and body (of length spcified in her head).
  """
  while True:
    for hdr in tcpRecvIter( sock, 8 ):
      if hdr is None:
        yield None
      else:
        break
    assert len(hdr)==8
    (sz, t),_,_ = unpack( hdr, (uint32, uint32))
    for body in tcpRecvIter( sock, sz-8 ):
      if body is None:
        yield None
      else:
        break
    yield t, hdr, body
      
class AbstractQTM_Data(object):
  """
  'Abstract' class AbstractQTM_Data
  
  NOTE: This class isn't really abstract, because instances of it may
    appear if unexpected packet types are encountered.
  
  The AbstractQTM_Data exists primarily for organizational purposes as
  the superclass of all data containers for QTM protocol messages.
  
  Every subclass follows the following convention:
  (1) Its constructor takes t, hdr, dat
  (2) It overrides the QTM_TYPE class variable with the QTM type. 
      Calling __init__ with a t != QTM_TYPE raises a TypeError.
  (3) Has qtmType, qtmRawHeader and qtmRawData members that give access
      to the associated unprocessed QTM protocol data.
  """
  QTM_TYPE = 666
  
  def __init__(self, t, hdr, dat):
    self.qtmType = int(t)
    self.qtmRawHeader = hdr
    self.qtmRawData = dat
    self.dcr = "q%d:%d:%d" % (self.qtmType,len(hdr),len(dat))
    if self.qtmType != self.QTM_TYPE:
      raise TypeError('Class does not match QTM type code')
      
  def __repr__(self):
    sup = object.__repr__(self)
    return '%s %s>' % (sup[:-1],self.dcr)
    
class XML_Packet( AbstractQTM_Data ):
  QTM_TYPE = 2
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    et = ElementTree.fromstring(dat[:-1].strip())
    self.root = et
    
  @classmethod
  def build( cls, tree ):
    if type(tree) is str:
      # If a string, make sure it is correct XML
      ElementTree.fromstring(tree)
      msg = tree
    else:
      msg = ElementTree.tostring(tree)
    hdr = asarray( [len(msg)+9, cls.QTM_TYPE], uint32 ).tostring()
    dat = msg + chr(0)
    return hdr+dat

class DataPacket( AbstractQTM_Data ):
  QTM_TYPE = 3
  def __init__(self, t, hdr, dat):
    """Based on pp. 35 section 5.6"""
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    (self.time, self.frame, N),_,tail = unpack( dat, (uint64, uint32, uint32) )
    self.comp = [ newComponent(*fr) for fr in blobFrameIter( tail ) ]

  def __repr__(self):
    sup = AbstractQTM_Data.__repr__(self)
    return "%s #%d t=%d %s>" % (sup[:-1],self.frame, self.time, repr(self.comp))

class NestedData( AbstractQTM_Data ):
  def __init__(self, item, fmt, t, hdr, dat ):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    (count, self.dropRate, self.noSyncRate),_,dat = unpack( dat, (uint32,uint16,uint16))
    res = []
    self.status = []
    self.count = count
    for n in xrange(count):
      (mc,st),_,dat = unpack(dat, (uint32, uint8))
      self.status.append(int(st))
      if st: # if camera reported error --> no data
        res.append(asarray([],dtype=fmt))
        continue
      try:
        (mrk,),_,dat = unpack(dat,[(mc,fmt)])
        res.append(mrk)
      except ValueError:
        self.status[-1] |= 0x80
        res.append(asarray([],dtype=fmt))
        continue
    setattr( self, item, res )
    self._item = item
  
  def __repr__(self):
    sup = AbstractQTM_Data.__repr__(self)
    return "%s #%d %s=%s>" % (sup[:-1],self.count,self._item,repr(self.status))
    
class FlatData( AbstractQTM_Data ):
  def __init__(self, item, fmt, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    (count, self.dropRate, self.noSyncRate),_,dat = unpack( dat, (uint32,uint16,uint16))
    self.count = count
    if count:
      (res,),_,dat = unpack(dat,[(count,fmt)])
    else:
      res = asarray([],dtype=fmt)
    setattr( self, item, res )
    self._item = item

  def __repr__(self):
    sup = AbstractQTM_Data.__repr__(self)
    return "%s #%d %s=[%d]>" % (sup[:-1],self.count,self._item,len(getattr(self,self._item)))
    
class C_2D( NestedData ):
  QTM_TYPE = 7
  def __init__(self,*args ):
    NestedData.__init__(self,'cam',[('x','i4'),('y','i4'),('dx','u2'),('dy','u2')],*args)
    #self.dcr = "2D"
    
class C_2DL( NestedData ):
  QTM_TYPE = 8
  def __init__(self,*args ):
    NestedData.__init__(self,'cam',[('x','i4'),('y','i4'),('dx','u2'),('dy','u2')],*args)

class C_3D( FlatData ):
  QTM_TYPE = 1
  def __init__(self,*args ):
    try:
      # Try the correct packet fomrat
      FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4'),('lbl','u4')],*args)
    except ValueError:
      # Try a fallback
      FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4')],*args)
    
class C_3D_NoLabel( FlatData ):
  QTM_TYPE = 2
  def __init__(self,*args ):
    try:
      # This is NOT what the packet format should be, but it is what we get over UDP 
      FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4'),('resid','f4')],*args)
    except ValueError:
      # Try the correct packet format
      FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4')],*args)
    
class C_3DR( FlatData ):
  QTM_TYPE = 9
  def __init__(self,*args ):
    FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4'),('lbl','u4'),('resid','f4')],*args)

class C_3DR_NoLabel( FlatData ):
  QTM_TYPE = 10
  def __init__(self,*args ):
    FlatData.__init__(self,'mrk',[('x','f4'),('y','f4'),('z','f4'),('resid','f4')],*args)

class ImageComponent( AbstractQTM_Data ):
  QTM_TYPE = 15
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    (count,),_,dat = unpack( dat, (uint32,))
    self.cam = []
    for k in xrange(count):
      img = ImageData(dat)
      self.cam.append( img )
      dat = dat[img.nbytes:]

class ImageData( AbstractQTM_Data ):
  def __init__(self, dat):
    (self.cam, self.imgFmt, w,h,cl,ct,cr,cb,self.nbytes),ofs,dat = unpack(dat,[int32]*9)
    self.QTM_TYPE = self.imgFmt
    AbstractQTM_Data.__init__(self,t,dat[:ofs],dat[ofs:])
    self.size = (w,h)
    self.crop = (cl,ct,cr,cb)
    self.rawImg = dat[:self.nbytes]
    if self.imgFmt == 0:
      self.img = self.rawImg.reshape(w,h)
      self.dcr = 'Raw Grayscale Image'
    elif self.imgFmt == 1:
      self.img = self.rawImg.reshape(w,h,3)
      self.dcr = 'Raw BGR Image'
    elif self.imgFmt == 2:
      self.img = RuntimeError('No parser for JPG images')
      self.dcr = "JPG Image"
    elif self.imgFmt == 3:
      self.img = RuntimeError('No parser for PNG images')
      self.dcr = "PNG Image"
    else:
      self.img = RuntimeError('Unknown image type')
      self.dcr = "<<Image of unknown type>>"

def newComponent( t, hdr, body ):
  TBL = { # table at 5.6.1 pp. 36
     7 : C_2D,
     8 : C_2DL,
     2 : C_3D_NoLabel,
     1 : C_3D,
    10 : C_3DR_NoLabel,
     9 : C_3DR,
    14 : ImageComponent,
  }
  cls = TBL.get(int(t), None)
  if cls is None:
    raise KeyError("Unknown Data Component type 0x%02x" % int(t))
  return cls( t, hdr, body )

class EndOfDataPacket( AbstractQTM_Data ):
  QTM_TYPE = 4
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    assert len(dat)==0

class EventPacket( AbstractQTM_Data ):
  EVT = [ x.strip() for x in """
    >DUMMY<
    Connected
    Connection Closed
    Capture Started
    Capture Stopped
    Fetching Finished
    Calibration Started
    Calibration Stopped
    RT From File Started
    RT From File Stopped
    Waiting For Trigger
    Camera Settings Changed
    QTM Shutting Down
  """.split("\n") ]
  
  QTM_TYPE = 6
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    self.evts = []
    for di in dat:
      evt = ord(di)
      if evt>0 and evt<len(EventPacket.EVT):
        self.evts.append((di,EventPacket.EVT[evt]))
      else:
        self.evts.append((di,"<<Unknown event type %d>>" % evt))
    if not self.evts:
      self.dcr = "(No events)"
    elif len(self.evts)==1:
      self.dcr = self.evts[0][1]
    else:
      self.dcr = "evts=[%d]" % len(self.evts)
  

class ErrorPacket( AbstractQTM_Data ):
  QTM_TYPE = 0
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    assert self.qtmType is 0
    if type(dat) is not str:
      dat = dat.tostring()
    self.msg = dat[:-1]

  def __repr__(self):
    sup = AbstractQTM_Data.__repr__(self)
    return "%s '%s'>" % (sup[:-1],self.msg)

class CommandPacket( AbstractQTM_Data ):
  QTM_TYPE = 1
  def __init__(self, t, hdr, dat):
    AbstractQTM_Data.__init__(self,t, hdr, dat)
    if type(dat) is not str:
      dat = dat.tostring()
    self.msg = dat[:-1]
  
  @classmethod
  def build( cls, msg ):
    hdr = asarray( [len(msg)+9, cls.QTM_TYPE], uint32 ).tostring()
    dat = msg+chr(0)
    return hdr+dat

  def __repr__(self):
    sup = AbstractQTM_Data.__repr__(self)
    return "%s '%s'>" % (sup[:-1],self.msg)
    
def newDiscoverPacket( port ):
  hdr = asarray( [10, 7], uint32 ).tostring()
  body = asarray( [(port>>8) &  0xFF, port & 0xFF], uint8 ).tostring()
  return hdr+body
  
PACKET_TYPES = { #: table mapping QTM type fields to packet classes
    0 : ErrorPacket,
    1 : CommandPacket,
    2 : XML_Packet,
    3 : DataPacket,
    4 : EndOfDataPacket,
    6 : EventPacket
}

def packetIter( frameIter ):
  """
  Returns an iterator that converts frames into AbstractQTM_Data objects.
  """
  for fr in frameIter:
    if fr is None:
      yield None
    else:
      cls = PACKET_TYPES.get( int(fr[0]), AbstractQTM_Data )
      obj = cls(*fr )
      yield obj
      

class QTM( object ):
  """
  Concrete class QTM 
  
  Instances of this class provide a communication interface to a
  QTM manager instance running somewhere on the network.
  
  THEORY OF OPERATION
  
  A QTM instance holds public attributes representing the tcp and udp
  sockets that communicate with the QTM manager. In addition, it holds
  two packetIterator-s, one for each socket, allowing QTM protocol 
  messages to be read from the sockets.
  
  METHODS
  
    connect -- connect to a specific host running QTM manager
    connectOnNet -- use a network broadcast discovery message to find
       a QTM manager instance and connect to it.
    isConnected -- true if connected to QTM manager
    command -- sent a QTM Command
    sendParamXML -- send a QTM XML Packet (parameter configuration)
    close -- close sockets
    
  TYPICAL USAGE
  
  >>> qtm = QTM()
  >>> qtm.connectOnNet( '10.255.255.255' )
  >>> for pkt in qtm.ptcp:
  ...   print ptk
  ...   if pkt is None:
  ...     break
  
  will connect to a local QTM manager and start a protocol Version 1.9
  session.
  """
  DISCOVER_PORT = 22226
  BASE_PORT = 22222
  
  def __init__(self):
    """Create a new QTM interface object
    
    ATTRIBUTES:
      .udp -- a UDP socket; always open
      .tcp -- a TCP socket; open between connect and kill
    """
    self._newSocks()
    
  def _newSocks( self ):
    self.tcp = socket(AF_INET,SOCK_STREAM)
    self.tcp.settimeout(0.2)
    u = socket(AF_INET,SOCK_DGRAM)
    u.bind(("",0))
    u.settimeout(0.2)
    u.setsockopt( SOL_SOCKET, SO_BROADCAST, 1 )
    self.udp = u
    self.pudp = packetIter(udpFrameIter( u ))
    self.ptcp = packetIter(tcpFrameIter( self.tcp ))    

  def connectOnNet( self, net ):
    """Use discovery to connect to a QTM server.
    
    Returns the response command packet if successful,
    None if failed
    """
    pkt = newDiscoverPacket( self.udp.getsockname()[-1] )
    self.udp.sendto( pkt, (net,self.DISCOVER_PORT) )
    try:
      pkt, addr = self.udp.recvfrom(1000)
    except SocketTimeout:
      raise SocketTimeout("QTM connection attempt timed out")
    self.connect(addr[0])
  
  def isConnected( self ):
    return self.tcp.getsockname()[-1] != 0
    
  def connect( self, addr ):
    if self.isConnected():
      self.close()
    # Test local endianism and connect accordingly
    if frombuffer('\0\1',uint16)==256:
      self.tcp.connect( (addr,self.BASE_PORT+1) )
    else:
      self.tcp.connect( (addr,self.BASE_PORT+2) )
    
  def command( self, cmd ):
    msg = CommandPacket.build(cmd)
    self.tcp.send(msg)

  def sendParamXML( self, xml ):
    pkt = XML_Packet.build(xml)
    self.tcp.send(pkt)
    
  def close(self):
    self.tcp.close()
    self.udp.close()
    self._newSocks()
    
