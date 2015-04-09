""" simpleData.py

    This module implements a loadData() function that can read 'simpleSave'
    files -- these are pairs of matching files with suffixes .sss and .ssd
    
    The .sss file is a text file containing code describing how to parse the
    binary data in the associated .ssd file.
    
    The approach used is a dirty hack: the .sss file is processed as a string
    to convert into valid python commmands. It is then evaluated *as code* in 
    an environment that has the .sss keywords bound to methods that implement
    the required actions.
    
    After completion, the final structure is (optionally) subjected to 
    a simplification operation that makes it more straight-forward to use.
    
    (c) By Shai Revzen, Berkeley 2008, U. Penn 2010    
"""
from numpy import *

class Struct:
  """Class representing simpleSave REC structures
  
     REC structures are arrays of records which have identical fields, though
     the values and structures of the values in different records may not match.
     
     Fields of a REC are each represented by an object array of the shape of
     the REC. For example, a (3,2) shaped REC with fields x,y,name will have
     fields x, y, and name, each of which is a (3,2) object array.
  """
  def __init__(self, sz, names):
    self._names = list(names)
    self._csr = (0,0)
    self.shape = sz
    self.isDone = False
    N = prod(sz)    
    for nm in names:
      if hasattr(self,nm):
        raise KeyError,"Field name %s conflicts with internal name" % nm
      setattr( self, nm, resize(array( None, dtype=object ),(N,)))
  
  def __getitem__( self, idx ):
    res = Struct(None,[])
    res._names = self._names
    for nm in self._names:
      val = getattr(self, nm)[idx]
      setattr( res, nm, val )
    res.shape = val.shape
    return res
  
  def __len__(self):
    return len(getattr(self,self._names[0]))
    
  def __setitem__(self, idx, val ):
    raise IndexError, "Indexed assignment is not supported"
  
  def addfield( self, **kw ):
    def sqz(shape):
      return tuple([s for s in shape if s != 1])
    for nm,val in kw.iteritems():
      assert sqz(asarray(val).shape) == sqz(self.shape),"Value must match data shape"
      assert not hasattr(self,nm),"Cannot overwrite existing fields"
    for nm,val in kw.iteritems():
      val = asarray(val)
      self._names.append(nm)
      setattr( self, nm, val )
    return self
  
  def rmfield( self, *fn ):
    for nm in fn:
      assert hasattr(self,nm),"Fields must exist"
    for nm in fn:
      self._names.remove(nm)
      delattr(self,nm)
      
  def _simplify( self ):
    # Simplify recusively
    for nm in self._names:
      p = getattr(self, nm)
      if prod(self.shape)==1:
        setattr( self, nm, simplify(p.flat[0]) )
      else:
        setattr( self, nm, simplify(p) )
    return self
      
  def dshow( self ):
    if self.isDone:
      return "<Struct@%x sz %s>" %(
        id(self), self.shape )      
    return "<Struct@%x at %s %s/%d>" % (
      id(self), self._names[self._csr[0]], self._csr[1],prod(self.shape))
    
  def _put(self, val):
    fi,pos = self._csr
    # Find the field being addressed and insert the new value
    p = getattr(self, self._names[fi])
    ### DEBUG: print self.dshow(), " = %s@%x" % (val.__class__.__name__, id(val))
    if p.shape==():
      p.put(0,val)
    else:
      p[pos]=val
    # Increment the index
    fi = (fi+1) % len(self._names)
    if fi == 0:
      pos += 1
      if pos==prod(self.shape):
        # Completed
        for nm in self._names:
          getattr(self, nm).shape = self.shape
        self.isDone = True
    self._csr=(fi,pos)
    return self.isDone

  def fieldnames( self ):
    """Return a list of the data field names in this struct"""
    return self._names
  
  def __repr__( self ):
    res = [ "%s.%s at 0x%X :" % (
        self.__module__,self.__class__.__name__,hash(self) )]
    for fn in self._names:
      fv = getattr(self,fn)
      if isinstance(fv,(Cell,Struct)):
        s = "    %s = %s" % (fn,object.__repr__(fv))
      elif isinstance(fv,ndarray):
        dt = fv.dtype
        if dt is dtype('object'):
          if isinstance(fv.flat[0],ndarray):
            dt = fv.flat[0].dtype
          else:
            dt = type(fv.flat[0])
        s = "    %s : %s, %s" % (fn,str(fv.shape),str(dt))
      else:
        s = "    %s = %s" % (fn,repr(fv))
      res.append(s)
    res.append("ENDS: "+res[0][:-1])
    return "\n".join(res)
      
class Cell:
  """Class representing simpleSave OBJ structures
  
     REC structures are represented by an object array of the same shape,
     stored in the field .at
  """
  def __init__(self, sz):
    self.sz = sz
    self.end = prod(sz)
    self.at = resize(array( None, dtype=object ),(self.end,))
    self.pos = 0
    self.isDone = False
  
  def _simplify( self ):
    return simplify( self.at)

  def dshow( self ):
    if self.isDone:
      return "<Cell@%x sz %s>" % (
        id(self), self.sz)      
    return "<Cell@%x at %d/%d>" % (
      id(self), self.pos, self.end)

  def _put( self, val ):
    if not self.isDone:
      self.at[self.pos] = val
      self.pos += 1      
    if self.pos >= self.end:
      self.at.shape = self.sz
      self.isDone = True
    return self.isDone
  
  def __repr__( self ):
    return object.__repr__(self) + ":" + repr(self.at)
    
class SimpleSaveReader:
  def __init__(self):
    self.stack = [Cell((1,))]
    self.rawfile = None
    self.root = None
  
  def open( self, fn ):
    self.rawfile = open( fn+".ssd", "rb" )
    self.fn = fn
 
  def getSFN( self ):
    return "%s.sss" % self.fn

  def _put( self, val, new = None ):
    """Put the newly generated object in the next location"""
    ### DEBUG: print "[%s]" % " ".join([x.dshow() for x in self.stack])
    # Put in top object
    top = self.stack[-1]
    top._put(val)
    
  def _pop(self):
    """Pop all complete objects off of the stack"""
    # Pop all complete objects
    while len(self.stack)>1:
      top = self.stack[-1]
      if not top.isDone:
        break
      ### DEBUG: print "   --> POP",top.dshow()
      self.stack.pop()
      
  def _read( self, sz, dtype ):
    """Return an array of the specified size and type taken from raw input"""
    dat = fromfile( self.rawfile, count=prod(sz), dtype=dtype )
    dat.shape = sz
    return dat
  
  def REC( self, sz, *names ):
    """Structs are instances of class Struct with a member for each field"""
    S = Struct(sz,names)
    self._put( S )
    self.stack.append(S)
    ### DEBUG: print "Struct @ %x" % id(S), ":= REC",sz,names  
    self._pop()

  def COMPLEX_double( self, sz ):
    re = self._read( sz, dtype=float64 )
    re.astype(complex128)
    im = self._read( sz, dtype=float64 )
    dat = re + 1j * im
    ### DEBUG: print "COMPLEX_double",sz
    self._put( dat )
    self._pop()
  
  def REAL_double( self, sz ):
    ### DEBUG: print "REAL_double",sz
    self._put( self._read( sz, dtype=float64 ))
    self._pop()
  
  def BOOL( self, sz ):
    ### DEBUG: print "BOOL",sz
    dat = self._read( sz, bool8 )
    self._put( dat )
    self._pop()
    
  def CHAR( self, sz ):
    # If this was just a string
    if len(sz)==2 and sz[1]==1:
      # Read as a string, using the python file.read method
      dat = self.rawfile.read(sz[0])
    else:
      # Otherwise, read as an array of bytes
      dat = self._read( sz, uint8 )
    self._put( dat )
    self._pop()
  
  def OBJ( self, sz ):
    C = Cell(sz)
    self._put( C )
    self.stack.append(C)
    ### DEBUG: print "Cell @ %x" % id(C), ":= OBJ",sz
    self._pop()

  def SZ( self, *dims ):
    """Matrix sizes. MatLab is row-major; numpy is column-major"""
    dims = list(dims)
    dims.reverse()
    return tuple(dims)
  
  def NAME( self, nm ):
    """Names are kept as-is"""
    return nm
  
  def FUNCTION_HANDLE( self, nm ):
    self._put("FUNCTION_HANDLE('%s')" % nm)
    self._pop()
    
  def INLINE( self, nm, expr ):
    self._put("INLINE('%s','%s')" % (nm,expr))
    self._pop()
    
  def DECL_ENDS( self, dummy ):
    """Ends of structure declarations"""
    ### DEBUG: print "<<DECL_ENDS>>"
    self.root = self.stack[-1]

def simplify( obj ):
  """Simplify objects read from a simpleSave file
     The simplification removes unnecessary encapsulation, 
     e.g. object arrays containing a single entry are replaced by 
       the entry itself. 
       
     Generally, the simplified result is closer to what you would expect
     but less regular in the way it treats the actual data structures
  """
  # Cell and Struct classes have specific methods for simplification
  if isinstance(obj,(Struct,Cell)):
    return obj._simplify()
  # Arrays require special handling
  if isinstance(obj,ndarray):
    # If only 1 element -- simplification is the simplified element
    if len(obj.flat)==1:
      return simplify(obj.flat[0])
    # If elements are objects -- recurse simplification 
    elif obj.dtype == dtype('object'):
      if len(obj.shape)==2 and obj.shape[1]==1:
        obj = obj.flatten()
      for k in xrange(len(obj.flat)):
        obj.flat[k] = simplify( obj.flat[k] )
      # if single elements then try to convert to numeric array
      if isinstance(obj.flat[0],number):
        try:
          obj = obj.astype(number)
        except ValueError:
          pass
          
  return obj

def loadData( fn, raw=False ):
  """Load simpleSave data from file path fn
  
     fn should not include the suffixes .sss or .ssd
     
     Returns an object heirarchy representing the data in the .sss and .ssd
     file pairs.
     
     If raw is true, the object heirarcy is not simplify()-ed after loading.
  """
  R = SimpleSaveReader()
  # open the .sss structure file and read all the structure into a string
  R.open(fn)
  try:
    f = open(R.getSFN(),"r")
    sd = "".join(f.readlines())
  finally:
    f.close()
  # convert comments to python style comments
  sd = sd.replace("%","#")
  # build an environment with the simpleSave operations calling bound methods
  #   of the object R, i.e. acting on R
  env = {}
  for nm in dir(R):
    if nm[0]>='A' and nm[0]<='Z':
      env[nm] = getattr(R,nm)
  # compile the .sss structure code 
  c = compile( sd, R.getSFN(), "exec" )
  # evaluate it using this environment to build the object
  eval( c, globals(), env )
  if raw:
    return R.root
  return simplify(R.root)


