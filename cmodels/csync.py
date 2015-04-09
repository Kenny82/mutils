import numpy
import re
from itertools import product as iproduct

def getSrc(fn):
  "Get contents of a source (text) file, e.g. src = getSrc('foo.c')"
  f = open(fn,'r')
  src = f.readlines()
  f.close()
  return src

def getSlice( src, st, en ):
  """getSlice( src, st, en ) -- scan src and return a list [m_st:m_en] 
     Where m_st is the position of the first match of st and
     m_en is the first match of en following m_st.
     st and en both have a .match(string) method (e.g. regex-s) """
  res = []
  for line in src:
    if not res:
      if st.match(line):
        res.append(line)
    else:
      if en.match(line):
        break
      res.append(line)
  return res

def getFields( src, nm, NL ):
  """getFields( src, nm, NL ) -- parse C source src to find 
     definition of struct nm (all fields must be doubles) and
     list the fields. 
     
     The legacy form of this function handles arrays only with the 
     form double foo [NUM_LEGS]. In this case, NL is an integer and
     the entry is expanded to a list of fields foo_0 foo_1, foo_(NL-1)
     
     In its newer form, NL is a dictionary mapping names to tuples of
     integers. The field double bar [BAR], with NL = {'BAR':(2,3)} 
     gives fields bar_0_0, bar_0_1, bar_0_2, bar_1_0, bar_1_1, bar_1_2
     In addition, integers in the brackets are converted to vector
     lengths, i.e. double foo[2] becomes foo_0, foo_1.     
  """
  # Backward compatible NL format
  if type(NL) != dict:
    NL = {'NUM_LEGS' : (NL,) }
  sl = getSlice( src, 
    re.compile('\s*struct\s+%s\s*[{].*' % nm),
    re.compile('\s*[}]\s*[;][^E]*ENDS:\s+struct\s+%s.*' % nm) )
  if not sl:
    return None
  sl.pop(0)
  res = []
  rex = re.compile('\s*double\s+(\w+)\s*(?:\[\s*(\w+)\s*\])?.*')
  for item in sl:
    m = rex.match(item)
    if m is None:
      raise ValueError,"Line '%s' cannot be parsed" % item
    var = m.group(1)
    if m.group(2):
      # Try to resolve from NL dictionary
      sz = NL.get(m.group(2),None)
      # If failed --> resolve as integer
      if sz is not None:
        idx = [ var+"".join(["_%d" % k for k in x]) 
            for x in iproduct(*map(xrange,sz))]
      else:
        sz = int(m.group(2))
        idx = [ "%s_%d" % (var,idx) for idx in xrange(sz) ]
      res.extend(idx)
    else:
      res.append(var)
  return res
  
class StructOfDoubles(object):
  """Superclass of 'easy' interface to C structs of doubles:
     
     Usage:
       import csync
       src = csync.getSrc( 'MyC.c' )
       class MyStruct( csync.StructOfDoubles ):
       Fields = csync.getFields(src,'MyStruct',{ 'N':(2,3) })
     
     Instances of MyStruct will have the same fields in Python as in C,
     and the conversion M.toArray() will make a numpy double array from
     the Python object.
     
     SOD-s also have a convenient set(**kw) method for setting fields
     
     The default value of fields in the SOD is their index, so:
       y[sod.foo] has the same value as sod.foo after sod.fromArray(y)
  """
  def __init__( self, nm=None ):
    if nm is None:
      nm = self.__class__.Fields
    self.__nm = nm
    for k in xrange(len(nm)):
      self.__dict__[nm[k]] = k

  def __len__( self ):
    return len(self.__nm)
  
  def __getitem__( self, idx ):
    if type(idx) is slice:
      idx = xrange(*idx.indices(len(self)))
    elif type(idx) in [float,int]:
      idx = [int(idx)]
    return numpy.array([
        self.__dict__[self.__nm[k]] for k in idx
        ], dtype=numpy.double, order='C')
  
  def __setitem__( self, idx, val ):
    for k,v in zip(idx,val):
      self.__dict__[self.__nm[k]] = v
      
  def set( self, **kw ):
    self.__dict__.update(kw)
    return self

  def fromArray( self, ar ):
    for f,val in zip(self.__nm,ar):
      self.__dict__[f] = val
    return self

  def update( self ):
    "used by subclasses"
    pass
  
  def saveStr( self ):
    self.update()
    res = [ "%s = %s" % (k,self.__dict__[k]) for k in self.__nm ]
    return ("%s().set(\n  " % self.__class__.__name__
           +",\n  ".join(res)
           +" )")
    
  def toArray( self ):
    self.update()
    return self[:]

  def toDict( self ):
    self.update()
    res = {}
    for k in self.__nm:
      res[k] = self.__dict__[k]
    return res

def dim( cls ):
  """Dimension of a StructOfDoubles class"""
  return len(cls.Fields)

class FlexSOD( StructOfDoubles ):
  """A Flexible Structure of Doubles, which allows additional
     attributes to be added and used for updating the superclass
     attributes.
     
     A new update attribute can be created by naming a method
       def updPOS_ATTR( self, name, value)
     Where POS is a sorting key used set the order of updates, e.g.
       upd1_x precedes upd10_yy precedes updAA_z
     ATTR is the attribute name. Attributes that are omitted or set
       to None are skipped in the update process.
       
     If fsd is a FlexSOD with the above methods, the commands:
       fsd.x = 7
       print fsd.toArray()
     Will cause upd1_x to be called before the toArray() call.
  """
  def __init__( self, *argv, **kwarg ):
    StructOfDoubles.__init__(self,*argv,**kwarg)
    # scan for update methods
    plan = []
    for k,v in self.__class__.__dict__.iteritems():
      spl = k.split("_",1)
      if len(spl)!=2 or spl[0][:3]!="upd" or not callable(v):
        continue
      plan.append( (spl[0],spl[1],v) )
    plan.sort()
    self.__upd = [(attr,meth) for pos,attr,meth in plan]
  
  def update( self ):
    for attr,meth in self.__upd:
      val = getattr( self, attr, None )
      if val is None:
        continue
      meth( self, attr, val )

