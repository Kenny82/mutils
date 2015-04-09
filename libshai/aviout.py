import time, glob, os
from pylab import *

class AviOut:
  def __init__( self, pth, fps=25):
    self.n = 1
    self.fig = gcf()
    if type(pth)==str:
      pfx = pth[:-4]
      if pth[-4:].lower() != ".avi":
        raise TypeError,"Filename pattern must end with .avi"
      if glob.glob(pfx):
        raise KeyError,"A file/dirctory by the name '%s' exists -- bailing out" % pfx
      os.system('mkdir -p %s' % pfx)
      self.pfx = pfx
      self.pth = pth
      self.fps = fps
      self.step = self._step_AVI
      self.stop = self._stop_AVI
    else:
      self.dur = float(pth)
      self.step = self._step_PAUSE
      self.stop = self._stop_PAUSE
    self.fig.set_visible(False)
      
  def _step_PAUSE(self):
    self.fig.set_visible(True)
    draw()
    time.sleep(self.dur)
    self.fig.set_visible(False)

  def _stop_PAUSE(self):
    self.fig.set_visible(True)
    draw()
  
  def _step_AVI(self):
    self.fig.set_visible(True)
    self.n = self.n + 1
    draw()
    savefig( "%s/fr-%04d.png" % (self.pfx,self.n) )
    self.fig.set_visible(False)

  def _stop_AVI(self):
    self.fig.set_visible(True)
    draw()
    exc = None
    try:
      os.system("mencoder mf://%s/fr-*.png "
        "-mf fps=%d:type=png -ovc lavc "
        "-lavcopts vcodec=mpeg4:mbd=2:trell"
        " -oac copy -o %s" % (self.pfx,self.fps,self.pth))
    except ex:
      exc = ex
    os.system('rm -rf %s' % self.pfx )      
    if exc is not None:
      raise exc

