from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Drift(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def length(self):
        pass
    
    def plotObject(self):
        rect = patches.Rectangle((0, -0), self.length(), 0)
        rect.set_facecolor = 'k'
        return rect