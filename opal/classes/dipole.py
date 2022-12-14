from abc import abstractmethod
from matplotlib import patches
from opal.classes.trackable import Trackable

class Dipole(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def field(self):
        pass
    
    def plotObject(self):
        rect = patches.Rectangle((0, -1), self.length(), 2)
        rect.set_facecolor = 'r'
        return rect