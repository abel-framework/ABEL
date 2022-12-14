from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Source(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def energy(self):
        pass
    
    def plotObject(self):
        rect = patches.Rectangle((0, -0.5), self.length(), 1)
        return rect