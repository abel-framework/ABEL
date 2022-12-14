from abc import abstractmethod
from matplotlib import patches
from opal.classes.trackable import Trackable
import numpy as np

class Quadrupole(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def strength(self):
        pass
    
    def plotObject(self):
        return patches.Rectangle((0, 0), self.length(), float(np.sign(self.strength())))