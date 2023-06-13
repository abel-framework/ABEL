from abc import abstractmethod
from matplotlib import patches
from opal import Trackable
import numpy as np

class Quadrupole(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_strength(self):
        pass
    
    def plotObject(self):
        return patches.Rectangle((0, 0), self.get_length(), float(np.sign(self.get_strength())))