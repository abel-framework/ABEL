from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Stage(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stageNumber += 1
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def energyGain(self):
        pass
    
    @abstractmethod
    def energyEfficiency(self):
        pass
    
    @abstractmethod
    def energyUsage(self):
        pass
    
    def plotObject(self):
        return patches.Rectangle((0, -1), self.length(), 2)