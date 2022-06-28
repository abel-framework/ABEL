from src.classes.Trackable import *
from abc import abstractmethod
from matplotlib import patches

class Stage(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam):
        beam.stageNumber += 1
        return super().track(beam)
    
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def energyGain(self):
        pass
    
    def plotObject(self):
        return patches.Rectangle((0, -1), self.length(), 2)