from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Interstage(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam):
        return super().track(beam)
    
    @abstractmethod
    def length(self):
        pass
    
    def plotObject(self):
        rect = patches.Rectangle((0, -0.05), self.length(), 0.1)
        rect.set_facecolor = 'k'
        return rect
        