from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class BeamDeliverySystem(Trackable):
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def length(self):
        pass
    
    def plotObject(self):
        rect = patches.Rectangle((0, -0.1), self.length(), 0.2)
        rect.set_facecolor = 'r'
        return rect