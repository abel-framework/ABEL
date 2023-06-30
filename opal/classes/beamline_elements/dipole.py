from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Dipole(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_angle(self):
        pass
    
    def survey_object(self):
        rect = patches.Rectangle((0, -1), self.get_length(), 2)
        rect.set_facecolor = 'r'
        return rect