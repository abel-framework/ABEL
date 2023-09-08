from abc import abstractmethod
from matplotlib import patches
from abel import Trackable

class Spectrometer(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    def survey_object(self):
        rect = patches.Rectangle((0, -0.05), self.get_length(), 0.1)
        rect.set_facecolor = 'k'
        return rect