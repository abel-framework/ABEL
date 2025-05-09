from abc import abstractmethod
from abel.classes.trackable import Trackable
import numpy as np

class PlasmaLens(Trackable):
    
    @abstractmethod
    def __init__(self, length, radius, current, offset_x=0, offset_y=0):

        super().__init__()
        
        # common variables
        self.length = length
        self.radius = radius
        self.current = current

        self.offset_x = offset_x
        self.offset_y = offset_y
        
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod
    def get_focusing_gradient(self):
        pass
        
    def get_length(self):
        return self.length

    def survey_object(self):
        from matplotlib import patches
        return patches.Rectangle((0, -1), self.get_length(), 2)
    