from abc import abstractmethod
from matplotlib import patches
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np

class Interstage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self):
        
        super().__init__()
        
        self.evolution = SimpleNamespace()

    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    
    @abstractmethod
    def get_length(self):
        pass
    
    def get_cost_breakdown(self):
        return ('Interstage', self.get_length() * CostModeled.cost_per_length_interstage)
    
    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'Interstage'
        color = 'orange'
        return x_points, y_points, final_angle, label, color
        