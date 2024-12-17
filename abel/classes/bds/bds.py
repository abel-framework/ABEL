from abc import abstractmethod
from matplotlib import patches
import numpy as np
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

class BeamDeliverySystem(Trackable, CostModeled):
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass

    def get_cost_breakdown(self):
        return ('BDS', self.get_length() * CostModeled.cost_per_length_bds)
    
    @abstractmethod
    def get_nom_energy(self):
        pass
    
    def survey_object(self):
        #rect = patches.Rectangle((0, -0.1), self.get_length(), 0.2)
        #rect.set_facecolor = 'r'
        #return rect
        
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'BDS'
        color = 'lightgray'
        return x_points, y_points, final_angle, label, color