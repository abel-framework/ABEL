from abc import abstractmethod
from abel.classes.trackable import Trackable

class Spectrometer(Trackable):
    
    @abstractmethod
    def __init__(self, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=None, object_plane_y=None, magnification_x=None, magnification_y=None):
        super().__init__()

        self.imaging_energy_x = imaging_energy_x
        self.imaging_energy_y = imaging_energy_y
        self.object_plane_x = object_plane_x
        self.object_plane_y = object_plane_y
        self.magnification_x = magnification_x
        self.magnification_y = magnification_y
        
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    def survey_object(self):
        from matplotlib import patches
        rect = patches.Rectangle((0, -0.05), self.get_length(), 0.1)
        rect.set_facecolor = 'k'
        return rect