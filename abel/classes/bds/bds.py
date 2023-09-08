from abc import abstractmethod
from matplotlib import patches
from abel import Trackable

class BeamDeliverySystem(Trackable):
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_nom_energy(self):
        pass
    
    def survey_object(self):
        rect = patches.Rectangle((0, -0.1), self.get_length(), 0.2)
        rect.set_facecolor = 'r'
        return rect