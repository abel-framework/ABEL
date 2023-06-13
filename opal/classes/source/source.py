from abc import abstractmethod
from matplotlib import patches
from opal import Trackable

class Source(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_energy(self):
        pass
    
    def get_energy_gain(self):
        return self.get_energy()
        
    @abstractmethod
    def get_energy_efficiency(self):
        pass
    
    @abstractmethod
    def get_charge(self):
        pass
    
    def get_energy_usage(self):
        return self.get_energy_gain()*abs(self.get_charge())/self.get_energy_efficiency()
    
    def survey_object(self):
        rect = patches.Rectangle((0, -0.5), self.get_length(), 1)
        return rect
    