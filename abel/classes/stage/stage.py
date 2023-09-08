from abc import abstractmethod
from matplotlib import patches
from abel import Trackable

class Stage(Trackable):
    
    @abstractmethod
    def __init__(self, length, nom_energy_gain, plasma_density):
        self.nom_energy_gain = nom_energy_gain
        self.length = length
        self.plasma_density = plasma_density
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stage_number += 1
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_nom_energy_gain(self):
        pass
    
    @abstractmethod
    def matched_beta_function(self, energy):
        pass
    
    @abstractmethod
    def energy_efficiency(self):
        pass
    
    @abstractmethod
    def energy_usage(self):
        pass
    
    @abstractmethod
    def plot_wakefield(self):
        pass
    
    def survey_object(self):
        return patches.Rectangle((0, -1), self.get_length(), 2)