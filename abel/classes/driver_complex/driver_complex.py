from abc import abstractmethod
from matplotlib import patches
from abel import Trackable

class DriverComplex(Trackable):
    
    @abstractmethod
    def __init__(self, source=None, rf_accelerator=None, nom_energy=None, num_drivers=None):
        self.source = source
        self.rf_accelerator = rf_accelerator
        self.nom_energy = nom_energy
        self.num_drivers = num_drivers
        
    @abstractmethod
    def track(self, _=None, savedepth=0, runnable=None, verbose=False, stage_number=None):
        pass
    
    def energy_usage(self):
        return self.num_drivers * (self.source.energy_usage() + self.rf_accelerator.energy_usage())
    
    # driver train rep rate
    def rep_rate_average(self):
        return self.rf_accelerator.rep_rate_average()
        
    def wallplug_power(self):
        return self.rep_rate_average() * self.energy_usage()

    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_cost(self):
        pass
    
    # make survey rectangles
    def survey_object(self):
        objs = []
        objs.append(self.source.survey_object())
        objs.append(self.rf_accelerator.survey_object())
        return objs
        