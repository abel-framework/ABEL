from abc import abstractmethod
from matplotlib import patches
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np

class Interstage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None, beta0=None):
        
        super().__init__()

        self.nom_energy = nom_energy
        self._dipole_length = dipole_length
        self._dipole_field = dipole_field
        self._beta0 = beta0
        
        self.evolution = SimpleNamespace()

    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    def __energy_fcn(self, fnc_or_value):
        if callable(fnc_or_value):
            return fnc_or_value(self.nom_energy)
        else:
            return fnc_or_value
            
    # evaluate dipole length (if it is a function)
    @property
    def dipole_length(self) -> float:
        return self.__energy_fcn(self._dipole_length)
    @dipole_length.setter
    def dipole_length(self, val):
        self._dipole_length = val

    
    # evaluate dipole field (if it is a function)
    @property
    def dipole_field(self) -> float:
        return self.__energy_fcn(self._dipole_field)
    @dipole_field.setter
    def dipole_field(self, val):
        self._dipole_field = val

    # evaluate initial beta function (if it is a function)
    @property
    def beta0(self) -> float:
        return self.__energy_fcn(self._beta0)
    @beta0.setter
    def beta0(self, val):
        self._beta0 = val
    
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
        