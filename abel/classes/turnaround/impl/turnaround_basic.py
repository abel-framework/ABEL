import scipy.constants as SI
import numpy as np
from abel.classes.turnaround.turnaround import Turnaround

class TurnaroundBasic(Turnaround):
    
    def __init__(self, nom_energy=None, use_semi_circle=False, fill_factor=0.9, max_dipole_field=1.4, max_rel_energy_loss=0.005):

        super().__init__(nom_energy, use_semi_circle)
        
        self.fill_factor = fill_factor
        self.max_dipole_field = max_dipole_field
        self.max_rel_energy_loss = max_rel_energy_loss
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_bend_radius(self):

        # minimum bend radius based on B-field only
        bend_radius_field = self.nom_energy / (SI.c * self.max_dipole_field)
        
        # minimum bend radius based on
        bend_radius_rad = self.get_length_per_radius() * SI.e**2 * (self.nom_energy*SI.e)**3 / (6 * np.pi * SI.epsilon_0 * self.max_rel_energy_loss * (SI.m_e*SI.c**2)**4)
        
        # use the maximum of the two radii above
        return max(bend_radius_field, bend_radius_rad)

    
    def get_length_per_radius(self):
        if self.use_semi_circle:
            return np.pi
        else:
            return 2 * np.pi

    
    def get_length(self):
        return self.get_length_per_radius() * self.get_bend_radius() / self.fill_factor
