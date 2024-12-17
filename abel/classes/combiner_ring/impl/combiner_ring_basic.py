import scipy.constants as SI
import copy
import numpy as np
from abel.classes.combiner_ring.combiner_ring import CombinerRing

class CombinerRingBasic(CombinerRing):
    
    def __init__(self, nom_energy=None, compression_factor=None, exit_angle=CombinerRing.default_exit_angle):
        
        super().__init__(nom_energy=nom_energy, compression_factor=None, exit_angle=exit_angle)

        self.max_dipole_field = 1.5
        self.max_rel_energy_loss = 0.01
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # outgoing bunch separation (compressed)
        self.bunch_separation = beam0.bunch_separation/self.compression_factor
        self.num_bunches_in_train = beam0.num_bunches_in_train
        
        # compress the train
        beam = copy.deepcopy(beam0)
        beam.bunch_separation = self.bunch_separation

        # warn if field or energy loss is too high
        if not self.nom_energy is None:
            if self.get_average_dipole_field() > self.max_dipole_field:
                print(f"Dipole field too high: {self.get_average_dipole_field():.1f} T")
            if self.get_rel_energy_loss() > self.max_rel_energy_loss:
                print(f"Relative energy loss too high: {100*self.get_rel_energy_loss():.1f}%")
        else:
            print("CombinerRingBasic: Could not evaluate dipole field or energy loss limits because nom_energy not set")
        
        return super().track(beam, savedepth, runnable, verbose)

    def get_bend_radius(self):
        return self.get_length()/(2*np.pi)
    
    def get_length(self):
        return self.get_bunch_separation_outgoing()*self.num_bunches_in_train*SI.c

    def get_average_dipole_field(self):
        # average dipole field based on energy and radius
        return self.nom_energy / (SI.c * self.get_bend_radius())
    
    def get_num_turns(self):
        return self.compression_factor
        
    def get_rel_energy_loss(self):
        
        rel_energy_loss_per_turn = 2 * np.pi * SI.e**2 * (self.nom_energy*SI.e)**3 / (6 * np.pi * SI.epsilon_0 * self.get_bend_radius() * (SI.m_e*SI.c**2)**4)

        return self.get_num_turns() * rel_energy_loss_per_turn
        
