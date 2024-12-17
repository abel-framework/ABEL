import scipy.constants as SI
import numpy as np
from abel.classes.damping_ring.damping_ring import DampingRing

class DampingRingBasic(DampingRing):
    
    def __init__(self, nom_energy=None, emit_nx_target=None, emit_ny_target=None, bunch_separation_in_ring=8e-9, fill_factor=0.9, max_dipole_field=1, rel_energy_loss_per_turn=0.001, num_rings=1):

        super().__init__(nom_energy, emit_nx_target, emit_ny_target, bunch_separation_in_ring, num_rings)
        
        self.fill_factor = fill_factor
        self.max_dipole_field = max_dipole_field
        self.rel_energy_loss_per_turn = rel_energy_loss_per_turn

        self.wallplug_to_rf_efficiency = 0.5

        self.bunch_charge = None
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # set the emittance to be lower
        self.bunch_charge = beam.abs_charge()
        self.emit_nx_initial = beam.norm_emittance_x()
        self.emit_ny_initial = beam.norm_emittance_y()

        # set the emittance to be lower
        beam.scale_norm_emittance_x(self.emit_nx_target)
        beam.scale_norm_emittance_y(self.emit_ny_target)
        
        return super().track(beam, savedepth, runnable, verbose)


    def get_damping_time(self):

        # damping ratio in each plane
        damping_ratio_x = self.emit_nx_initial/self.emit_nx_target
        damping_ratio_y = self.emit_ny_initial/self.emit_ny_target

        # number of damping times required
        num_damping_times_x = np.log(damping_ratio_x) / self.num_rings
        num_damping_times_y = np.log(damping_ratio_y) / self.num_rings

        # calculate the characteristic damping time 
        char_damping_time_x = self.get_time_per_turn() / self.rel_energy_loss_per_turn
        char_damping_time_y = 2 * char_damping_time_x

        # calculate the total damping time
        total_damping_time_x = num_damping_times_x * char_damping_time_x
        total_damping_time_y = num_damping_times_y * char_damping_time_y
        total_damping_time = max(total_damping_time_x, total_damping_time_y)
        
        return total_damping_time

    
    def get_time_per_turn(self):
        return self.get_circumference() / SI.c

    
    def get_emitted_power(self):
        energy_loss_per_turn = self.rel_energy_loss_per_turn * self.nom_energy
        return self.num_bunches_in_train * self.bunch_charge * energy_loss_per_turn / self.get_time_per_turn()

    
    def get_circumference(self):
        
        # minimum bend radius based on B-field only
        bend_radius_field = self.nom_energy / (SI.c * self.max_dipole_field)

        # minimum size to contain all bunches
        bunch_train_length = self.num_bunches_in_train * self.bunch_separation_in_ring * SI.c
        
        # use the maximum of the two radii above
        return max(2*np.pi*bend_radius_field, bunch_train_length)

    
    def wallplug_power(self):
        energy_per_train = self.get_emitted_power() * self.get_damping_time()
        wallplug_energy_per_train = energy_per_train / self.wallplug_to_rf_efficiency
        return wallplug_energy_per_train * self.rep_rate_trains
        
    def energy_usage(self):
        return self.wallplug_power() / (self.rep_rate_trains * self.num_bunches_in_train)
        
        
    def get_length(self):
        return 0.0 # does not add to the traversed distance
