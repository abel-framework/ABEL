from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI
from abel.utilities.beam_physics import generate_trace_space
from abel.utilities.relativity import energy2gamma

class Source(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, length=0, charge=None, energy=None, accel_gradient=None, wallplug_efficiency=1, x_offset=0, y_offset=0, x_angle=0, y_angle=0, norm_jitter_emittance_x=None, norm_jitter_emittance_y=None, waist_shift_x=0, waist_shift_y=0, rep_rate_trains=None, num_bunches_in_train=None, bunch_separation=None):

        super().__init__(num_bunches_in_train=num_bunches_in_train, bunch_separation=bunch_separation, rep_rate_trains=rep_rate_trains)
        
        self.length = length
        self.energy = energy
        self.charge = charge
        self.accel_gradient = accel_gradient
        self.wallplug_efficiency = wallplug_efficiency
        
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_angle = x_angle
        self.y_angle = y_angle

        self.norm_jitter_emittance_x = norm_jitter_emittance_x
        self.norm_jitter_emittance_y = norm_jitter_emittance_y
        
        self.waist_shift_x = waist_shift_x
        self.waist_shift_y = waist_shift_y
        
        self.jitter = SimpleNamespace()
        self.jitter.x = 0
        self.jitter.y = 0
        self.jitter.z = 0
        self.jitter.t = 0
        self.jitter.xp = 0
        self.jitter.yp = 0
        self.jitter.E = 0

        self.is_polarized = False
    
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # add offsets and angles and jitter (horizontal)
        if self.norm_jitter_emittance_x is not None:
            x_jitter, xp_jitter = generate_trace_space(self.norm_jitter_emittance_x/beam.gamma(), beam.beta_x(), beam.alpha_x(), 1)         
        else:
            x_jitter, xp_jitter = np.random.normal(scale=self.jitter.x), np.random.normal(scale=self.jitter.xp)
        beam.set_xs(beam.xs() + self.x_offset + x_jitter)
        beam.set_xps(beam.xps() + self.x_angle + xp_jitter)
        
        if self.norm_jitter_emittance_y is not None:
            y_jitter, yp_jitter = generate_trace_space(self.norm_jitter_emittance_y/beam.gamma(), beam.beta_y(), beam.alpha_y(), 1)
        else:
            y_jitter, yp_jitter = np.random.normal(scale=self.jitter.y), np.random.normal(scale=self.jitter.yp)
        beam.set_ys(beam.ys() + self.y_offset + y_jitter)
        beam.set_yps(beam.yps() + self.y_angle + yp_jitter)

        # shift the waist location
        beam.set_xs(beam.xs()-self.waist_shift_x*beam.xps())
        beam.set_ys(beam.ys()-self.waist_shift_y*beam.yps())
        
        # add longitudinal and energy jitter
        if abs(self.jitter.t) > 0:
            self.jitter.z = self.jitter.t*SI.c
        beam.set_zs(beam.zs() + np.random.normal(scale=self.jitter.z))
        beam.set_Es(beam.Es() + np.random.normal(scale=self.jitter.E))

        # set the bunch train pattern
        if self.num_bunches_in_train is not None:
            beam.num_bunches_in_train = self.num_bunches_in_train
        if self.bunch_separation is not None:
            beam.bunch_separation = self.bunch_separation
        
        # set metadata
        beam.location = 0
        beam.stage_number = 0
        beam.trackable_number = -1
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_length(self):
        if self.accel_gradient is not None:
            return self.energy/self.accel_gradient
        else:
            return self.length

    
    def get_cost_breakdown(self):
        if self.is_polarized:
            if self.charge < 0:
                return ('Polarized electron source', CostModeled.cost_per_source_polarized_electrons)
            else:
                return ('Polarized positron source', CostModeled.cost_per_source_polarized_positrons)
        else:
            return ('Source', CostModeled.cost_per_source)
        
        
    def get_energy(self):
        return self.energy

    def get_nom_energy(self):
        return self.get_energy()
    
    def energy_efficiency(self):
        return self.wallplug_efficiency
    
    def get_charge(self):
        return self.charge
    
    def get_average_beam_current(self):
        if self.get_rep_rate_average() is not None:
            return self.get_charge() * self.get_rep_rate_average()
    
    def energy_usage(self):
        return self.get_energy()*abs(self.get_charge())/self.energy_efficiency()

    def wallplug_power(self):
        if self.rep_rate is not None:
            return self.energy_usage() * self.rep_rate
    
    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'Source'
        color = 'black'
        return x_points, y_points, final_angle, label, color
        
    