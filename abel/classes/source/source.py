from abc import abstractmethod
from matplotlib import patches
from abel import Trackable
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class Source(Trackable):
    
    @abstractmethod
    def __init__(self, length=0, charge=None, energy=None, accel_gradient=None, wallplug_efficiency=1, x_offset=0, y_offset=0, x_angle=0, y_angle=0, waist_shift_x=0, waist_shift_y=0):
        
        self.length = length
        self.energy = energy
        self.charge = charge
        self.accel_gradient = accel_gradient
        self.wallplug_efficiency = wallplug_efficiency
        
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_angle = x_angle
        self.y_angle = y_angle
        
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
    
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # add offsets and angles
        beam.set_xs(beam.xs() + np.random.normal(loc=self.x_offset, scale=self.jitter.x))
        beam.set_ys(beam.ys() + np.random.normal(loc=self.y_offset, scale=self.jitter.y))
        beam.set_xps(beam.xps() + np.random.normal(loc=self.x_angle, scale=self.jitter.xp))
        beam.set_yps(beam.yps() + np.random.normal(loc=self.y_angle, scale=self.jitter.yp))

        # shift the waist location
        beam.set_xs(beam.xs()-self.waist_shift_x*beam.xps())
        beam.set_ys(beam.ys()-self.waist_shift_y*beam.yps())
        
        # add longitudinal and energy jitter
        if abs(self.jitter.t) > 0:
            self.jitter.z = self.jitter.t*SI.c
        beam.set_zs(beam.zs() + np.random.normal(scale=self.jitter.z))
        beam.set_Es(beam.Es() + np.random.normal(scale=self.jitter.E))

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
    
    def get_energy(self):
        return self.energy
        
    def energy_efficiency(self):
        return self.wallplug_efficiency
    
    def get_charge(self):
        return self.charge
    
    def energy_usage(self):
        return self.get_energy()*abs(self.get_charge())/self.energy_efficiency()
    
    def survey_object(self):
        rect = patches.Rectangle((0, -0.5), self.get_length(), 1)
        return rect
    