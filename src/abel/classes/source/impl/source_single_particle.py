import time
import numpy as np
import scipy.constants as SI
from types import SimpleNamespace
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space_xy
from abel.utilities.relativity import energy2gamma
from abel import Trackable

class SourceSingleParticle(Source):
    
    def __init__(self, length=0, energy=None, charge=-SI.e, z=0, x=0, y=0, xp=0, yp=0, wallplug_efficiency=1, accel_gradient=None):
        self.energy = energy
        self.charge = charge
        self.z_offset = z # [m]
        self.x_offset = x # [m]
        self.y_offset = y # [m]
        self.x_angle = xp
        self.y_angle = yp
        self.length = length # [m]
        
        self.wallplug_efficiency = wallplug_efficiency
        self.accel_gradient = accel_gradient
        
        self.jitter = SimpleNamespace()
        self.jitter.x = 0
        self.jitter.y = 0
        self.jitter.z = 0
        self.jitter.t = 0
        self.jitter.xp = 0
        self.jitter.yp = 0
        self.jitter.E = 0 

        self.waist_shift_x = 0
        self.waist_shift_y = 0
        
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.energy)
        
        # create phase space
        beam.set_phase_space(xs=np.array([self.x_offset]), ys=np.array([self.y_offset]), zs=np.array([self.z_offset]), xps=np.array([self.x_angle]), yps=np.array([self.y_angle]), Es=self.energy, Q=self.charge)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def get_length(self):
        if self.accel_gradient is not None:
            return self.energy/self.accel_gradient
        else:
            return self.length
    
    def get_charge(self):
        return self.charge
    
    def get_energy(self):
        return self.energy
    
    def energy_efficiency(self):
        return self.wallplug_efficiency
    