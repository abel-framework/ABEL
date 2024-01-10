import time
import numpy as np
import scipy.constants as SI
from types import SimpleNamespace
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space_xy
from abel.utilities.relativity import energy2gamma

class SourceSingleParticle(Source):
    
    def __init__(self, length=0, energy=None, charge=-SI.e, z=0, x=0, y=0, xp=0, yp=0, wallplug_efficiency=1, accel_gradient=None):
        self.energy = energy
        self.charge = charge
        self.z = z # [m]
        self.x = x # [m]
        self.y = y # [m]
        self.xp = xp
        self.yp = yp
        self.length = length # [m]
        self.wallplug_efficiency = wallplug_efficiency
        self.accel_gradient = accel_gradient
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.energy)
        
        # create phase space
        beam.set_phase_space(xs=np.array([self.x]), ys=np.array([self.y]), zs=np.array([self.z]), xps=np.array([self.xp]), yps=np.array([self.yp]), Es=self.energy, Q=self.charge)
        
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
    