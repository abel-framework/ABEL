import time
import numpy as np
import scipy.constants as SI
from types import SimpleNamespace
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space_xy
from abel.utilities.relativity import energy2gamma

class SourceFromFile(Source):
    
    def __init__(self, length=0, accel_gradient=None, file=None, wallplug_efficiency=1):
        
        self.length = length # [m]
        self.accel_gradient = accel_gradient
        self.wallplug_efficiency = wallplug_efficiency
        self.file = file
        
        self.energy = None
        self.charge = None
        
        self.jitter = SimpleNamespace()
        self.jitter.x = 0
        self.jitter.y = 0
        self.jitter.z = 0
        self.jitter.t = 0
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam.load(self.file)


        self.energy = beam.energy()
        self.charge = beam.charge()

        
        
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
    