import time
import numpy as np
import scipy.constants as SI
from types import SimpleNamespace
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space
from abel.utilities.relativity import energy2gamma

class SourceBasic(Source):
    
    def __init__(self, length=0, num_particles=1000, energy=None, charge=0, rel_energy_spread=None, energy_spread=None, bunch_length=None, z_offset=0, x_offset=0, y_offset=0, emit_nx=0, emit_ny=0, beta_x=None, beta_y=None, alpha_x=0, alpha_y=0, wallplug_efficiency=1, accel_gradient=None, symmetrize=False):
        self.energy = energy
        self.charge = charge
        self.rel_energy_spread = rel_energy_spread # [eV]
        self.energy_spread = energy_spread
        self.bunch_length = bunch_length # [m]
        self.z_offset = z_offset # [m]
        self.x_offset = x_offset # [m]
        self.y_offset = y_offset # [m]
        self.num_particles = num_particles
        self.emit_nx = emit_nx # [m rad]
        self.emit_ny = emit_ny # [m rad]
        self.beta_x = beta_x # [m]
        self.beta_y = beta_y # [m]
        self.alpha_x = alpha_x # [m]
        self.alpha_y = alpha_y # [m]
        self.length = length # [m]
        self.wallplug_efficiency = wallplug_efficiency
        self.accel_gradient = accel_gradient
        self.symmetrize = symmetrize
        
        self.jitter = SimpleNamespace()
        self.jitter.x = 0
        self.jitter.y = 0
        self.jitter.z = 0
        self.jitter.t = 0
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.energy)

        # horizontal and vertical phase spaces
        xs, xps = generate_trace_space(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.num_particles, symmetrize=self.symmetrize)
        ys, yps = generate_trace_space(self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, symmetrize=self.symmetrize)
        
        # add transverse jitters and offsets
        xs += np.random.normal(scale = self.jitter.x) + self.x_offset
        ys += np.random.normal(scale = self.jitter.y) + self.y_offset
        
        # generate relative/absolute energy spreads
        if self.rel_energy_spread is not None:
            if self.energy_spread is None:
                self.energy_spread = self.energy * self.rel_energy_spread
            elif abs(self.energy_spread - self.energy * self.rel_energy_spread) > 0:
                raise Exception("Both absolute and relative energy spread defined.")
        
        # add longitudinal jitter
        if abs(self.jitter.t) > 0:
            z_jitter = np.random.normal(scale = self.jitter.t*SI.c)
        else:
            z_jitter = np.random.normal(scale = self.jitter.z)
        
        # longitudinal phase space
        if self.symmetrize:
            zs = np.tile(np.random.normal(loc = self.z_offset + z_jitter, scale = self.bunch_length, size=round(self.num_particles/4)), 4)
            Es = np.tile(np.random.normal(loc = self.energy, scale = self.energy_spread, size=round(self.num_particles/4)), 4)
        else:
            zs = np.random.normal(loc = self.z_offset + z_jitter, scale = self.bunch_length, size=self.num_particles)
            Es = np.random.normal(loc = self.energy, scale = self.energy_spread, size=self.num_particles)
        
        # create phase space
        beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.charge)
        
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
    