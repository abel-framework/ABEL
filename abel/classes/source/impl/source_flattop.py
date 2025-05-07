import numpy as np
import scipy.constants as SI
from abel.classes.source.source import Source

class SourceFlatTop(Source):
    
    def __init__(self, length=0, num_particles=1000, energy=None, charge=0, rel_energy_spread=None, energy_spread=None, bunch_length=None, z_offset=0, x_offset=0, y_offset=0, x_angle=0, y_angle=0, emit_nx=0, emit_ny=0, beta_x=None, beta_y=None, alpha_x=0, alpha_y=0, angular_momentum=0, wallplug_efficiency=1, accel_gradient=None, symmetrize=False):
        
        self.energy_spread = energy_spread # [eV]
        self.rel_energy_spread = rel_energy_spread # [eV]
        self.bunch_length = bunch_length # [m]
        self.z_offset = z_offset # [m]
        self.num_particles = num_particles
        self.emit_nx = emit_nx # [m rad]
        self.emit_ny = emit_ny # [m rad]
        self.beta_x = beta_x # [m]
        self.beta_y = beta_y # [m]
        self.alpha_x = alpha_x # [m]
        self.alpha_y = alpha_y # [m]
        self.angular_momentum = angular_momentum
        self.symmetrize = symmetrize
        
        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle)

    
    def track(self, _ = None, savedepth=0, runnable=None, verbose=False):

        from abel.classes.beam import Beam
        from abel.utilities.beam_physics import generate_trace_space_xy
        from abel.utilities.relativity import energy2gamma
        
        # make empty beam
        beam = Beam()
        
        # horizontal and vertical phase spaces
        gamma = energy2gamma(self.energy)
        xs, xps, ys, yps = generate_trace_space_xy(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, self.angular_momentum/gamma, symmetrize=self.symmetrize)

        # generate relative/absolute energy spreads
        if self.rel_energy_spread is not None:
            if self.energy_spread is None:
                self.energy_spread = self.energy * self.rel_energy_spread
            elif abs(self.energy_spread - self.energy * self.rel_energy_spread) > 0:
                raise Exception("Both absolute and relative energy spread defined.")

        if self.symmetrize:
            num_tiling = 4
            num_particles_actual = round(self.num_particles/num_tiling)
        else:
            num_particles_actual = self.num_particles
        
        # energies
        Es = np.random.normal(loc=self.energy, scale=self.energy_spread, size=num_particles_actual)

        # longitudinal positions
        zs = np.random.uniform(low=self.z_offset-self.bunch_length, high=self.z_offset, size=num_particles_actual)

        weightings = np.ones(zs.shape)*self.charge/(SI.e*self.num_particles)
        
        # symmetrize
        if self.symmetrize:
            zs = np.tile(zs, num_tiling)
            Es = np.tile(Es, num_tiling)
            weightings = np.tile(weightings, num_tiling)


        # create phase space
        beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.charge, weightings=weightings)

        # add jitters and offsets in super function
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

