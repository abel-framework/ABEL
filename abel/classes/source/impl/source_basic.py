import numpy as np
import scipy.constants as SI
from abel.classes.beam import Beam
from abel.classes.source.source import Source
from abel.utilities.beam_physics import generate_trace_space_xy, generate_symm_trace_space_xyz
from abel.utilities.relativity import energy2gamma

class SourceBasic(Source):
    
    def __init__(self, length=0, num_particles=1000, energy=None, charge=0, rel_energy_spread=None, energy_spread=None, bunch_length=None, z_offset=0, x_offset=0, y_offset=0, x_angle=0, y_angle=0, emit_nx=0, emit_ny=0, beta_x=None, beta_y=None, alpha_x=0, alpha_y=0, angular_momentum=0, wallplug_efficiency=1, accel_gradient=None, symmetrize=False, symmetrize_6d=False, z_cutoff=None):
        
        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle)
        
        self.rel_energy_spread = rel_energy_spread # [eV]
        self.energy_spread = energy_spread
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
        self.symmetrize_6d = symmetrize_6d
        self.z_cutoff = z_cutoff
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        rng = np.random.default_rng()
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.energy)

        # generate relative/absolute energy spreads
        if self.rel_energy_spread is not None:
            if self.energy_spread is None:
                self.energy_spread = self.energy * self.rel_energy_spread
            elif abs(self.energy_spread - self.energy * self.rel_energy_spread) > 0:
                raise Exception("Both absolute and relative energy spread defined.")

        if self.symmetrize_6d is False:
            
            # horizontal and vertical phase spaces
            xs, xps, ys, yps = generate_trace_space_xy(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, rng, self.angular_momentum/gamma, symmetrize=self.symmetrize)
            
            # longitudinal phase space
            if self.symmetrize:
                num_tiling = 4
                num_particles_actual = round(self.num_particles/num_tiling)
            else:
                num_particles_actual = self.num_particles
            zs = rng.normal(loc=self.z_offset, scale=self.bunch_length, size=num_particles_actual)
            Es = rng.normal(loc=self.energy, scale=self.energy_spread, size=num_particles_actual)
            if self.symmetrize:
                zs = np.tile(zs, num_tiling)
                Es = np.tile(Es, num_tiling)

        else:
            xs, xps, ys, yps, zs, Es = generate_symm_trace_space_xyz(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, self.bunch_length, self.energy_spread, rng, self.angular_momentum/gamma)
            
            # add longitudinal offsets
            zs += self.z_offset
            Es += self.energy
        
        # create phase space
        beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.charge)

        # Apply filter(s) if desired
        if self.z_cutoff is not None:
            beam = self.z_filter(beam)

        # add jitters and offsets in super function
        return super().track(beam, savedepth, runnable, verbose)


    # ==================================================
    # Filter out particles whose z < z_cutoff for testing instability etc.
    def z_filter(self, beam):
        xs = beam.xs()
        ys = beam.ys()
        zs = beam.zs()
        pxs = beam.pxs()
        pys = beam.pys()
        pzs = beam.pzs()
        weights = beam.weightings()

        # Apply the filter
        bool_indices = (zs > self.z_cutoff)
        zs_filtered = zs[bool_indices]
        xs_filtered = xs[bool_indices]
        ys_filtered = ys[bool_indices]
        pxs_filtered = pxs[bool_indices]
        pys_filtered = pys[bool_indices]
        pzs_filtered = pzs[bool_indices]
        weights_filtered = weights[bool_indices]

        # Initialise ABEL Beam object
        beam_out = Beam()
        
        # Set the phase space of the ABEL beam
        beam_out.set_phase_space(Q=np.sum(weights_filtered)*-SI.e,
                             xs=xs_filtered,
                             ys=ys_filtered,
                             zs=zs_filtered, 
                             pxs=pxs_filtered,  # Always use single particle momenta?
                             pys=pys_filtered,
                             pzs=pzs_filtered)

        return beam_out
    