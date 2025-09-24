# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

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
        
        # make empty beam
        beam = Beam()
        
        # Lorentz gamma
        gamma = energy2gamma(self.energy)

        # generate relative/absolute energy spreads
        if self.rel_energy_spread is not None:
            if self.energy_spread is None:
                self.energy_spread = self.energy * self.rel_energy_spread
            elif abs(self.energy_spread - self.energy * self.rel_energy_spread) > 0:
                self.energy_spread = self.energy * self.rel_energy_spread
                #raise Exception("Both absolute and relative energy spread defined")

        if self.symmetrize_6d is False:
            
            # horizontal and vertical phase spaces
            xs, xps, ys, yps = generate_trace_space_xy(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, self.angular_momentum/gamma, symmetrize=self.symmetrize)
            
            # longitudinal phase space
            if self.symmetrize:
                num_tiling = 4
                num_particles_actual = round(self.num_particles/num_tiling)
            else:
                num_particles_actual = self.num_particles
            zs = np.random.normal(loc=self.z_offset, scale=self.bunch_length, size=num_particles_actual)
            Es = np.random.normal(loc=self.energy, scale=self.energy_spread, size=num_particles_actual)
            if self.symmetrize:
                zs = np.tile(zs, num_tiling)
                Es = np.tile(Es, num_tiling)

        else:
            xs, xps, ys, yps, zs, Es = generate_symm_trace_space_xyz(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, self.bunch_length, self.energy_spread, self.angular_momentum/gamma)
            
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
        spxs = beam.spxs()
        spys = beam.spys()
        spzs = beam.spzs()
        weights = beam.weightings()

        # Apply the filter
        bool_indices = (zs > self.z_cutoff)
        zs_filtered = zs[bool_indices]
        xs_filtered = xs[bool_indices]
        ys_filtered = ys[bool_indices]
        pxs_filtered = pxs[bool_indices]
        pys_filtered = pys[bool_indices]
        pzs_filtered = pzs[bool_indices]
        spxs_filtered = spxs[bool_indices]
        spys_filtered = spys[bool_indices]
        spzs_filtered = spzs[bool_indices]
        weights_filtered = weights[bool_indices]

        # Initialise ABEL Beam object
        beam_out = Beam()
        
        # Set the phase space of the ABEL beam
        beam_out.set_phase_space(Q=np.sum(weights_filtered)*np.sign(self.charge)*SI.e,
                             xs=xs_filtered,
                             ys=ys_filtered,
                             zs=zs_filtered, 
                             pxs=pxs_filtered,  # Always use single particle momenta?
                             pys=pys_filtered,
                             pzs=pzs_filtered,
                             spxs=spxs_filtered,
                             spys=spys_filtered,
                             spzs=spzs_filtered,
                             particle_mass=beam.particle_mass
                             )

        return beam_out
    

    # ==================================================
    def print_summary(self):
        print('Type: ', type(self))
        print('Number of macro particles: ', self.num_particles)
        print('Charge [nC]: ', self.charge*1e9)
        print('Energy [GeV]: ', self.energy/1e9)
        print('Normalised x emittance [mm mrad]: ', self.emit_nx*1e6)
        print('Normalised y emittance [mm mrad]: ', self.emit_ny*1e6)
        print('x beta function [mm]: ', self.beta_x*1e3)
        print('y beta function [mm]: ', self.beta_y*1e3)
        print('Relative energy spread [%]: ', self.rel_energy_spread*100)
        print('Bunch length [um]: ', self.bunch_length*1e6)
        print('x-offset [um]: ', self.x_offset*1e6)
        print('y-offset [um]: ', self.y_offset*1e6)
        print('z-offset [um]: ', self.z_offset*1e6)
        print('x-jitter [nm]: ', self.jitter.x*1e9)
        print('y-jitter [nm]: ', self.jitter.y*1e9)
        print('t-jitter [ns]: ', self.jitter.t*1e9)
        print('Normalised x emittance jitter [mm mrad]: ', 
            self.norm_jitter_emittance_x * 1e6 if self.norm_jitter_emittance_x is not None else "None")
        print('Normalised y emittance jitter [mm mrad]: ', 
            self.norm_jitter_emittance_y * 1e6 if self.norm_jitter_emittance_y is not None else "None")
        if self.symmetrize:
            print('Symmetrisation: ', self.symmetrize)
        else:
            if self.symmetrize_6d:
                print('6D Symmetrisation: ', self.symmetrize_6d)
            else:
                print('Symmetrisation: ', self.symmetrize)