# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np
import scipy.constants as SI
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space_xy
from abel.utilities.relativity import energy2gamma

class SourceTrapezoid(Source):
    """
    Beam particle source implementation that generates a longitudinally 
    trapezoidal, transversely Gaussian particle distribution.

    This class extends ``Source`` and models bunches with an adjustable ratio 
    of trapezoidal-to-uniform charge regions. The resulting longitudinal 
    distribution can also be smoothed with a Gaussian blur and optionally made 
    front-heavy (i.e. denser at the bunch head).

    Inherits all attributes from ``Source``.

    Attributes
    ----------
    num_particles : int
        Number of macro-particles to sample.
        
    rel_energy_spread : float
        Relative energy spread, defined as the ratio of the standard deviation 
        of the energy distribution to the mean energy. If provided, 
        ``energy_spread`` will be set to ``energy * rel_energy_spread`` during 
        ``track()`` (overriding any previously-set absolute ``energy_spread``).
        
    energy_spread : [eV] float
        Absolute energy spread (standard deviation). If ``rel_energy_spread`` is 
        supplied, this value is overwritten to ``energy * rel_energy_spread`` 
        when ``track()`` is called.

    bunch_length : [m] float
        Longitudinal bunch length used as the standard deviation for
        Gaussian sampling of z-positions.

    gaussian_blur : [m] float
        Gaussian blur applied to the front and back of the z-distribution to 
        smooth the transition to zero.

    current_head : [A] float
        Current at the head of the bunch. Used to define the trapezoidal 
        shape by controlling the uniform-to-triangular charge ratio.

    z_offset : [m] float
        Mean longitudinal offset for the sampled z-positions.

    emit_nx, emit_ny : [m rad] float
        Normalized transverse emittances.

    beta_x, beta_y : [m] float
        Twiss beta functions.

    alpha_x, alpha_y : float
        Twiss alpha functions.

    angular_momentum : [m rad] float
        Normalized angular momentum.

    symmetrize : bool
        If ``True`` and ``symmetrize_6d`` is ``False``, the generated particle 
        distribution is transversely symmetrised in x, y, x' and y'.

    front_heavy_distribution : bool
        If ``True``, redistributes the z-distribution to favor the bunch head
        (lower z).
    """
    
    def __init__(self, length=0, num_particles=1000, energy=None, charge=0, rel_energy_spread=None, energy_spread=None, bunch_length=None, gaussian_blur=0, current_head=0, z_offset=0, x_offset=0, y_offset=0, x_angle=0, y_angle=0, emit_nx=0, emit_ny=0, beta_x=None, beta_y=None, alpha_x=0, alpha_y=0, angular_momentum=0, wallplug_efficiency=1, accel_gradient=None, symmetrize=False, front_heavy_distribution=False):
        
        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle)
        
        self.energy_spread = energy_spread # [eV]
        self.rel_energy_spread = rel_energy_spread # [eV]
        self.current_head = current_head # [A]
        self.bunch_length = bunch_length # [m]
        self.gaussian_blur = gaussian_blur # [m]
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
        self.front_heavy_distribution = front_heavy_distribution
        
    
    def track(self, _ = None, savedepth=0, runnable=None, verbose=False):
        """
        Generate a ``Beam`` object.
        """
        
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
           
        # longitudinal positions
        Q_uniform = abs(self.current_head) * self.bunch_length / SI.c
        if Q_uniform > 2*abs(self.charge):
            Q_triangle = abs(self.charge)
            Q_uniform = 0
            zmode = self.z_offset
        elif abs(self.charge) > Q_uniform:
            Q_triangle = abs(self.charge) - Q_uniform
            zmode = self.z_offset - self.bunch_length
        else:
            Q_triangle = Q_uniform - abs(self.charge)
            Q_uniform = abs(self.charge) - Q_triangle
            zmode = self.z_offset
        
        if self.symmetrize:
            num_tiling = 4
            num_particles_actual = round(self.num_particles/num_tiling)
        else:
            num_particles_actual = self.num_particles
            
        # construct shape
        index_split = round(num_particles_actual*abs(Q_uniform)/abs(self.charge))
        inds = np.random.permutation(num_particles_actual)
        mask_uniform = inds[0:index_split]
        mask_triangle = inds[index_split:num_particles_actual]
        zs = np.zeros(num_particles_actual)
        zs[mask_uniform] = np.random.uniform(low=self.z_offset-self.bunch_length, high=self.z_offset, size=len(mask_uniform))
        zs[mask_triangle] = np.random.triangular(left=self.z_offset-self.bunch_length, right=self.z_offset, mode=zmode, size=len(mask_triangle))

        # add Gaussian blur/convolve with Gaussian
        zs = zs + np.random.normal(scale=self.gaussian_blur, size=len(zs))
        
        # make particle distribution front-heavy (more particles at the bunch head)
        if self.front_heavy_distribution:

            # normalize distance within bunch
            zmax = zs.max()
            zmin = zs.min()
            zs_norm = -(zs-zmax)/(zmax-zmin)

            # shift normalized particles
            front_heaviness = 20
            zs_norm_redist = (1+1/front_heaviness)/((1-zs_norm)*front_heaviness+1)-1/front_heaviness
            zs = zmax - (zmax-zmin)*zs_norm_redist

            # recalculate weightings (lower where there are more particles)
            weightings = np.ones(zs.shape)*np.abs(self.charge)/( SI.e * self.num_particles )
            weightings[mask_uniform] = (1+front_heaviness*zs_norm_redist[mask_uniform])**2
            weightings[mask_uniform] = weightings[mask_uniform]/(np.sum(weightings[mask_uniform])/(np.abs(self.charge)/SI.e) )*np.sum(mask_uniform)/len(zs)
            weightings[mask_triangle] = (1+front_heaviness*zs_norm_redist[mask_triangle])**3
            weightings[mask_triangle] = weightings[mask_triangle]/(np.sum(weightings[mask_triangle])/(np.abs(self.charge)/SI.e) )*np.sum(mask_triangle)/len(zs)
            
        else:
            weightings = np.ones(zs.shape)*np.abs(self.charge)/(SI.e * self.num_particles)
        
        # energies
        Es = np.random.normal(loc=self.energy, scale=self.energy_spread, size=num_particles_actual)
        
        # symmetrize
        if self.symmetrize:
            zs = np.tile(zs, num_tiling)
            Es = np.tile(Es, num_tiling)
            weightings = np.tile(weightings, num_tiling)
        
        # create phase space
        beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=self.charge, weightings=weightings)

        # add jitters and offsets in super function
        return super().track(beam, savedepth, runnable, verbose)
    

    def print_summary(self):
        """
        Print a summary for the source.
        """
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
        print('Gaussian blur [um]: ', self.gaussian_blur*1e6)
        print('Current head [A]: ', self.current_head)
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
        print('Symmetrisation: ', self.symmetrize)