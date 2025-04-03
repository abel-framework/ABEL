import time
import numpy as np
import scipy.constants as SI
from abel import Source, Beam
from abel.utilities.beam_physics import generate_trace_space_xy, generate_symm_trace_space_xyz
from abel.utilities.relativity import energy2gamma

class SourceEnergyComb(Source):
    """
    A source class that extends the functionality of `SourceBasic` by distributing
    particle energies within a given list of energies. This is useful for creating
    comb-like energy distributions for resolution estimates.

    Parameters
    ----------
    length : float, optional
        Length of the source [m]. Default is 0.

    num_particles : int, optional
        Number of particles in the beam. Default is 1000.

    energy : float, optional
        Central energy of the beam [eV]. Default is None.

    charge : float, optional
        Total charge of the beam [C]. Default is 0.

    rel_energy_spread : float, optional
        Relative energy spread (fraction). Default is None.

    energy_spread : float, optional
        Absolute energy spread [eV]. Default is None.

    bunch_length : float, optional
        RMS bunch length [m]. Default is None.

    z_offset : float, optional
        Longitudinal offset of the beam [m]. Default is 0.

    x_offset, y_offset : float, optional
        Transverse offsets of the beam [m]. Default is 0.

    x_angle, y_angle : float, optional
        Transverse angles of the beam [rad]. Default is 0.

    emit_nx, emit_ny : float, optional
        Normalized emittances in x and y [m rad]. Default is 0.

    beta_x, beta_y : float, optional
        Beta functions in x and y [m]. Default is None.

    alpha_x, alpha_y : float, optional
        Alpha functions in x and y. Default is 0.

    angular_momentum : float, optional
        Angular momentum of the beam. Default is 0.

    wallplug_efficiency : float, optional
        Efficiency of the source. Default is 1.

    accel_gradient : float, optional
        Acceleration gradient [V/m]. Default is None.

    symmetrize : bool, optional
        Whether to symmetrize the transverse phase space. Default is False.

    symmetrize_6d : bool, optional
        Whether to symmetrize the full 6D phase space. Default is False.

    z_cutoff : float, optional
        Longitudinal cutoff for filtering particles [m]. Default is None.

    energy_comb_delta : float, optional
        Relative energy step size for creating comb-like energy distributions. Default is None.
    """

    def __init__(self, length=0, num_particles=1000, energy=None, charge=0, rel_energy_spread=None, energy_spread=None, bunch_length=None, z_offset=0, x_offset=0, y_offset=0, x_angle=0, y_angle=0, emit_nx=0, emit_ny=0, beta_x=None, beta_y=None, alpha_x=0, alpha_y=0, angular_momentum=0, wallplug_efficiency=1, accel_gradient=None, symmetrize=False, symmetrize_6d=False, z_cutoff=None, energy_comb_delta=None):

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

        # TODO: Make this an energy_diff or delta instead of a linspace
        # the diff should be in percentage
        # this will be easier for scans
        self.energy_comb_delta = energy_comb_delta


    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        """
        Tracks the beam through the source, generating its phase space and applying
        energy comb distributions or filters if specified.

        Parameters
        ----------
        _ : None, optional
            Placeholder parameter. Default is None.

        savedepth : int, optional
            Depth of saved tracking data. Default is 0.

        runnable : object, optional
            Runnable object for tracking. Default is None.

        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        Beam
            The tracked beam object.
        """

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
            xs, xps, ys, yps = generate_trace_space_xy(self.emit_nx/gamma, self.beta_x, self.alpha_x, self.emit_ny/gamma, self.beta_y, self.alpha_y, self.num_particles, self.angular_momentum/gamma, symmetrize=self.symmetrize)

            # longitudinal phase space
            if self.symmetrize:
                num_tiling = 4
                num_particles_actual = round(self.num_particles/num_tiling)
            else:
                num_particles_actual = self.num_particles
            zs = np.random.normal(loc=self.z_offset, scale=self.bunch_length, size=num_particles_actual)
            # Es = np.random.normal(loc=self.energy, scale=self.energy_spread, size=num_particles_actual)
            # this should yield a uniform energy distribution with length 2*energy_spread around self.energy
            Es = (np.random.random(size=num_particles_actual) - 0.5) * 2 * self.energy_spread + self.energy
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

        if self.energy_comb_delta is not None:
            self.reorder_particle_energies(beam)

        # add jitters and offsets in super function
        return super().track(beam, savedepth, runnable, verbose)

    def reorder_particle_energies(self, beam, plot=False):
        """
        Reorders the particle energies in the beam to create a comb-like energy
        distribution based on `self.energy_comb_delta`.

        Parameters
        ----------
        beam : Beam
            The beam object whose particle energies will be reordered.

        plot : bool, optional
            Whether to plot the longitudinal phase space after reordering. Default is False.

        Returns
        -------
        None
        """

        # takes in the beam and uses the attribute self.energy_comb_delta
        # to create a list of energies that will be given to each beam particle
        # to create a comb-like energy distribution for resolution estimates

        # the energies of all bunch particles
        vals = beam.Es()

        energy_beam = beam.energy()

        # the energy delta is given as relative, thus multiply times the average
        # particle energy
        energy_delta = self.energy_comb_delta * energy_beam

        # half of the number of steps depending on the step size and the energy
        # range of the beam particles
        num_steps_half = int((vals.max() - vals.min()) * 0.5
                             / energy_delta  + 1)

        # go half the number of steps in both directions, with the bunch
        # average energy in the middle
        energy_steps = np.arange(-num_steps_half, num_steps_half+1)
        energies_comb = energy_beam + energy_steps * energy_delta

        # find the index for each value that shows the closest value in energies
        # this function requires "left" or "right" to be specified
        # use "left"
        idx = np.searchsorted(energies_comb, vals, side="left")

        # this is required as otherwise the energies_comb[idx] in the next line
        # will throw an error
        idx[idx == len(energies_comb)] = len(energies_comb) - 1

        # find out if the value with the left-index or the value with the
        # right-index value is closer and accordingly subtract the index value
        idx = idx - (np.abs(vals - energies_comb[idx-1])
                     < np.abs(vals - energies_comb[idx]))

        # energies_comb[idx] has all the energies that were closest for each
        # particle.  Set the particle energies to this new energies
        beam.set_Es(energies_comb[idx])

        if plot:
            # print(np.diff(energies_comb)*1E-9)
            beam.plot_lps()

        return


    # ==================================================
    # Filter out particles whose z < z_cutoff for testing instability etc.
    def z_filter(self, beam):
        """
        Filters out particles whose longitudinal positions (z) are below the
        specified cutoff (`self.z_cutoff`).

        Parameters
        ----------
        beam : Beam
            The beam object to be filtered.

        Returns
        -------
        Beam
            A new beam object with particles filtered based on the z cutoff.
        """
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
