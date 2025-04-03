"""
Module: spectrometer_achromatic_basic

This module defines the `SpectrometerAchromaticImpactX` class, which is a
subclass of `SpectrometerAchromatic`. It provides methods to initialize
the spectrometer and track the beam.

Classes
-------
SpectrometerAchromaticImpactX : SpectrometerAchromatic
    A basic implementation of an achromatic spectrometer.

Imports
-------
from abel import SpectrometerAchromatic, PlasmaLensNonlinearThin
import scipy.constants as SI
import numpy as np
"""

from abel import SpectrometerAchromatic, PlasmaLensNonlinearThin
from abel.apis.impactx.impactx_api import run_impactx
import scipy.constants as SI
import numpy as np

class SpectrometerAchromaticImpactX(SpectrometerAchromatic):
    """
    A basic implementation of an achromatic spectrometer.

    Parameters
    ----------
    length_drift_to_dipole : float, optional
        Length of the drift space to the dipole magnet [m]. Default is 0.2.

    field_dipole : float, optional
        Magnetic field strength of the dipole magnet [T]. Negative values
        bend the beam downwards. Default is -0.85.

    length_dipole : float, optional
        Length of the dipole magnet [m]. Default is 0.15.

    length_drift_dipole_to_lens : float, optional
        Length of the drift space from the dipole magnet to the lens [m].
        Default is 0.15.

    length_plasma_lens : float, optional
        Length of the plasma lens [m]. Default is 20E-3.

    radius_plasma_lens : float, optional
        Radius of the plasma lens [m]. Default is 1E-3.

    length_drift_lens_to_screen : float, optional
        Length of the drift space from the lens to the screen [m]. Default
        is 0.4.

    imaging_energy : float or None, optional
        Imaging energy [eV]. Default is None.

    disable_lens_nonlinearity : bool, optional
        Flag to disable lens nonlinearity. Default is True.
    """

    def __init__(self,
                 length_drift_to_dipole=0.2,
                 field_dipole=-0.85,
                 length_dipole=0.15,
                 length_drift_dipole_to_lens=0.15,
                 length_plasma_lens=20E-3,
                 radius_plasma_lens=1E-3,
                 length_drift_lens_to_screen=0.4,
                 imaging_energy=None,
                 disable_lens_nonlinearity=True,
                 enable_space_charge=False,
                 enable_csr=False,
                 enable_isr=False):
        """
        Initializes the `SpectrometerAchromaticImpactX` with the given
        parameters.

        Parameters
        ----------
        length_drift_to_dipole : float, optional
            Length of the drift space to the dipole magnet [m]. Default is
            0.2.

        field_dipole : float, optional
            Magnetic field strength of the dipole magnet [T]. Negative
            values bend the beam downwards. Default is -0.85.

        length_dipole : float, optional
            Length of the dipole magnet [m]. Default is 0.15.

        length_drift_dipole_to_lens : float, optional
            Length of the drift space from the dipole magnet to the lens
            [m]. Default is 0.15.

        length_plasma_lens : float, optional
            Length of the plasma lens [m]. Default is 20E-3.

        radius_plasma_lens : float, optional
            Radius of the plasma lens [m]. Default is 1E-3.

        length_drift_lens_to_screen : float, optional
            Length of the drift space from the lens to the screen [m].
            Default is 0.4.

        imaging_energy : float or None, optional
            Imaging energy [eV]. Default is None.

        disable_lens_nonlinearity : bool, optional
            Flag to disable lens nonlinearity. Default is True.
        """

        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr

        super().__init__(length_drift_to_dipole=length_drift_to_dipole,
                         field_dipole=field_dipole,
                         length_dipole=length_dipole,
                         length_drift_dipole_to_lens=length_drift_dipole_to_lens,
                         length_plasma_lens=length_plasma_lens,
                         radius_plasma_lens=radius_plasma_lens,
                         length_drift_lens_to_screen=length_drift_lens_to_screen,
                         imaging_energy=imaging_energy,
                         disable_lens_nonlinearity=disable_lens_nonlinearity)

    def get_lattice(self):

        import impactx
        
        # initialize lattice
        lattice = []

        ns = 25  # number of slices per ds in the element

        drift_source_to_dipole = impactx.elements.ExactDrift(name="drift_source_to_dipole", ds=self.length_drift_to_dipole, nslice=ns)
        
        # define dipole
        E0 = self.nom_energy
        gamma0 = E0 * SI.e / (SI.m_e * SI.c**2)
        v0 = SI.c * np.sqrt(1 - 1 / gamma0**2)
        p0 = SI.m_e * gamma0 * v0
        Dx = self.dipole_length**2 * self.dipole_field * SI.e / (2 * p0)
        phi = self.field_dipole * self.length_dipole * SI.e / p0
        ns = 25  # number of slices per ds in the element

        # phi=np.rad2deg(phi)
        bend = impactx.elements.ExactSbend(name="dipole", ds=self.length_dipole, phi=0, B=self.field_dipole, nslice=ns)

        drift_dipole_to_lens = impactx.elements.ExactDrift(name="drift_dipole_to_lens", ds=self.length_drift_dipole_to_lens, nslice=ns)
        
        # calculate the focal length
        length_object_to_lens = (self.length_drift_to_dipole
                                 + self.length_dipole
                                 + self.length_drift_dipole_to_lens
                                 + self.length_plasma_lens/2)
        length_lens_to_screen = (self.length_plasma_lens/2
                                 + self.length_drift_lens_to_screen)
        focal_length = 1 / (1 / length_object_to_lens
                            + 1 / length_lens_to_screen)
        
        # define the nonlinearity/taper of the lens
        if not self.disable_lens_nonlinearity:
            dtaper = - self.radius_plasma_lens / Dx
        else:
            dtaper = 0.0

        # define plasma lens
        plasma_lens = impactx.elements.TaperedPL(k=1/focal_length, taper=dtaper, name="plasmalens")

        drift_lens_to_screen = impactx.elements.ExactDrift(name="drift_lens_to_screen", ds=self.length_drift_lens_to_screen, nslice=ns)

        # add beam diagnostics
        monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

        # specify the lattice sequence
        lattice.append(monitor)
        lattice.append(drift_source_to_dipole)
        lattice.append(monitor)
        lattice.append(bend)
        lattice.append(monitor)
        lattice.extend(drift_dipole_to_lens)
        lattice.append(monitor)
        lattice.extend(plasma_lens)
        lattice.append(monitor)
        lattice.extend(drift_lens_to_screen)
        lattice.append(monitor)

        return lattice


    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        """
        Tracks the beam through the spectrometer.

        Parameters
        ----------
        beam0 : Beam
            The initial beam to be tracked.

        savedepth : int, optional
            The depth at which to save the beam state. Default is 0.

        runnable : callable, optional
            A callable to be run during tracking. Default is None.

        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        Beam
            The tracked beam object.
        """

        # get lattice
        lattice = self.get_lattice()
        
        # run ImpactX
        beam, evol = run_impactx(lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)

        # as impactX does the dipole bending and the nonlinearity in the plasma lens
        # only in the horizontal axis, the two axis have to be swapped around
        # as in this spectrometer the trajectory is bend downwards
        xs, uxs = beam.xs(), beam.uxs()
        beam.set_xs(beam.ys())
        beam.set_uxs(beam.uys())
        beam.set_ys(xs)
        beam.set_uys(uxs)

        # swap values in evol
        self.__swap_around_axis_evol(evol)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)

    def __swap_around_axis_evol(self, evol):

        print('__swap_around_axis_evol is untested')

        # HINT: Swapping of second order data is not implemented

        to_swap = ['emittance_{}n', 'beta_{}', 'sig_{}', '{}_mean', 'dispersion_{}']

        for key in to_swap:

            tempx = evol[key.format('x')]
            evol[key.format('x')] = evol[key.format('y')]
            evol[key.format('y')] = tempx

        return evol

    def get_bending_angle(self, energy):
        """
        Calculates the bending angle of the spectrometer for a given energy.

        Parameters
        ----------
        energy : float
            The energy of the particle beam [eV].

        Returns
        -------
        float
            The calculated bending angle [rad].
        """

        raise NotImplementedError

        return bend_angle

    def get_dispersion(self, energy=None, bend_angle=None, length=None):
        """
        Calculates the dispersion of the spectrometer for a given energy or
        bending angle.

        Parameters
        ----------
        energy : float, optional
            The energy of the particle beam [eV]. Either `energy` or
            `bend_angle` must be provided.

        bend_angle : float, optional
            Bending angle of the dipole [rad]. Either `bend_angle` or
            `energy` must be provided.

        length : float, optional
            The distance to use for the calculation [m]. If not provided,
            it defaults to the distance from the center of the dipole to
            the screen.

        Returns
        -------
        float
            The calculated dispersion value [m].
        """

        raise NotImplementedError

        return dispersion
