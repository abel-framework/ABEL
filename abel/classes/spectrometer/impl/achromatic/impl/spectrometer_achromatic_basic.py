"""
Module: spectrometer_achromatic_basic

This module defines the `SpectrometerAchromaticBasic` class, which is a
subclass of `SpectrometerAchromatic`. It provides methods to initialize
the spectrometer and track the beam.

Classes
-------
SpectrometerAchromaticBasic : SpectrometerAchromatic
    A basic implementation of an achromatic spectrometer.

Imports
-------
from abel import SpectrometerAchromatic, PlasmaLensNonlinearThin
import scipy.constants as SI
import numpy as np
"""

from abel import SpectrometerAchromatic, PlasmaLensNonlinearThin
import scipy.constants as SI
import numpy as np

class SpectrometerAchromaticBasic(SpectrometerAchromatic):
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
                 disable_lens_nonlinearity=True):
        """
        Initializes the `SpectrometerAchromaticBasic` with the given
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

        super().__init__(length_drift_to_dipole=length_drift_to_dipole,
                         field_dipole=field_dipole,
                         length_dipole=length_dipole,
                         length_drift_dipole_to_lens=length_drift_dipole_to_lens,
                         length_plasma_lens=length_plasma_lens,
                         radius_plasma_lens=radius_plasma_lens,
                         length_drift_lens_to_screen=length_drift_lens_to_screen,
                         imaging_energy=imaging_energy,
                         disable_lens_nonlinearity=disable_lens_nonlinearity)

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

        # transport the beam to the center of the dipole
        beam0.transport(self.length_drift_to_dipole + self.length_dipole/2)

        # this function allows to plot the trace space for debugging
        # beam0.plot_trace_space_y()

        # computing the bending angle of the dipole for each particle
        # sin(bend_angle/2) = length_dipole*0.5 / bending_radius
        # bending_radius = momentum / (B * e)
        # for small angles sin(bend_angle/2) \approx bend_angle/2
        # bend_angle = length_dipole * B * e / momentum
        # momentum [eV/c]
        bend_angles = self.get_bending_angle(beam0.Es())
        # bend_angles = self.length_dipole * self.field_dipole * SI.c / beam0.Es()

        # if self.imaging_energy is None:
        #     energy_nom = beam0.energy()
        # else:
        energy_nom = beam0.energy()

        # the bending angle for the nominal beam energy
        bend_angle_nom = self.get_bending_angle(energy_nom)
        # bend_angle_nom = self.length_dipole * self.field_dipole * SI.c / energy_nom

        # the bending angle of each particle relative to the bending of
        # the beam at nominal energy
        bend_angles_rel = bend_angles - bend_angle_nom

        # updating the angles of the particles with the dipole kick
        beam0.set_yps(beam0.yps() + bend_angles_rel)

        # transport the beam to the lens with the new angles of the particles
        beam0.transport(self.length_dipole/2 + self.length_drift_dipole_to_lens)

        # calculate the focal length
        length_object_to_lens = (self.length_drift_to_dipole
                                 + self.length_dipole
                                 + self.length_drift_dipole_to_lens
                                 + self.length_plasma_lens/2)
        length_lens_to_screen = (self.length_plasma_lens/2
                                 + self.length_drift_lens_to_screen)
        focal_length = 1 / (1 / length_object_to_lens
                            + 1 / length_lens_to_screen)

        # calculate the lens strength based on the focal length
        k = 1 / (focal_length * self.length_plasma_lens)

        if self.imaging_energy is None:
            imaging_energy = beam0.energy()
        else:
            imaging_energy = self.imaging_energy

        # calculate the magnetic field gradient with the imaging energy
        g = k / (SI.c/ imaging_energy)

        # calculate the current based on the magnetic field gradient
        current = -g * 2 * np.pi * self.radius_plasma_lens**2 / (SI.mu_0)

        # calculate the dispersion introduced by the dipole at the position of
        # the plasma lens
        length_middle_dipole_to_lens = (self.length_dipole/2
                                        + self.length_drift_dipole_to_lens
                                        + self.length_plasma_lens/2)
        dispersion = self.get_dispersion(bend_angle=bend_angle_nom,
                                         length=length_middle_dipole_to_lens)

        # calculate the nonlinearity of the plasma lens, which is
        # defined by R/Dy (negative sign found by trial and error)
        if not self.disable_lens_nonlinearity:
            rel_nonlinearity = - self.radius_plasma_lens / dispersion
        else:
            rel_nonlinearity = 0.0

        # use the already implemented PlasmaLensNonlinearThin class to calculate
        # the kick produced by the plasma lens. The kicks are updated in the
        # class beam
        # nonlinearity_in_x=False as the dipole here disperses vertically
        plasma_lens = PlasmaLensNonlinearThin(length=self.length_plasma_lens,
                                              radius=self.radius_plasma_lens,
                                              current=current,
                                              rel_nonlinearity=rel_nonlinearity,
                                              nonlinearity_in_x=False)

        beam = plasma_lens.track(beam=beam0)

        # transport the beam with the updated angles
        length_middle_lens_to_screen = (self.length_plasma_lens/2
                                        + self.length_drift_lens_to_screen)
        beam.transport(length_middle_lens_to_screen)

        # shift beam by the dispersion (special case for spectrometers)
        # this will allow to plot the beam as if there was a screen
        length_middle_dipole_to_screen = (self.length_dipole / 2
                                           + self.length_drift_dipole_to_lens
                                           + self.length_plasma_lens
                                           + self.length_drift_lens_to_screen)
        # dispersion_on_screen = bend_angle_nom * length_middle_dipole_to_screen
        dispersion_on_screen = self.get_dispersion(bend_angle=bend_angle_nom,
                                                   length= length_middle_dipole_to_screen)
        beam.set_ys(beam.ys() + dispersion_on_screen)

        return super().track(beam, savedepth, runnable, verbose)

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

        # computing the bending angle of the dipole for each particle
        # sin(bend_angle/2) = length_dipole*0.5 / bending_radius
        # bending_radius = momentum / (B * e)
        # for small angles sin(bend_angle/2) \approx bend_angle/2
        # bend_angle = length_dipole * B * e / momentum
        # momentum [eV/c]
        bend_angle = self.length_dipole * self.field_dipole * SI.c / energy

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

        text = 'Please give either the bending angle *or* the energy.'
        assert ((bend_angle is None and energy is not None)
                or (bend_angle is not None and energy is None)), text

        if bend_angle is None:
            bend_angle = self.get_bending_angle(energy)

        if length is None:
            # assume lenth from center of dipole to the screen
            length = (self.length_dipole / 2
                      + self.length_drift_dipole_to_lens
                      + self.length_plasma_lens
                      + self.length_drift_lens_to_screen)

        dispersion = bend_angle * length

        return dispersion
