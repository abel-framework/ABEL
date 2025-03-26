from abc import abstractmethod
from abel import Spectrometer
import numpy as np
import matplotlib.pyplot as plt
from abel import CONFIG


class SpectrometerAchromatic(Spectrometer):
    """
    Achromatic Spectrometer class.

    Parameters
    ----------
    length_drift_to_dipole : float, optional
        Length of the drift space to the dipole magnet in meters. Default is 0.2.
    field_dipole : float, optional
        Magnetic field strength of the dipole magnet in Tesla.
        Negative means bending the beam downwards.
        Default is -0.85.
    length_dipole : float, optional
        Length of the dipole magnet in meters. Default is 0.8.
    length_drift_dipole_to_lens : float, optional
        Length of the drift space from the dipole magnet to the lens in meters. Default is 0.2.
    length_plasma_lens : float, optional
        Length of the plasma lens in meters. Default is 20E-3.
    radius_plasma_lens : float, optional
        Radius of the plasma lens in meters. Default is 1E-3.
    length_drift_lens_to_screen : float, optional
        Length of the drift space from the lens to the screen in meters. Default is 0.4.
    imaging_energy : float or None, optional
        Imaging energy in electron volts. Default is None.
    disable_lens_nonlinearity : bool, optional
        Flag to disable lens nonlinearity. Default is True.
    """

    @abstractmethod
    def __init__(self,
                 length_drift_to_dipole,
                 field_dipole,
                 length_dipole,
                 length_drift_dipole_to_lens,
                 length_plasma_lens,
                 radius_plasma_lens,
                 length_drift_lens_to_screen,
                 imaging_energy,
                 disable_lens_nonlinearity):

        self.length_drift_to_dipole = length_drift_to_dipole  # [m]
        self.field_dipole = field_dipole  # [T]
        self.length_dipole = length_dipole  # [m]

        self.length_drift_dipole_to_lens = length_drift_dipole_to_lens  # [m]
        self.length_plasma_lens = length_plasma_lens  # [m]

        self.imaging_energy = imaging_energy  # [eV]
        self.length_drift_lens_to_screen = length_drift_lens_to_screen  # [m]

        self._radius_plasma_lens = radius_plasma_lens  # [m]
        self.disable_lens_nonlinearity = disable_lens_nonlinearity  # [bool]

        super().__init__()

    def get_length(self):
        """
        Calculate the total length of the spectrometer lattice.

        Returns
        -------
        float
            Total length of the spectrometer lattice in meters.
        """
        total_length = (self.length_drift_to_dipole
                        + self.length_dipole
                        + self.length_drift_dipole_to_lens
                        + self.length_plasma_lens
                        + self.length_drift_lens_to_screen)
        return total_length

    @property
    def radius_plasma_lens(self):
        return self._radius_plasma_lens

    @radius_plasma_lens.setter
    def radius_plasma_lens(self, value):
        assert value >= 0, 'Radius of the plasma lens must be greater than or equal to zero.'
        self._radius_plasma_lens = value

    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        """
        Track the beam through the spectrometer.

        Parameters
        ----------
        beam : object
            The beam object to be tracked.
        savedepth : int, optional
            Depth of saved states during tracking. Default is 0.
        runnable : object or None, optional
            Runnable object for tracking. Default is None.
        verbose : bool, optional
            Flag to enable verbose output. Default is False.

        Returns
        -------
        object
            Result of the tracking process.
        """
        return super().track(beam, savedepth, runnable, verbose)
