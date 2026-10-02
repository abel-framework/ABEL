# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target import Target
from abc import abstractmethod

class TargetSFQED(Target):

    @abstractmethod
    def __init__(self, laser_a0, laser_waist_size, laser_duration, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0):
        """
        Abstract base class for SFQED targets.

        Parameters
        ----------
        laser_a0 : float
            Normalized vector potential of the laser at focus.
    
        laser_spot_size : [m] float
            Laser spot size at the focus/waist, assumed radially symmetric.

        laser_duration : [s] float
            Duration of the laser pulse (rms)

        laser_wavelength : [m] float, optional
            Wavelength of the laser. Default set to infrared (Ti:sapphire).

        laser_polarization : str, optional
            Polarization of the laser: [circular] or [linear]. Default set to ``circular``.

        collision_angle_deg : [deg] float, optional
            Collision angle of the laser with respect to the beam axis. Default set to 0.

        """

        super().__init__()
        
        # common variables
        self.laser_a0 = laser_a0
        self.laser_waist_size = laser_waist_size
        self.laser_duration = laser_duration
        self.laser_wavelength = laser_wavelength
        self.laser_polarization = laser_polarization
        self.collision_angle_deg = collision_angle_deg
        
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod
    def peak_chi(self):
        pass

    def get_length(self):
        return 0.0
        