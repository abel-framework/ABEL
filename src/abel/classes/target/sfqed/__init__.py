# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target import Target
from abc import abstractmethod
import numpy as np
from scipy import constants as SI

class TargetSFQED(Target):

    @abstractmethod
    def __init__(self, laser_a0, laser_waist_radius, laser_duration_fwhm, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0, nom_energy=None):
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
        self.laser_waist_radius = laser_waist_radius
        self.laser_duration_fwhm = laser_duration_fwhm
        self.laser_wavelength = laser_wavelength
        self.laser_polarization = laser_polarization
        self.collision_angle_deg = collision_angle_deg
        self.nom_energy = nom_energy
        
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod
    def peak_chi(self):
        pass


    def laser_intensity(self):
        K = 2*np.pi**2*(SI.epsilon_0*SI.m_e**2*SI.c**5/SI.e**2)
        return K * (self.laser_a0 / self.laser_wavelength)**2

    def laser_peak_power(self):
        return (np.pi/2) * self.laser_waist_radius**2 * self.laser_intensity()
        
    def laser_energy(self):
        return self.laser_peak_power() * self.laser_duration_fwhm * (2.35/2)
        
        
    
    def peak_chi_ideal(self):

        if self.nom_energy is not None:
            
            # collision angle in radians
            theta = self.collision_angle_deg*np.pi/180
            
            # laser angular frequency in [rad/s]
            omega_laser = 2*np.pi*SI.c/ self.laser_wavelength
            
            # energy parameter (using the beam's mean energy)
            from abel.utilities.relativity import energy2gamma
            eta = energy2gamma(self.nom_energy) * SI.hbar * omega_laser * (1 + np.cos(theta)) / (SI.m_e * SI.c**2)
            
            # quantum parameter
            chi = self.laser_a0 * eta
            return chi
            
        else:
            
            return None

    def print_laser_parameters(self):
        print('Laser parameters:')
        print(f'> Wavelength:   {self.laser_wavelength*1e9:.0f} nm')
        print(f'> a0:           {self.laser_a0:.1f}')
        print(f'> Duration:     {self.laser_duration_fwhm*1e15:.0f} fs FWHM')
        print(f'> Waist radius: {self.laser_waist_radius*1e6:.1f} µm')
        print('------------------------------')
        print(f'> Intensity:    {self.laser_intensity()/1e4:.2g} W/cm^2')
        print(f'> Peak power:   {self.laser_peak_power()/1e15:.2f} PW')
        print(f'> Energy:       {self.laser_energy():.2f} J')
            
    def get_length(self):
        return 0.0
        