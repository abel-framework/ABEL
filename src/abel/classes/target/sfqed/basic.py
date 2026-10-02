# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target.sfqed import TargetSFQED
import numpy as np
from scipy import constants as SI

class TargetSFQEDBasic(TargetSFQED):
    
    def __init__(self, laser_a0=None, laser_waist_size=None, laser_duration=None, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0):
        
        super().__init__(laser_a0=laser_a0, laser_waist_size=laser_waist_size, laser_duration=laser_duration, laser_wavelength=laser_wavelength, laser_polarization=laser_polarization, collision_angle_deg=collision_angle_deg)

        self.__peak_chi = None
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # collision angle in radians
        theta = self.collision_angle_deg*np.pi/180
        
        # laser angular frequency in [rad/s]
        omega_laser = 2*np.pi*SI.c/ self.laser_wavelength
        
        # energy parameter
        eta = beam.gamma() * SI.hbar * omega_laser * (1 + np.cos(theta)) / (SI.m_e * SI.c**2)
        
        # quantum parameter
        chi = self.laser_a0 * eta
        
        self.__peak_chi = chi

        # calculate peak chi based on beam and laser parameters
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def peak_chi(self):
        if self.__peak_chi is not None:
            return self.__peak_chi