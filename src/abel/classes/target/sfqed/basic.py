# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target.sfqed import TargetSFQED

class TargetSFQEDBasic(TargetSFQED):
    
    def __init__(self, laser_a0=None, laser_waist_radius=None, laser_duration_fwhm=None, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0, nom_energy=None):
        
        super().__init__(laser_a0=laser_a0, laser_waist_radius=laser_waist_radius, laser_duration_fwhm=laser_duration_fwhm, laser_wavelength=laser_wavelength, laser_polarization=laser_polarization, collision_angle_deg=collision_angle_deg, nom_energy=nom_energy)

        self.__peak_chi = None
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # if not externally set, set the nominal energy
        if self.nom_energy is None:
            self.nom_energy = beam.energy()
            
        # calculate peak chi based on beam and laser parameters
        self.__peak_chi = self.peak_chi_ideal()
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def peak_chi(self):
        return self.__peak_chi
    