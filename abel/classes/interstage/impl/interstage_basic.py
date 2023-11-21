import scipy.constants as SI
from abel import Interstage
import numpy as np
import matplotlib.pyplot as plt

class InterstageBasic(Interstage):
    
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None, beta0=None, phase_advance=1.5*np.pi):
        self.nom_energy = nom_energy
        self.dipole_length = dipole_length
        self.dipole_field = dipole_field
        self.beta0 = beta0
        self.phase_advance = phase_advance
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # compress beam
        beam.compress(R_56=self.R_56(), nom_energy=self.nom_energy)
        
        # rotate transverse phase spaces (assumed achromatic)
        if callable(self.beta0):
            betas = self.beta0(beam.Es())
        else:
            betas = self.beta0
        theta = self.phase_advance
        xs_rotated = beam.xs()*np.cos(theta) + betas*beam.xps()*np.sin(theta)
        xps_rotated = -beam.xs()*np.sin(theta)/betas + beam.xps()*np.cos(theta)
        beam.set_xs(xs_rotated)
        beam.set_xps(xps_rotated)
        ys_rotated = beam.ys()*np.cos(theta) + betas*beam.yps()*np.sin(theta)
        yps_rotated = -beam.ys()*np.sin(theta)/betas + beam.yps()*np.cos(theta)
        beam.set_ys(ys_rotated)
        beam.set_yps(yps_rotated)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    # evaluate dipole length (if it is a function)
    def __eval_dipole_length(self):
        if callable(self.dipole_length):
            return self.dipole_length(self.nom_energy)
        else:
            return self.dipole_length
    
    # evaluate longitudinal dispersion (R56)
    def R_56(self):
        return -self.dipole_field**2*SI.c**2*self.__eval_dipole_length()**3/(3*self.nom_energy**2)
    
    
    # lattice length
    def get_length(self):
        return 4.7875*self.__eval_dipole_length()
        