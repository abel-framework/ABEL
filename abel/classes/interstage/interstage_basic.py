import scipy.constants as SI
from abel.classes.interstage import Interstage
import numpy as np

class InterstageBasic(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, use_nonlinearity=True, use_sextupole=True,
                       enable_csr=True, enable_isr=True, enable_space_charge=False, phase_advance=2*np.pi):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, 
                         use_nonlinearity=use_nonlinearity, use_sextupole=use_sextupole, 
                         enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)
        
        self.phase_advance = phase_advance
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # compress beam
        beam.compress(R_56=self.R56, nom_energy=self.nom_energy)
        
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
        