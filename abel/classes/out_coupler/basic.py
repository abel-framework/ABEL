from abel.classes.out_coupler import OutCoupler
import scipy.constants as SI
import numpy as np

class OutCouplerBasic(OutCoupler):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole)
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        print('TODO')
        
        return super().track(beam, savedepth, runnable, verbose)
        