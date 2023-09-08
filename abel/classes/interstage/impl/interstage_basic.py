import scipy.constants as SI
from abel import Interstage

class InterstageBasic(Interstage):
    
    def __init__(self, nom_energy=None, dipole_length=None, dipole_field=None):
        self.nom_energy = nom_energy
        self.dipole_length = dipole_length
        self.dipole_field = dipole_field
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # compress beam
        beam.compress(R_56=self.R_56(), nom_energy=self.nom_energy)

        # flip transverse phase spaces
        beam.flip_transverse_phase_spaces()
        
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
        