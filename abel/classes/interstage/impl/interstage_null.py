import scipy.constants as SI
from abel import Interstage

class InterstageNull(Interstage):
    
    def __init__(self, nom_energy=None):
        #self.nom_energy = nom_energy
        self.dipole_length = 0.0
        self.dipole_field = 0.0
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # compress beam
        #beam.compress(R_56=self.R_56(), nom_energy=self.nom_energy) ############

        # flip transverse phase spaces
        #beam.flip_transverse_phase_spaces()  ##########
        
        return super().track(beam, savedepth, runnable, verbose)


    # lattice length
    def get_length(self):
        return 0.0