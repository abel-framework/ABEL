from abc import abstractmethod
from abel import Beamline

class Linac(Beamline):
    
    def __init__(self, nom_energy=None, num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None):
        
        # set bunch pattern
        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)

        self.nom_energy = nom_energy
        

    def get_nom_energy(self):
        return self.nom_energy
        
    
    def get_effective_gradient(self):
        return self.get_nom_energy()/self.get_length()
    