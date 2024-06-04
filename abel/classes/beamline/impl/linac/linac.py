from abc import abstractmethod
from abel import Beamline

class Linac(Beamline):
    
    def __init__(self, bunch_separation=None, num_bunches_in_train=None, rep_rate_trains=None):
        
        self.bunch_separation = bunch_separation
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains
        
        super().__init__()

    @abstractmethod
    def get_nom_energy(self):
        pass

    def rep_rate_average(self):
        if self.rep_rate_trains is not None and self.num_bunches_in_train is not None:
            return self.rep_rate_trains*self.num_bunches_in_train
        else:
            return None  
    
    def effective_gradient(self):
        return self.nom_energy()/self.get_length()
    