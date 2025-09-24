from abc import abstractmethod
from abel.classes.beamline.beamline import Beamline

class Linac(Beamline):
    
    def __init__(self, source=None, nom_energy=None, num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None):
        
        # set bunch pattern
        super().__init__(num_bunches_in_train=num_bunches_in_train, bunch_separation=bunch_separation, rep_rate_trains=rep_rate_trains)

        self.source = source
        self.nom_energy = nom_energy

    
    def assemble_trackables(self):
        
        # if not set, use the source bunch pattern
        if self.source.bunch_separation is not None and self.bunch_separation is None:
            self.bunch_separation = self.source.bunch_separation
        if self.source.num_bunches_in_train is not None and self.num_bunches_in_train is None:
            self.num_bunches_in_train = self.source.num_bunches_in_train
        if self.source.rep_rate_trains is not None and self.rep_rate_trains is None:
            self.rep_rate_trains = self.source.rep_rate_trains

        # set the bunch train pattern etc.
        super().assemble_trackables()
    

    @property
    def nom_energy(self) -> float | None:
        "The nominal energy [eV] of the linac."
        return self._nom_energy
    @nom_energy.setter
    def nom_energy(self, energy : float | None):
        if energy is not None and energy < 0.0:
            raise ValueError('Nominal energy cannot be negative.')
        self._nom_energy = energy
    _nom_energy = None


    def get_nom_energy(self):
        "Alias of linac nominal energy."
        return self.nom_energy
    
    
    def get_nom_beam_power(self):
        return abs(self.nom_energy * self.source.get_charge() * self.get_rep_rate_average())
    

    def get_effective_gradient(self):
        return self.get_nom_energy()/self.get_length()

        
    def energy_usage(self):
        if self.trackables is None:
            self.assemble_trackables()
        Etot = 0
        for trackable in self.trackables:
            Etot += trackable.energy_usage()
        return Etot


    def get_cost_breakdown(self):
        "Cost breakdown for the linac [ILC units]"
        
        breakdown = []
        
        # cost of the civil construction
        for trackable in self.trackables:
            breakdown.append(trackable.get_cost_breakdown())

        return (self.name, breakdown)