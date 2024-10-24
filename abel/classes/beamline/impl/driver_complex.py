from abc import abstractmethod
from matplotlib import patches
from abel.classes.beamline.beamline import Beamline
from abel.classes.source.source import Source
from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.combiner_ring.combiner_ring import CombinerRing
from abel.classes.turnaround.turnaround import Turnaround

class DriverComplex(Beamline):
    
    def __init__(self, source=None, rf_accelerator=None, combiner_ring=None, turnaround=None, nom_energy=None, num_drivers=None, num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None):

        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)
        
        self.num_drivers = num_drivers
        self.nom_energy = None
        
        self.source = source
        self.rf_accelerator = rf_accelerator
        self.combiner_ring = combiner_ring
        self.turnaround = turnaround
        
    
    # assemble the trackables
    def assemble_trackables(self):
        
        self.trackables = []
        
        # add source
        assert(isinstance(self.source, Source))
        self.trackables.append(self.source)

        # add RF accelerator
        assert(isinstance(self.rf_accelerator, RFAccelerator))
        self.trackables.append(self.rf_accelerator)
        
        # set nominal energy
        if self.nom_energy is None:
            self.nom_energy = self.source.get_energy() + self.rf_accelerator.get_nom_energy_gain() 
        else:
            self.rf_accelerator.nom_energy_gain = self.nom_energy - self.source.get_energy()

        # add combiner ring
        if self.combiner_ring is not None:
            assert(isinstance(self.combiner_ring, CombinerRing))
            self.combiner_ring.nom_energy = self.nom_energy
            self.trackables.append(self.combiner_ring)
            
        # add turnaround
        if self.turnaround is not None:
            assert(isinstance(self.turnaround, Turnaround))
            self.turnaround.nom_energy = self.nom_energy
            self.trackables.append(self.turnaround)
        
        # set the bunch train pattern etc.
        super().assemble_trackables()
        
    
    def energy_usage(self):
        return self.source.energy_usage() + self.rf_accelerator.energy_usage()

    def get_charge(self):
        return self.source.get_charge()

    def get_nom_energy(self):
        return self.nom_energy
    
    def get_cost_breakdown(self):
        "Cost breakdown for the driver complex [ILC units]"
        
        breakdown = []
        breakdown.append(self.source.get_cost_breakdown())
        breakdown.append(self.rf_accelerator.get_cost_breakdown())
        if self.combiner_ring is not None:
            breakdown.append(self.combiner_ring.get_cost_breakdown())
        if self.turnaround is not None:
            breakdown.append(self.turnaround.get_cost_breakdown())
        breakdown.append(self.get_cost_breakdown_civil_construction())
        
        return ('Driver complex', breakdown)
        