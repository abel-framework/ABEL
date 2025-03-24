from abc import abstractmethod
from matplotlib import patches
from abel.classes.beamline.beamline import Beamline
from abel.classes.source.source import Source
from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.combiner_ring.combiner_ring import CombinerRing
from abel.classes.turnaround.turnaround import Turnaround
from abel.classes.transfer_line.transfer_line import TransferLine

class DriverComplex(Beamline):
    
    def __init__(self, source=None, rf_accelerator=None, combiner_ring=None, turnaround=None, transfer_line=None, nom_energy=None, num_drivers=None, num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None):

        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)
        
        self.num_drivers = num_drivers
        self.nom_energy = None
        
        self.source = source
        self.rf_accelerator = rf_accelerator
        self.combiner_ring = combiner_ring
        self.transfer_line = transfer_line
        self.turnaround = turnaround
        
    
    # assemble the trackables
    def assemble_trackables(self):
        
        self.trackables = []
        
        # add source
        assert(isinstance(self.source, Source))
        self.source.name = 'Driver source'
        self.trackables.append(self.source)

        # add RF accelerator
        assert(isinstance(self.rf_accelerator, RFAccelerator))
        self.rf_accelerator.name = 'Driver RF linac'
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
        if self.transfer_line is not None:
            assert(isinstance(self.transfer_line, TransferLine))
            self.transfer_line.nom_energy = self.nom_energy
            self.trackables.append(self.transfer_line)
        
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

    def get_nom_beam_power(self):
        return abs(self.nom_energy * self.source.get_charge() * self.get_rep_rate_average())

    def get_cost_breakdown_civil_construction(self):
        breakdown = []
        for trackable in self.trackables:
            if isinstance(trackable, Source):
                breakdown.append((f'{trackable.name} (cut & cover + surface building)', trackable.get_cost_civil_construction(cut_and_cover=True, surface_building=True)))
            elif isinstance(trackable, RFAccelerator):
                breakdown.append((f'{trackable.name} (cut & cover + surface building)', trackable.get_cost_civil_construction(cut_and_cover=True, surface_building=True)))
            elif isinstance(trackable, CombinerRing):
                breakdown.append((f'{trackable.name} (cut & cover)', trackable.get_cost_civil_construction(cut_and_cover=True)))
            elif isinstance(trackable, TransferLine):
                breakdown.append((f'{trackable.name} (small tunnel)', trackable.get_cost_civil_construction(tunnel_diameter=4)))
            elif isinstance(trackable, Turnaround):
                breakdown.append((f'{trackable.name} (small tunnel)', trackable.get_cost_civil_construction(tunnel_diameter=4)))
        return ('Civil construction', breakdown)
        
    def get_cost_breakdown(self):
        "Cost breakdown for the driver complex [ILC units]"
        
        breakdown = []
        breakdown.append(self.source.get_cost_breakdown())
        breakdown.append(self.rf_accelerator.get_cost_breakdown())
        if self.combiner_ring is not None:
            breakdown.append(self.combiner_ring.get_cost_breakdown())
        if self.transfer_line is not None:
            breakdown.append(self.transfer_line.get_cost_breakdown())
        if self.turnaround is not None:
            breakdown.append(self.turnaround.get_cost_breakdown())
        breakdown.append(self.get_cost_breakdown_civil_construction())
        
        return ('Driver complex', breakdown)
        