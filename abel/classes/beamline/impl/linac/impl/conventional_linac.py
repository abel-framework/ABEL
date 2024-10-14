from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.source.source import Source
from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.damping_ring.damping_ring import DampingRing
from abel.classes.turnaround.turnaround import Turnaround
from abel.classes.bds.bds import BeamDeliverySystem

class ConventionalLinac(Linac):
    
    def __init__(self, source=None, rf_injector=None, damping_ring=None, rf_accelerator=None, turnaround=None, bds=None, nom_energy=None, num_bunches_in_train=None,  bunch_separation=None, rep_rate_trains=None):
        
        self.source = source
        self.rf_injector = rf_injector
        self.damping_ring = damping_ring
        self.rf_accelerator = rf_accelerator
        self.turnaround = turnaround
        self.bds = bds
        
        self.name = 'RF linac'
        
        super().__init__(nom_energy, num_bunches_in_train, bunch_separation, rep_rate_trains)

    
    # assemble the trackables
    def assemble_trackables(self):
        
        # declare list of trackables
        self.trackables = []
        
        # add source
        assert(isinstance(self.source, Source))
        self.trackables.append(self.source)

        # add RF injector (optional)
        if self.rf_injector is not None:
            assert(isinstance(self.rf_injector, RFAccelerator))
            self.rf_injector.name = 'RF injector'
            self.trackables.append(self.rf_injector)

        # add damping ring (optional)
        if self.damping_ring is not None:
            assert(isinstance(self.damping_ring, DampingRing))
            self.trackables.append(self.damping_ring)
        
        # add RF accelerator
        assert(isinstance(self.rf_accelerator, RFAccelerator))
        self.trackables.append(self.rf_accelerator)

        # add turnaround (optional)
        if self.turnaround is not None:
            assert(isinstance(self.turnaround, Turnaround))
            self.trackables.append(self.turnaround)

        
        # set the nominal energy or gains in the damping ring and main rf accelerator
        E0 = self.source.get_energy()
        if self.rf_injector is not None:
            E0 += self.rf_injector.get_nom_energy_gain()
            self.rf_injector.nom_energy = E0
        if self.damping_ring is not None:
            self.damping_ring.nom_energy = E0
        if self.nom_energy is None:
            self.nom_energy = E0 + self.rf_accelerator.get_nom_energy_gain()
        else:
            self.rf_accelerator.nom_energy_gain = self.nom_energy - E0
        if self.turnaround is not None:
            self.turnaround.nom_energy = self.nom_energy
        
        # add beam delivery system
        if self.bds is not None:
            assert(isinstance(self.bds, BeamDeliverySystem))

            # TODO: set to nominal if not already set to a length
            if self.bds.length is None or self.bds.length == 0:
                self.bds.length = None
                self.bds.nom_energy = self.get_nom_energy()
                self.bds.length = self.bds.get_length()
            self.trackables.append(self.bds)

        # set the bunch train pattern etc.
        super().assemble_trackables()
        
    
    def energy_usage(self):
        E = self.source.energy_usage()
        if self.rf_injector is not None:
            E += self.rf_injector.energy_usage()
        if self.damping_ring is not None:
            E += self.damping_ring.energy_usage()
        E += self.rf_accelerator.energy_usage()
        return E

    
    def get_nom_energy(self):
        return self.source.get_energy() + self.get_nom_energy_gain()


    def get_nom_energy_gain(self):

        nom_energy_gain = 0
        
        # add injector energy (if exists)
        if self.rf_injector is not None:
            nom_energy_gain += self.rf_injector.get_nom_energy_gain()

        # add main RF accelerator energy
        if self.rf_accelerator.get_nom_energy_gain() is not None:
             nom_energy_gain += self.rf_accelerator.get_nom_energy_gain()

        return nom_energy_gain
    

    def get_cost_breakdown(self):

        breakdown = []
        breakdown.append(self.source.get_cost_breakdown())
        if self.rf_injector is not None:
            breakdown.append((self.rf_injector.get_cost_breakdown()))
        if self.damping_ring is not None:
            breakdown.append(self.damping_ring.get_cost_breakdown())
        breakdown.append(self.rf_accelerator.get_cost_breakdown())
        if self.turnaround is not None:
            breakdown.append((self.turnaround.get_cost_breakdown()))
        if self.bds is not None:
            breakdown.append(self.bds.get_cost_breakdown())
        breakdown.append(self.get_cost_breakdown_civil_construction())
        
        return (self.name, breakdown)
    