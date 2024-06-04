from abel import Linac, Source, RFAccelerator, BeamDeliverySystem

class ConventionalLinac(Linac):
    
    def __init__(self, source=None, rf_accelerator=None, bds=None, nom_energy=None, bunch_separation=None, num_bunches_in_train=None, rep_rate_trains=None):
        
        self.source = source
        self.rf_accelerator = rf_accelerator
        self.bds = bds
        self.nom_energy = nom_energy
        
        super().__init__(bunch_separation=bunch_separation, num_bunches_in_train=num_bunches_in_train, rep_rate_trains=rep_rate_trains)

    # assemble the trackables
    def assemble_trackables(self):
        
        # declare list of trackables
        self.trackables = [None] * (2 + int(self.bds is not None))

        # add source
        assert(isinstance(self.source, Source))
        self.trackables[0] = self.source

        # add RF accelerator
        assert(isinstance(self.rf_accelerator, RFAccelerator))
        self.trackables[1] = self.rf_accelerator

        # set the nominal energy
        if self.nom_energy is None:
            self.nom_energy = self.source.get_energy() + self.rf_accelerator.get_nom_energy_gain()
        else:
            self.rf_accelerator.nom_energy_gain = self.nom_energy - self.source.get_energy()
            
        # set the bunch train pattern
        if self.num_bunches_in_train is not None and self.rep_rate_trains is not None:
            self.source.rep_rate = self.num_bunches_in_train*self.rep_rate_trains
            self.rf_accelerator.bunch_separation = self.bunch_separation
            self.rf_accelerator.num_bunches_in_train = self.num_bunches_in_train
            self.rf_accelerator.rep_rate_trains = self.rep_rate_trains
                   
        # add beam delivery system
        if self.bds is not None:
            assert(isinstance(self.bds, BeamDeliverySystem))
            if self.bds.nom_energy is None:
                self.bds.nom_energy = self.source.get_energy() + self.rf_accelerator.get_nom_energy_gain()
            self.trackables[2] = self.bds
    
    def energy_usage(self):
        return self.source.energy_usage() + self.rf_accelerator.energy_usage()
        
    def get_nom_energy(self):
        if self.rf_accelerator.get_nom_energy_gain() is not None:
            return self.source.get_energy() + self.rf_accelerator.get_nom_energy_gain()
        else:
            return None
    