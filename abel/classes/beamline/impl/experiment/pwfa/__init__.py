from abel.classes.beamline.impl.experiment import Experiment
from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
from abel.classes.beamline.impl.linac.linac import Linac

class ExperimentPWFA(Experiment):
    
    def __init__(self, linac=None, stage=None, spectrometer=None):
        self.stage = stage
        
        super().__init__(linac=linac, test_device=stage, spectrometer=spectrometer)
            
    
    # assemble the trackables
    def assemble_trackables(self):

        if isinstance(self.linac, Source):
            nom_energy = self.linac.energy
        elif isinstance(self.linac, Linac):
            snom_energy = self.linac.source.energy
        else:
            raise ValueError("You must define a linac that is either a type Source or a type Linac")

        # check element classes, then assemble
        if self.test_device is None:
            self.test_device = self.stage
        assert(isinstance(self.test_device, Stage))

        self.stage.nom_energy = nom_energy

        if self.spectrometer.imaging_energy_x is None:
            self.spectrometer.imaging_energy_x = nom_energy + self.stage.nom_energy_gain
        
        # run beamline constructor
        super().assemble_trackables()

    