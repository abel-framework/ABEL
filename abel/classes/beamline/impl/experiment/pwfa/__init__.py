from abel.classes.beamline.impl.experiment import Experiment
from abel.classes.stage.stage import Stage
from abel.classes.spectrometer.spectrometer import Spectrometer
from abel.classes.beamline.impl.linac.linac import Linac

class ExperimentPWFA(Experiment):
    
    def __init__(self, linac=None, stage=None, spectrometer=None):
        self.stage = stage
        
        super().__init__(linac=linac, test_device=stage, spectrometer=spectrometer)
            
    
    # assemble the trackables
    def assemble_trackables(self):

        # check element classes, then assemble
        
        assert(isinstance(self.linac, Linac))
        
        if self.test_device is None:
            self.test_device = self.stage
        assert(isinstance(self.test_device, Stage))

        assert(isinstance(self.spectrometer, Spectrometer))

        # assemble the RF linac
        self.linac.assemble_trackables()
        
        # set the nominal energy of the stage as that coming out of the linac
        self.stage.nom_energy = self.linac.nom_energy

        # set the spectrometer imaging energy to the nominal gain
        if self.spectrometer.imaging_energy_x is None:
            self.spectrometer.imaging_energy_x = self.stage.nom_energy + self.stage.nom_energy_gain
        
        # run beamline constructor
        super().assemble_trackables()


    # TODO: add useful scans (e.g., plasma-density scan, object-plane scan, etc.)

    

    