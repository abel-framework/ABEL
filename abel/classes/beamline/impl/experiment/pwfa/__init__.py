from abel.classes.beamline.impl.experiment import Experiment
from abel.classes.stage.stage import Stage

class ExperimentPWFA(Experiment):
    
    def __init__(self, linac=None, stage=None, spectrometer=None):
        self.stage = stage
        
        super().__init__(linac=linac, test_device=stage, spectrometer=spectrometer)
            
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assemble
        if self.test_device is None:
            self.test_device = self.stage
        assert(isinstance(self.test_device, Stage))
        
        # run beamline constructor
        super().assemble_trackables()

    