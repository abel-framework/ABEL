from abel.classes.beamline.impl.experiment import Experiment
from abel.classes.plasma_lens.plasma_lens import PlasmaLens

class ExperimentAPL(Experiment):
    
    def __init__(self, linac=None, plasma_lens=None, spectrometer=None):
        self.plasma_lens = plasma_lens
        
        super().__init__(linac=linac, test_device=plasma_lens, spectrometer=spectrometer)
    
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assemble
        if self.test_device is None:
            self.test_device = self.plasma_lens
        assert(isinstance(self.test_device, PlasmaLens))
        
        # run beamline constructor
        super().assemble_trackables()
    