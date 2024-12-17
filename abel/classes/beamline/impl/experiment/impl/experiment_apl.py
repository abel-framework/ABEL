from abel import Experiment, Linac, PlasmaLens, Spectrometer

class ExperimentAPL(Experiment):
    
    def __init__(self, linac=None, plasma_lens=None, spectrometer=None):
        self.plasma_lens = plasma_lens
        
        super().__init__(linac=linac, component=plasma_lens, spectrometer=spectrometer)
    
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assemble
        if self.component is None:
            self.component = self.plasma_lens
        assert(isinstance(self.component, PlasmaLens))
        
        # run beamline constructor
        super().assemble_trackables()
    