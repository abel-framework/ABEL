# Copyright 2022-, The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

from abel.classes.beamline.impl.experiment.experiment import Experiment
from abel.classes.plasma_lens.plasma_lens import PlasmaLens

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
    