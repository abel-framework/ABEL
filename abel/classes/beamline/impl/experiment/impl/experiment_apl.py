# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
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
    