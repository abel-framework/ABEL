# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.spectrometer.spectrometer import Spectrometer

class SpectrometerBasicCLEAR(Spectrometer):
    
    def __init__(self, use_otr_screen=True):
        
        super().__init__()
        
        self.use_otr_screen = use_otr_screen
        self.location_otr_screen = 0.3 # [m]
    
    # lattice length
    def get_length(self):
        if self.use_otr_screen:
            return self.location_otr_screen
        else:
            return 0
    
    # tracking function
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # drift transport
        if self.use_otr_screen:            
            beam.transport(self.location_otr_screen)
        
        return super().track(beam, savedepth, runnable, verbose)
        
    