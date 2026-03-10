# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abc import abstractmethod
from abel.classes.trackable import Trackable
import numpy as np

class PlasmaLens(Trackable):
    
    @abstractmethod
    def __init__(self, length, radius, current, offset_x=0, offset_y=0):

        super().__init__()
        
        # common variables
        self.length = length
        self.radius = radius
        self.current = current

        self.offset_x = offset_x
        self.offset_y = offset_y
        
    
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    @abstractmethod
    def get_focusing_gradient(self):
        pass
        
    def get_length(self):
        return self.length

    def survey_object(self):
        from matplotlib import patches
        return patches.Rectangle((0, -1), self.get_length(), 2)
    