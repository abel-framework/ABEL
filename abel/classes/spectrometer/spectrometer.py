# Copyright 2022-, The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

from abc import abstractmethod
from abel.classes.trackable import Trackable

class Spectrometer(Trackable):
    
    @abstractmethod
    def __init__(self):
        super().__init__()
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    def survey_object(self):
        from matplotlib import patches
        rect = patches.Rectangle((0, -0.05), self.get_length(), 0.1)
        rect.set_facecolor = 'k'
        return rect