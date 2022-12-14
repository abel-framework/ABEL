from opal import Quadrupole
from opal.utilities import SI
import numpy as np
class QuadrupoleBasic(Quadrupole):
    
    def __init__(self, L, k):
        self.L = L
        self.k = k
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get phase space
        pzs = beam.pzs() # [eV/c]
        X0 = beam.transverseVector()
        
        # TODO: calculate effect of quadrupole (energy dependent)
        X = X0
        
        # save transverse vector
        beam.setTransverseVector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def length(self):
        return self.L
    
    def strength(self):
        return self.k
    
    def plotObject(self):
        rect = patches.Rectangle((0, 0), self.length(), 0)
        rect.set_facecolor = 'k'
        return rect