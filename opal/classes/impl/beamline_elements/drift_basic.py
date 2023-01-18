from opal import Drift
from opal.utilities import SI
import numpy as np
class DriftBasic(Drift):
    
    def __init__(self, L):
        self.L = L
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get phase space
        pzs = beam.pzs() # [eV/c]
        X0 = beam.transverseVector()
        
        # define drift transfer matrix
        R = np.eye(4)
        R[0,1] = self.L
        R[2,3] = self.L
        
        # apply transfer matrix
        X = np.dot(R, X0)
        
        # save transverse vector
        beam.setTransverseVector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def length(self):
        return self.L
    
    def plotObject(self):
        rect = patches.Rectangle((0, 0), self.length(), 0)
        rect.set_facecolor = 'k'
        return rect
    
    def transferMatrix(self):
        R = np.eye(4)
        R[0,1] = self.L
        R[2,3] = self.L
        return R