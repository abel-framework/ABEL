from opal import Drift
import scipy.constants as SI
import numpy as np
class DriftBasic(Drift):
    
    def __init__(self, length):
        self.length = length
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get phase space
        pzs = beam.pzs() # [eV/c]
        X0 = beam.transverse_vector()
        
        # define drift transfer matrix
        R = np.eye(4)
        R[0,1] = self.length
        R[2,3] = self.length
        
        # apply transfer matrix
        X = np.dot(R, X0)
        
        # save transverse vector
        beam.set_transverse_vector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        return self.length
    
    def survey_object(self):
        rect = patches.Rectangle((0, 0), self.get_length(), 0)
        rect.set_facecolor = 'k'
        return rect
    
    def transferMatrix(self):
        R = np.eye(4)
        R[0,1] = self.length
        R[2,3] = self.length
        return R