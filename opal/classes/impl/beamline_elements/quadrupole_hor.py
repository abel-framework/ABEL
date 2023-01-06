from opal import Quadrupole
from opal.utilities import SI
import numpy as np
class QuadrupoleHor(Quadrupole):
    
    def __init__(self, L, k):
        self.L = L
        self.k = k
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        #Get transverse vector
        X0 = beam.transverseVector()
        X = np.zeros((4,beam.Npart()))
        
        #Compute transformation vector
        TransFormVector = np.zeros((4,4))
        TransFormVector[0,0] = np.cos(self.L*np.sqrt(self.k))
        TransFormVector[0,1] = np.sin(self.L*np.sqrt(self.k))/np.sqrt(self.k)
        TransFormVector[1,1] = np.cos(self.L*np.sqrt(self.k))
        TransFormVector[1,0] = np.sin(self.L*np.sqrt(self.k))*(-np.sqrt(self.k))
        TransFormVector[2,2] = np.cosh(self.L*np.sqrt(self.k))
        TransFormVector[2,3] = np.sinh(self.L*np.sqrt(self.k))/np.sqrt(self.k)
        TransFormVector[3,2] = np.sinh(self.L*np.sqrt(self.k))*(np.sqrt(self.k))
        TransFormVector[3,3] = np.cosh(self.L*np.sqrt(self.k))
            
        #Find and set new transverse vector
        for i in range(beam.Npart()):
            X[:,i] =  np.dot(X0[:,i],TransFormVector)
            
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