from opal import Quadrupole
from opal.utilities import SI
import numpy as np
class QuadrupoleBasic(Quadrupole):
    
    def __init__(self, L, k, E0=None):
        self.L = L
        self.k = k
        self.E0 = E0
        
    def transferMatrix(self, E=None):
        
        # scale by energy offset
        if E is not None:
            k = self.k * self.E0 / E
        else:
            k = self.k
            
        if (self.k>0):
            R = np.zeros((4,4))
            R[3,3] = np.cos(self.L*np.sqrt(abs(k)))
            R[2,3] = np.sin(self.L*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[2,2] = np.cos(self.L*np.sqrt(abs(k)))
            R[3,2] = np.sin(self.L*np.sqrt(abs(k)))*(-np.sqrt(abs(k)))
            R[0,0] = np.cosh(self.L*np.sqrt(abs(k)))
            R[0,1] = np.sinh(self.L*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[1,0] = np.sinh(self.L*np.sqrt(abs(k)))*(np.sqrt(abs(k)))
            R[1,1] = np.cosh(self.L*np.sqrt(abs(k)))
            
        elif (self.k<0):
            R = np.zeros((4,4))
            R[0,0] = np.cos(self.L*np.sqrt(abs(k)))
            R[0,1] = np.sin(self.L*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[1,1] = np.cos(self.L*np.sqrt(abs(k)))
            R[1,0] = np.sin(self.L*np.sqrt(abs(k)))*(-np.sqrt(abs(k)))
            R[2,2] = np.cosh(self.L*np.sqrt(abs(k)))
            R[2,3] = np.sinh(self.L*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[3,2] = np.sinh(self.L*np.sqrt(abs(k)))*(np.sqrt(abs(k)))
            R[3,3] = np.cosh(self.L*np.sqrt(abs(k)))
            
        else: 
            R = np.eye(4)
            R[0,1] = self.L
            R[2,3] = self.L

        return R
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        #Get transverse vector
        X0 = beam.transverseVector()
        X = np.zeros((4,beam.Npart()))
        
        #Find and set new transverse vector
        Es = beam.Es()
        for i in range(beam.Npart()):
            M = self.transferMatrix(Es[i])
            X[:,i] =  np.dot(M, X0[:,i])
            
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