from opal import Quadrupole
import scipy.constants as SI
import numpy as np
class QuadrupoleBasic(Quadrupole):
    
    def __init__(self, length, strength, nom_energy=None):
        self.length = length
        self.strength = strength
        self.nom_energy = nom_energy
        
    def transferMatrix(self, E=None):
        
        # scale by energy offset
        if E is not None:
            k = self.strength * self.nom_energy / E
        else:
            k = self.strength
            
        if (self.strength>0):
            R = np.zeros((4,4))
            R[3,3] = np.cos(self.length*np.sqrt(abs(k)))
            R[2,3] = np.sin(self.length*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[2,2] = np.cos(self.length*np.sqrt(abs(k)))
            R[3,2] = np.sin(self.length*np.sqrt(abs(k)))*(-np.sqrt(abs(k)))
            R[0,0] = np.cosh(self.length*np.sqrt(abs(k)))
            R[0,1] = np.sinh(self.length*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[1,0] = np.sinh(self.length*np.sqrt(abs(k)))*(np.sqrt(abs(k)))
            R[1,1] = np.cosh(self.length*np.sqrt(abs(k)))
            
        elif (self.strength<0):
            R = np.zeros((4,4))
            R[0,0] = np.cos(self.length*np.sqrt(abs(k)))
            R[0,1] = np.sin(self.length*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[1,1] = np.cos(self.length*np.sqrt(abs(k)))
            R[1,0] = np.sin(self.length*np.sqrt(abs(k)))*(-np.sqrt(abs(k)))
            R[2,2] = np.cosh(self.length*np.sqrt(abs(k)))
            R[2,3] = np.sinh(self.length*np.sqrt(abs(k)))/np.sqrt(abs(k))
            R[3,2] = np.sinh(self.length*np.sqrt(abs(k)))*(np.sqrt(abs(k)))
            R[3,3] = np.cosh(self.length*np.sqrt(abs(k)))
            
        else: 
            R = np.eye(4)
            R[0,1] = self.length
            R[2,3] = self.length

        return R
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        #Get transverse vector
        X0 = beam.transverse_vector()
        X = np.zeros((4,len(beam)))
        
        #Find and set new transverse vector
        Es = beam.Es()
        for i in range(len(beam)):
            M = self.transferMatrix(Es[i])
            X[:,i] =  np.dot(M, X0[:,i])
            
        beam.set_transverse_vector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        return self.length
    
    def get_strength(self):
        return self.k
    
    def survey_object(self):
        rect = patches.Rectangle((0, 0), self.get_length(), 0)
        rect.set_facecolor = 'k'
        return rect