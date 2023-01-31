import numpy as np
from opal import BeamDeliverySystem, DriftBasic
from copy import deepcopy

class BeamDeliverySystemBasic(BeamDeliverySystem):
    
    def __init__(self, beta_waist_x = None, beta_waist_y = None, L = 0, E0 = None):
        self.beta_waist_x = beta_waist_x
        self.beta_waist_y = beta_waist_y
        self.L = L
        self.E0 = E0
    
    def length(self):
        if self.E0 is None:
            return self.L
        else:
            return np.sqrt(self.E0/1e12)*2250 # [m] scaled from ILC
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # transport phase spaces to waist (in each plane)
        ds_x = beam.alphaX()/beam.gammaX()
        ds_y = beam.alphaY()/beam.gammaY()
        
        # find waist beta functions (in each plane)
        beamx = DriftBasic(ds_x).track(deepcopy(beam))
        beamy = DriftBasic(ds_y).track(deepcopy(beam))
        
        # scale the waist phase space by beta functions
        X = beamx.transverseVector()
        Y = beamy.transverseVector()
        X[0,:] = X[0,:] * np.sqrt(self.beta_waist_x/beamx.betaX())
        X[1,:] = X[1,:] / np.sqrt(self.beta_waist_x/beamx.betaX())
        X[2,:] = Y[2,:] * np.sqrt(self.beta_waist_y/beamy.betaY())
        X[3,:] = Y[3,:] / np.sqrt(self.beta_waist_y/beamy.betaY()) 
        beam.setTransverseVector(X)
        
        return super().track(beam, savedepth, runnable, verbose)
        
        
        
        
        
        