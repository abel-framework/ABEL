import copy
import numpy as np
from abel import BeamDeliverySystem

class BeamDeliverySystemBasic(BeamDeliverySystem):
    
    def __init__(self, beta_x=None, beta_y=None, bunch_length=None, length=0, nom_energy=None):
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.bunch_length = bunch_length
        self.length = length
        self.nom_energy = nom_energy
    
    def get_length(self):
        if self.nom_energy is not None:
            return np.sqrt(self.nom_energy/500e9)*2250 # [m] scaled from ILC
        else:
            return self.length
    
    def get_nom_energy(self):
        return self.nom_energy
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # transport phase spaces to waist (in each plane)
        ds_x = beam.alpha_x()/beam.gamma_x()
        ds_y = beam.alpha_y()/beam.gamma_y()
        
        # find waist beta functions (in each plane)
        Rx = np.eye(4)
        Rx[0,1] = ds_x
        Rx[2,3] = ds_x
        beamx = copy.deepcopy(beam)
        beamx.set_transverse_vector(np.dot(Rx, beamx.transverse_vector()))

        Ry = np.eye(4)
        Ry[0,1] = ds_y
        Ry[2,3] = ds_y
        beamy = copy.deepcopy(beam)
        beamy.set_transverse_vector(np.dot(Ry, beamy.transverse_vector()))
        
        # scale the waist phase space by beta functions
        X = beamx.transverse_vector()
        Y = beamy.transverse_vector()
        X[0,:] = X[0,:] * np.sqrt(self.beta_x/beamx.beta_x())
        X[1,:] = X[1,:] / np.sqrt(self.beta_x/beamx.beta_x())
        X[2,:] = Y[2,:] * np.sqrt(self.beta_y/beamy.beta_y())
        X[3,:] = Y[3,:] / np.sqrt(self.beta_y/beamy.beta_y()) 
        beam.set_transverse_vector(X)
        
        # stretch or compress longitudinally 
        if self.bunch_length is not None:
            beam.scale_to_length(self.bunch_length)
        
        return super().track(beam, savedepth, runnable, verbose)
        
        
        
        
        
        