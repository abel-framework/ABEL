import numpy as np
from abel import BeamDeliverySystem

class BeamDeliverySystemFlatBeamTransformer(BeamDeliverySystem):
    
    def __init__(self, length=0, nom_energy=None, beta=None):
        self.length = length
        self.nom_energy = nom_energy
        self.beta = beta
        self.phase_advance_sign = -1
    
    def get_length(self):
        return self.length
        
    def get_nom_energy(self):
        return self.nom_energy
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # Based on: R. Brinkmann, Y. Derbenev, and K. Flöttmann PRAB 2001 (10.1103/PhysRevSTAB.4.053501)

        # assuming beta is identical in both planes
        if self.beta is None:
            self.beta = beam.beta_x(clean=True)
        #elif callable(self.beta):
        #    return self.beta(self.nom_energy)
        #else:
        #    return self.beta

        # calculate first 2x2 matrix
        mu = 0#*np.pi/4
        M = np.array([[np.cos(mu), self.beta*np.sin(mu)], [-np.sin(mu)/self.beta, np.cos(mu)]])

        # calculate second 2x2 matrix
        mu2 = mu + np.sign(self.phase_advance_sign)*np.pi/2
        N = np.array([[np.cos(mu2), self.beta*np.sin(mu2)], [-np.sin(mu2)/self.beta, np.cos(mu2)]])

        # calculate 4x4 skew-quadrupole matrix
        Cplus = (N+M)/2
        Cminus = (N-M)/2
        Cskew = np.concatenate([np.concatenate([Cplus, Cminus]), np.concatenate([Cminus, Cplus])], axis=1)

        # apply change
        beam.set_transverse_vector(Cskew @ beam.transverse_vector())
        
        return super().track(beam, savedepth, runnable, verbose)
        
        
        
        
        
        