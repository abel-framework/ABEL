from opal import Dipole
import scipy.constants as SI
import numpy as np

class DipoleSpectrometerBasic(Dipole):
    
    def __init__(self, length, field, is_vertical=False):
        self.length = length
        self.field = field
        self.is_vertical = is_vertical
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get phase space
        pzs = beam.pzs()
        qSigns = np.sign(beam.qs())
        X0 = beam.transverse_vector()
        
        # calculate final transverse vector (Lorentz force, particle by particle)
        X = np.zeros((4,len(beam)))
        for i in range(len(beam)):
            if not self.is_vertical:
                dxp = float(qSigns[i])*self.field*SI.e*self.length/pzs[i]
                X[0,i] = dxp*self.length/2 + X0[1,i]*self.length + X0[0,i] 
                X[1,i] = dxp + X0[1,i] 
                X[2,i] = X0[3,i]*self.length + X0[2,i] 
                X[3,i] = X0[3,i] 
            else:
                dyp = float(qSigns[i])*self.field*SI.e*self.length/pzs[i]
                X[0,i] = X0[1,i]*self.length + X0[0,i] 
                X[1,i] = X0[1,i] 
                X[2,i] = dyp*self.length/2 + X0[3,i]*self.length + X0[2,i]
                X[3,i] = dyp + X0[3,i]
            
        # save transverse vector
        beam.set_transverse_vector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        return self.length
    
    def get_field(self):
        return self.field
    
    # TODO: make correct (fringe fields and weak focusing)
    def transferMatrix(self, nom_energy=None):
        
        # first make a drift
        R = np.eye(4)
        R[0,1] = self.length
        R[2,3] = self.length

        if nom_energy is not None:
            
            # calculate bending radius
            rho = self.field*SI.c/nom_energy
            
            if not self.is_vertical: # horizontal dipole
                R[0,0] = np.cos(self.length/rho)
                R[0,1] = rho*np.sin(self.length/rho)
                R[1,0] = -1/rho*np.sin(self.length/rho)
                R[1,1] = np.cos(self.length/rho)
            else: # vertical dipole
                R[2,2] = np.cos(self.length/rho)
                R[2,3] = rho*np.sin(self.length/rho)
                R[3,2] = -1/rho*np.sin(self.length/rho)
                R[3,3] = np.cos(self.length/rho)

        return R