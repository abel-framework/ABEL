from opal import Dipole
from opal.utilities import SI
import numpy as np
class DipoleSpectrometerBasic(Dipole):
    
    def __init__(self, L, B, isVertical=False):
        self.L = L
        self.B = B
        self.isVertical = isVertical
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get phase space
        pzs = beam.pzs() # [eV/c]
        qSigns = np.sign(beam.qs())
        X0 = beam.transverseVector()
        
        # calculate final transverse vector (Lorentz force, particle by particle)
        X = np.zeros((4,beam.Npart()))
        for i in range(beam.Npart()):
            if not self.isVertical:
                dxp = float(qSigns[i])*SI.c*self.B*self.L/pzs[i]
                X[0,i] = dxp*self.L/2 + X0[1,i]*self.L + X0[0,i] 
                X[1,i] = dxp + X0[1,i] 
                X[2,i] = X0[3,i]*self.L + X0[2,i] 
                X[3,i] = X0[3,i] 
            else:
                dyp = float(qSigns[i])*SI.c*self.B*self.L/pzs[i]
                X[0,i] = X0[1,i]*self.L + X0[0,i] 
                X[1,i] = X0[1,i] 
                X[2,i] = dyp*self.L/2 + X0[3,i]*self.L + X0[2,i]
                X[3,i] = dyp + X0[3,i]
            
        # save transverse vector
        beam.setTransverseVector(X)
        
        return super().track(beam, savedepth, runnable, verbose)

    def length(self):
        return self.L
    
    def field(self):
        return self.B
    
    # TODO: make correct (fringe fields and weak focusing)
    def transferMatrix(self, E0=None):
        
        # first make a drift
        R = np.eye(4)
        R[0,1] = self.L
        R[2,3] = self.L

        if E0 is not None:
            
            # calculate bending radius
            rho = self.B*SI.c/E0
            
            if not self.isVertical: # horizontal dipole
                R[0,0] = np.cos(self.L/rho)
                R[0,1] = rho*np.sin(self.L/rho)
                R[1,0] = -1/rho*np.sin(self.L/rho)
                R[1,1] = np.cos(self.L/rho)
            else: # vertical dipole
                R[2,2] = np.cos(self.L/rho)
                R[2,3] = rho*np.sin(self.L/rho)
                R[3,2] = -1/rho*np.sin(self.L/rho)
                R[3,3] = np.cos(self.L/rho)

        return R