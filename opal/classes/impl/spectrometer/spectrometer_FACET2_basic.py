from opal import Spectrometer, Beamline, DriftBasic, DipoleSpectrometerBasic, QuadrupoleBasic
import numpy as np
from numpy.linalg import multi_dot
from scipy.optimize import minimize

class SpectrometerFACET2Basic(Spectrometer):
    
    def __init__(self, B_dip = None, E_img = None, s_obj = None):
        self.B_dip = B_dip        
        self.E_img = E_img
        self.s_obj = s_obj
        
        self.ks = [-0.330432, 0.54533, -0.330432] # [m^-2]

    def beamline(self):
        
        # starts at z = 1998.708952 - 1.695 - 1.17 m
        drift0 = DriftBasic(0.202439 + 1.695 + 1.17 - self.s_obj)
        quad_Q0 = QuadrupoleBasic(1, self.ks[0])
        drift1 = DriftBasic(0.286595 + 0.580966 + 0.356564)
        quad_Q1 = QuadrupoleBasic(1, self.ks[1])
        drift2 = DriftBasic(0.286595 + 0.754657 + 0.183182)
        quad_Q2 = QuadrupoleBasic(1, self.ks[2])
        drift3 = DriftBasic(0.286595 + 0.056305 + 3.177152733425373)
        dipole = DipoleSpectrometerBasic(0.9779, self.B_dip, True)
        drift4 = DriftBasic(0.357498 + 1.22355893395943 + 1.36 + 1.13 + 0.5 + 4.26)
        
        return Beamline([drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4])
        
        
    def length(self):
        return self.beamline().length()
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return self.beamline().track(beam, savedepth, runnable, verbose)
    
    
        
    
    # TODO: function to calculate transfer matrix (based on k values)
    def transferMatrix(self, quadrupole_0, quadrupole_1, quadrupole_2, driftspace_0, driftspace_1,driftspace_2,driftspace_3):
        TransferMatrixQuad_0 = quadrupole_0.transferMatrix()
        TransferMatrixQuad_1 = quadrupole_1.transferMatrix()
        TransferMatrixQuad_2 = quadrupole_2.transferMatrix()
        TransferMatrixDrift_0 = driftspace_0.transferMatrix()
        TransferMatrixDrift_1 = driftspace_1.transferMatrix()
        TransferMatrixDrift_2 = driftspace_2.transferMatrix()
        TransferMatrixDrift_3 = driftspace_3.transferMatrix()
        
        TotTransFormVector = multi_dot([TransferMatrixDrift_3, TransferMatrixQuad_2, TransferMatrixDrift_2, TransferMatrixQuad_1, TransferMatrixDrift_1, TransferMatrixQuad_0, TransferMatrixDrift_0])
        
        return TotTransFormVector
    
        
    def minimizingFunction(self, ks):
        k_0, k_1, k_2 = ks
        drift_0 = QuadrupoleBasic(1, 0)
        quad_0 = QuadrupoleBasic(1, k_0)
        drift_1 = QuadrupoleBasic(1, 0)
        quad_1 = QuadrupoleBasic(1, k_1)
        drift_2 = QuadrupoleBasic(1, 0)
        quad_2 = QuadrupoleBasic(1, k_2)
        drift_3 = QuadrupoleBasic(1, 0)
        transferMatrix = self.transferMatrix(quad_0, quad_1, quad_2, drift_0, drift_1, drift_2, drift_3)
        minimizeFunction = (transferMatrix[0,1])**2 + (transferMatrix[2,3])**2 + (transferMatrix[0,0]-5)**2
        
        return minimizeFunction
        
    def minimize(self, x0):
        res = minimize(self.minimizingFunction, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

        return res
        
      #  return 0
    
            
    # TODO: function to find imaging condition (m12 = m34 = 0, m11 = something, for a given energy and object plane)
        