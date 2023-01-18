from opal import Spectrometer, Beamline, DriftBasic, DipoleSpectrometerBasic, QuadrupoleBasic
import numpy as np
from numpy.linalg import multi_dot
import scipy.optimize as sciopt

class SpectrometerFACET2Basic(Spectrometer):
    
    def __init__(self, B_dip = None, E_img = None, s_obj = 0, mag_x = -8, E_img_y = None, s_obj_y = None):
        self.B_dip = B_dip        
        self.E_img = E_img
        self.s_obj = s_obj
        self.mag_x = mag_x
        self.E_img_y = E_img_y
        self.s_obj_y = s_obj_y
        
        self.ks = [0, 0, 0] # [m^-2]

    def beamline(self, ks=None, s_obj=None):
        
        # override quad strengths and object plane if given
        if ks is None:
            ks = self.ks
        if s_obj is None:
            s_obj = self.s_obj
            
        # starts at plasma exit: z = 1997.0195 m - object plane
        drift0 = DriftBasic(0.202439 + 1.695 - s_obj)
        quad_Q0 = QuadrupoleBasic(1, ks[0], self.E_img)
        drift1 = DriftBasic(0.286595 + 0.580966 + 0.356564)
        quad_Q1 = QuadrupoleBasic(1, ks[1], self.E_img)
        drift2 = DriftBasic(0.286595 + 0.754657 + 0.183182)
        quad_Q2 = QuadrupoleBasic(1, ks[2], self.E_img)
        drift3 = DriftBasic(0.286595 + 0.056305 + 3.177152733425373)
        dipole = DipoleSpectrometerBasic(0.9779, self.B_dip, True)
        drift4 = DriftBasic(0.357498 + 1.22355893395943 + 1.36 + 1.13 + 0.5 + 4.26)
        
        return Beamline([drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4])
        
        
    def length(self):
        return self.beamline(s_obj=0).length()
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return self.beamline(s_obj=0).track(beam, savedepth, runnable, verbose)
    
    
    # function to calculate transfer matrix (based on k values)
    def transferMatrix(self, ks=None, s_obj=None):

        # override quad strengths if given
        if ks is None:
            ks = self.ks
        if s_obj is None:
            s_obj = self.s_obj
        
        # calculate full transfer matrix
        Rtot = np.eye(4)
        for trackable in self.beamline(ks).trackables:
            R = trackable.transferMatrix()
            Rtot = multi_dot([R, Rtot])
            
        return Rtot
    
        
    def __imagingCondition(self, ks):
        
        # set vertical plane if not already
        if self.E_img_y is None:
            self.E_img_y = self.E_img
        if self.s_obj_y is None:
            self.s_obj_y = self.s_obj
        
        # calculate transfer matrices
        Rx = self.transferMatrix(ks, self.s_obj)
        Ry = self.transferMatrix(ks*self.E_img/self.E_img_y, self.s_obj_y)
        
        # return object function
        return (Rx[0,1])**2 + (Ry[2,3])**2 + (Rx[0,0]-self.mag_x)**2
        
    
    def setImaging(self):
        
        # perform minization (find k-values)
        ks0 = [-0.3, 0, 0.3]
        result = sciopt.minimize(self.__imagingCondition, ks0, tol=1e-5, options={'maxiter': 1000})
        
        # set solution to quads
        if result.fun < 1e-5:
            self.ks = result.x
        else:
            raise Exception('No imaging solution found.')
        