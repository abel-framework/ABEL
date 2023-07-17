from opal import Spectrometer, Beamline, DriftBasic, DipoleSpectrometerBasic, QuadrupoleBasic
import numpy as np
from numpy.linalg import multi_dot
import scipy.optimize as sciopt

class SpectrometerFacetBasic(Spectrometer):
    
    def __init__(self, bend_angle=-0.03, img_energy=None, obj_plane=0, mag_x=-8, img_energy_y=None, obj_plane_y=None):
        
        self.bend_angle = bend_angle        
        self.img_energy = img_energy
        self.obj_plane = obj_plane
        self.mag_x = mag_x
        self.img_energy_y = img_energy_y
        self.obj_plane_y = obj_plane_y
        
        self.ks = [-0.3, 0, 0.3] # [m^-2]

    def beamline(self, ks=None, obj_plane=None):
        
        # override quad strengths and object plane if given
        if ks is None:
            ks = self.ks
        if obj_plane is None:
            obj_plane = self.obj_plane
            
        # starts at plasma exit: z = 1997.0195 m - object plane
        drift0 = DriftBasic(0.202439 + 1.695 - obj_plane)
        quad_Q0 = QuadrupoleBasic(1, ks[0], self.img_energy)
        drift1 = DriftBasic(0.286595 + 0.580966 + 0.356564)
        quad_Q1 = QuadrupoleBasic(1, ks[1], self.img_energy)
        drift2 = DriftBasic(0.286595 + 0.754657 + 0.183182)
        quad_Q2 = QuadrupoleBasic(1, ks[2], self.img_energy)
        drift3 = DriftBasic(0.286595 + 0.056305 + 3.177152733425373)
        dipole = DipoleSpectrometerBasic(0.9779, self.bend_angle, True)
        drift4 = DriftBasic(0.357498 + 1.22355893395943 + 1.36 + 1.13 + 0.5 + 4.26)
        
        return Beamline([drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4])
        
        
    def get_length(self):
        return self.beamline(obj_plane=0).get_length()
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        self.set_imaging()
        return self.beamline(obj_plane=0).track(beam, savedepth, runnable, verbose)
    
    
    # function to calculate transfer matrix (based on k values)
    def transferMatrix(self, ks=None, obj_plane=None):

        # override quad strengths if given
        if ks is None:
            ks = self.ks
        if obj_plane is None:
            obj_plane = self.obj_plane
        
        # calculate full transfer matrix
        Rtot = np.eye(4)
        for trackable in self.beamline(ks).trackables:
            R = trackable.transferMatrix()
            Rtot = multi_dot([R, Rtot])
            
        return Rtot
    
        
    def __calculate_imaging_condition(self, ks):
        
        # set vertical plane if not already
        if self.img_energy_y is None:
            self.img_energy_y = self.img_energy
        if self.obj_plane_y is None:
            self.obj_plane_y = self.obj_plane
        
        # calculate transfer matrices
        Rx = self.transferMatrix(ks, self.obj_plane)
        Ry = self.transferMatrix(ks*self.img_energy_y/self.img_energy, self.obj_plane_y)
        
        # return object function
        return (Rx[0,1])**2 + (Ry[2,3])**2 + (Rx[0,0]-self.mag_x)**2
        
    
    def set_imaging(self):
        
        # perform minization (find k-values)
        result = sciopt.minimize(self.__calculate_imaging_condition, self.ks, tol=1e-5, options={'maxiter': 1000})
        
        # set solution to quads
        if result.fun < 1e-5:
            self.ks = result.x
        else:
            raise Exception('No imaging solution found.')
        