from abel.classes.spectrometer.quad_imaging.impactx import SpectrometerQuadImagingImpactX

class SpectrometerFACET2(SpectrometerQuadImagingImpactX):
    
    def __init__(self, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=0.0, object_plane_y=0.0, magnification_x=-10.0):
        
        super().__init__(imaging_energy_x=imaging_energy_x, imaging_energy_y=imaging_energy_y, object_plane_x=object_plane_x, object_plane_y=object_plane_y, magnification_x=magnification_x)
        
        self.angle_dipole = -6e-3 # [rad]
        self.length_dipole = 0.978 # [m]

        self.strengths_quads = [-5,10,-10]
        self.length_quad = 1.0
        
        self.length_drifts = [1.897, 1.224, 1.224, 3.520, 8.831]
    
        
        