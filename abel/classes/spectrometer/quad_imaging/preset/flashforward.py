from abel.classes.spectrometer.quad_imaging.impactx import SpectrometerQuadImagingImpactX

class SpectrometerFACET2(SpectrometerQuadImagingImpactX):
    
    def __init__(self, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=0.0, object_plane_y=0.0, magnification_x=-10.0):
        
        super().__init__(imaging_energy_x=imaging_energy_x, imaging_energy_y=imaging_energy_y, object_plane_x=object_plane_x, object_plane_y=object_plane_y, magnification_x=magnification_x)
        
        self.angle_dipole = -350e-3 # [rad]
        
        # pre-defined lengths
        self.length_dipole = 1.07 # [m]
        self.length_quad = 0.1137 # [m]

        # beam line positions (quad and dipole centers)
        self.s_CELLCENTRE = 209.932261 # [m]
        self.s_Q11FLFDIAG = 210.590261 # [m]
        self.s_Q12FLFDIAG = 210.856261 # [m]
        self.s_Q21FLFDIAG = 211.122261 # [m]
        self.s_Q22FLFDIAG = 211.388261 # [m]
        self.s_Q23FLFDIAG = 211.768178 # [m]
        self.s_dipoleLEMS = 215.686200 # [m]
        self.s_LEMS       = 217.045605 # [m] image plane

        # derived separations
        d1 = self.s_Q11FLFDIAG - self.s_CELLCENTRE - self.length_quad/2;
        d2 = self.s_Q12FLFDIAG - self.s_Q11FLFDIAG - self.length_quad;
        d3 = self.s_Q21FLFDIAG - self.s_Q12FLFDIAG - self.length_quad;
        d4 = self.s_Q22FLFDIAG - self.s_Q21FLFDIAG - self.length_quad;
        d5 = self.s_Q23FLFDIAG - self.s_Q22FLFDIAG - self.length_quad;
        d6 = self.s_dipoleLEMS - self.s_Q23FLFDIAG - self.length_quad/2 - self.length_dipole/2;
        d7 = self.s_LEMS       - self.s_dipoleLEMS - self.length_dipole/2;
        
        self.strengths_quads = [-5,10,-10]
        
        self.length_drifts = [d1, d2, d3, d4, d5, d6, d7]
