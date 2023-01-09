from opal import Spectrometer, Beamline, DriftBasic, DipoleSpectrometerBasic, QuadrupoleVer, QuadrupoleHor

class SpectrometerFACET2Basic(Spectrometer):
    
    def __init__(self, B_dip = None, E_img = None, s_obj = None):
        self.B_dip = B_dip        
        self.E_img = E_img
        self.s_obj = s_obj
        
        self.ks = [-0.330432, 0.54533, -0.330432] # [m^-2]

    def beamline(self):
        
        # starts at z = 1998.708952 m
        drift0 = DriftBasic(0.202439)
        quad_Q0 = QuadrupoleHor(1, -self.ks[0])
        drift1 = DriftBasic(0.286595 + 0.580966 + 0.356564)
        quad_Q1 = QuadrupoleVer(1, self.ks[1])
        drift2 = DriftBasic(0.286595 + 0.754657 + 0.183182)
        quad_Q2 = QuadrupoleHor(1, -self.ks[2])
        drift3 = DriftBasic(0.286595 + 0.056305 + 3.177152733425373)
        dipole = DipoleSpectrometerBasic(0.9779, self.B_dip, True)
        drift4 = DriftBasic(0.357498 + 1.22355893395943 + 1.36 + 1.13 + 0.5 + 4.26)
        
        return Beamline([drift0, quad_Q0, drift1, quad_Q1, drift2, quad_Q2, drift3, dipole, drift4])
        
        
    def length(self):
        return self.beamline().length()
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return self.beamline().track(beam, savedepth, runnable, verbose)
    
    # TODO: function to calculate transfer matrix (based on k values)
    
    # TODO: function to find imaging condition (m12 = m34 = 0, m11 = something, for a given energy and object plane)
        