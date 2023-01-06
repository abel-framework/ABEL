from opal import Spectrometer, Beamline, DriftBasic, DipoleSpectrometerBasic, QuadrupoleVer, QuadrupoleHor

class SpectrometerFACET2Basic(Spectrometer):
    
    def __init__(self, B_dip = None, E_img = None, s_obj = None):
        self.B_dip = B_dip        
        self.E_img = E_img
        self.s_obj = s_obj

    def beamline(self):
        drift1 = DriftBasic(2)
        dipole = DipoleSpectrometerBasic(1, self.B_dip, True)
        drift2 = DriftBasic(6)
        quadVer = QuadrupoleVer(1,2)
        quadHor = QuadrupoleHor(1,2)
        
        return Beamline([drift1, dipole, drift2,quadVer, quadHor])
        
        
    def length(self):
        return self.beamline().length()
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return self.beamline().track(beam, savedepth, runnable, verbose)
        