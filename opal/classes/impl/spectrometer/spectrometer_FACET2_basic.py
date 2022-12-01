from opal import Spectrometer

class SpectrometerFACET2Basic(Spectrometer):
    
    def __init__(self, B_spec = None, E_img = None, s_obj = None):
        self.B_spec = B_spec
        self.E_img = E_img
        self.s_obj = s_obj
        self.L = 10
        
    def length(self):
        if callable(self.L):
            return self.L(self.E0)
        else:
            return self.L
        
    def track(self, beam):
        
        # increment beam location
        beam.location += self.length()
        
        return super().track(beam)