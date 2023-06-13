from opal import BeamDeliverySystem, Beamline, DriftBasic

class BeamDeliverySystemFACET2Basic(BeamDeliverySystem):
    
    def __init__(self, beta_waist=None, s_waist=None):
        self.beta_waist = beta_waist
        self.s_waist = s_waist
        
    def beamline(self):
        drift1 = DriftBasic(0.1) # zero length for now
        return Beamline([drift1])
        
    def get_length(self):
        return self.beamline().get_length()
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return self.beamline().track(beam, savedepth, runnable, verbose)
    
    