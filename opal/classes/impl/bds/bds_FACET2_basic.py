from opal import BeamDeliverySystem

class BeamDeliverySystemFACET2Basic(BeamDeliverySystem):
    
    def __init__(self, beta_waist = None, s_waist = None, L = 10):
        self.beta_waist = beta_waist
        self.s_waist = s_waist
        self.L = L
        
    def length(self):
        return self.L
        
    def track(self, beam):

        # increment beam location
        beam.location += self.length()
        
        return super().track(beam)
    