from opal import Interstage

class InterstageBasic(Interstage):
    
    def __init__(self, R56 = None, E0 = None, L = None):
        self.R56 = R56
        self.E0 = E0
        self.L = L
        
    def length(self):
        if callable(self.L):
            return self.L(self.E0)
        else:
            return self.L
        
    def track(self, beam):

        # compress beam
        if callable(self.R56):
            R56 = self.R56(self.E0)
        else:
            R56 = self.R56
        beam.compress(R56 = R56, E0 = self.E0)

        # flip transverse phase spaces
        beam.flipTransversePhaseSpaces()

        # increment beam location
        beam.location += self.length()
        
        return super().track(beam)
    