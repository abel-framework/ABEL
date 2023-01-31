from opal import Interstage
from opal.utilities import SI

class InterstageBasic(Interstage):
    
    def __init__(self, E0 = None, Ldip = None, Bdip = None):
        self.E0 = E0
        self.Ldip = Ldip
        self.Bdip = Bdip
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # compress beam
        beam.compress(R56 = self.R56(), E0 = self.E0)

        # flip transverse phase spaces
        beam.flipTransversePhaseSpaces(flipMomenta=True, flipPositions=False)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # evaluate dipole length  
    def dipoleLength(self):
        if callable(self.Ldip):
            return self.Ldip(self.E0)
        else:
            return self.Ldip
    
    
    # evaluate longitudinal dispersion (R56)
    def R56(self):
        return -self.Bdip**2*SI.c**2*self.dipoleLength()**3/(3*self.E0**2)
    
    
    # lattice length
    def length(self):
        return 4.7875*self.dipoleLength()
        