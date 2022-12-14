from opal import Stage

class StageBasic(Stage):
    
    def __init__(self, deltaE = None, L = None, chirp = 0, z0 = 0):
        self.deltaE = deltaE
        self.L = L
        self.chirp = chirp
        self.z0 = z0
       
    def length(self):
        return self.L
    
    def energyGain(self):
        return self.deltaE
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # adiabatic damping
        beam.betatronDamping(self.deltaE)

        # flip transverse phase spaces
        beam.flipTransversePhaseSpaces()

        # accelerate beam
        beam.accelerate(deltaE = self.deltaE, chirp = self.chirp, z0 = self.z0)

        return super().track(beam, savedepth, runnable, verbose)
        
    