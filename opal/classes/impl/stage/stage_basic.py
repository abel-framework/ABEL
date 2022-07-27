from opal import Stage

class StageBasic(Stage):
    
    def __init__(self, deltaE = None, L = None, chirp = 0):
        self.deltaE = deltaE
        self.L = L
        self.chirp = chirp
       
    def length(self):
        return self.L
    
    def energyGain(self):
        return self.deltaE
    
    def track(self, beam):
        
        # adiabatic damping
        E0 = beam.energy()
        beam.adiabaticDamping(E0, E0+self.deltaE)

        # flip transverse phase spaces
        beam.flipTransversePhaseSpaces()

        # accelerate beam
        beam.accelerate(deltaE = self.deltaE, chirp = self.chirp)

        return super().track(beam)
    