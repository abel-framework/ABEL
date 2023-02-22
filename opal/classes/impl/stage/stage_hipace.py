from opal import Stage

class StageHiPACE(Stage):
    
    def __init__(self, deltaE=None, L=None, n0=None, driverSource=None, enableBetatron=True, addDriverToBeam=False):
        self.deltaE = deltaE
        self.L = L
        self.n0 = n0
        self.enableBetatron = enableBetatron
        self.addDriverToBeam = addDriverToBeam
        self.driverSource = driverSource

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        # TODO: use HiPACE API
        
        return super().track(beam, savedepth, runnable, verbose)
    
        
    def length(self):
        return self.L
    
    def energyGain(self):
        return self.deltaE
    
    def energyEfficiency(self):
        return None # TODO