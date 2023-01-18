from opal import Source
import numpy as np

class SourceCombiner(Source):
    
    def __init__(self, source1, source2):
        self.source1 = source1
        self.source2 = source2
    
    
    def track(self, _ = None, savedepth=0, runnable=None, verbose=False):
        
        # generate beams
        beam1 = self.source1.track(None, savedepth, runnable, verbose)
        beam2 = self.source2.track()
        
        # combine beams
        beam1.addBeam(beam2)
        
        return beam1
    
    
    def length(self):
        return max(self.source1.length(), self.source2.length())
    
    
    def energy(self):
        return np.mean(self.source1.energy(), self.source2.energy())