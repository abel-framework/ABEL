from abel import Source
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
        return beam1 + beam2
    
    
    def get_length(self):
        return max(self.source1.get_length(), self.source2.get_length())
    
    def get_charge(self):
        return self.source1.get_charge() + self.source2.get_charge()
        
    def get_energy(self):
        return (self.source1.get_energy()*self.source1.get_charge() + self.source2.get_energy()*self.source2.get_charge())/self.get_charge()
    
    def energy_efficiency(self):
        Etot1 = self.source1.get_energy()*self.source1.get_charge()
        Etot2 = self.source2.get_energy()*self.source2.get_charge()
        return (self.source1.energy_efficiency()*Etot1 + self.source2.energy_efficiency()*Etot2)/(Etot1+Etot2)
    
    