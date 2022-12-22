import numpy as np

class Event():
    
    # empty beam
    def __init__(self, input_beam1, input_beam2):
        
        self.input_beam1 = input_beam1
        self.input_beam2 = input_beam2
        
        # luminosity spectrum
        self.luminosity_full = None
        self.luminosity_geom = None
        self.luminosity_peak = None
    
    
    # calculate center of mass energy
    def centerOfMassEnergy(self):
        E1 = self.input_beam1.energy()
        E2 = self.input_beam2.energy()
        s = 4*E1*E2
        return np.sqrt(s)
    
    
    def geometricLuminosity(self):
        return self.luminosity_geom
    
    def fullLuminosity(self):
        return self.luminosity_full
    
    def peakLuminosity(self):
        return self.luminosity_peak
    
    
    # save event (from OpenPMD format)
    def save(runnable):
        pass
    
    # load event (from OpenPMD format)
    @classmethod
    def load(filename):
        return None # TODO: implement
        
    