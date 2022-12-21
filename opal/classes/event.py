import numpy as np

class Event():
    
    # empty beam
    def __init__(self, input_beam1=None, input_beam2=None):
        
        self.input_beam1 = input_beam1
        self.input_beam2 = input_beam2
        self.output_beam1 = None
        self.output_beam2 = None
        
        # luminosity spectrum
        self.luminosity_spectrum = None
    
    
    # calculate center of mass energy
    def centerOfMassEnergy(self):
        E1 = self.input_beam1.energy()
        E2 = self.input_beam2.energy()
        s = 4*E1*E2
        return np.sqrt(s)
    
    # load beam (from OpenPMD format)
    @classmethod
    def load(filename):
        return None # TODO: implement
        
    