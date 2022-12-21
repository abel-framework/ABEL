from opal import InteractionPoint, Event
from opal.utilities import SI
import numpy as np

class InteractionPointBasic(InteractionPoint):
    
    # constructor (taking IP beta functions)
    def __init__(self, beta_x=1e-2, beta_y=1e-4, frequency=1e4):
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.frequency = frequency
        
    # perform a simple interaction (geometric lumosity)
    def interact(self, beam1, beam2):
        
        # create event
        event = Event(beam1, beam2)
        
        # geometric factor
        H_D = 1
        
        # get beam sizes
        emx1 = beam1.geomEmittanceX()
        emx2 = beam2.geomEmittanceX()
        emy1 = beam1.geomEmittanceY()
        emy2 = beam2.geomEmittanceY()
        
        # find overlapping area
        sigx = np.sqrt(self.beta_x * np.sqrt(emx1 * emx2))
        sigy = np.sqrt(self.beta_y * np.sqrt(emy1 * emy2))
        
        # get charge
        N1 = beam1.charge()/SI.e
        N2 = beam2.charge()/SI.e
        
        # calculate the geometric luminosity
        lumi = self.frequency / (4*np.pi) * H_D * abs(N1 * N2) / (sigx * sigy)
        
        print("Center-of-mass energy = " + str(round(event.centerOfMassEnergy()/1e9, 2)) + " GeV")
        print("Luminosity: " + str(lumi / 1e4) + " cm^-2 s^-1")
        
        return event
        