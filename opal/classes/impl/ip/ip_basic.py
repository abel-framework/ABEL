from opal import InteractionPoint, Event
from opal.utilities import SI
import numpy as np

class InteractionPointBasic(InteractionPoint):
    
    # perform a simple interaction (geometric lumosity)
    def interact(self, beam1, beam2):
        
        # create event
        event = Event(beam1, beam2)
        
        # geometric factor
        H_D = 1
        
        # find overlapping area
        sigx = np.sqrt(beam1.beamSizeX()*beam2.beamSizeX())
        sigy = np.sqrt(beam1.beamSizeY()*beam2.beamSizeY())
        
        # get charge
        N1 = abs(beam1.charge()/SI.e)
        N2 = abs(beam2.charge()/SI.e)
        
        # calculate the geometric luminosity (per crossing)
        lumi = H_D / (4*np.pi) * N1 * N2 / (sigx * sigy)
        
        # save to event
        event.luminosity_full = lumi
        event.luminosity_peak = lumi
        event.luminosity_geom = lumi
        
        return event
        