# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.ip.ip import InteractionPoint

class InteractionPointBasic(InteractionPoint):

    def __init__(self):
        super().__init__()
    
    # perform a simple interaction (geometric lumosity)
    def interact(self, beam1, beam2):

        import numpy as np
        import scipy.constants as SI
        from abel.classes.event import Event
        
        # geometric factor
        H_D = 1
        
        # find overlapping area
        sigx = np.sqrt(beam1.beam_size_x()*beam2.beam_size_x())
        sigy = np.sqrt(beam1.beam_size_y()*beam2.beam_size_y())
        
        # get charge
        N1 = abs(beam1.charge()/SI.e)
        N2 = abs(beam2.charge()/SI.e)
            
        # calculate the geometric luminosity (per crossing)
        lumi = H_D / (4*np.pi) * N1 * N2 / (sigx * sigy)

        # if gamma-gamma collision, reduce by conversion factor squared
        if self.gamma_gamma:
            compton_conversion_efficiency = 0.45
            lumi = lumi * compton_conversion_efficiency**2
        
        # create event
        event = Event()
        
        # save beams
        event.input_beam1 = beam1
        event.input_beam2 = beam2
        event.output_beam1 = beam1
        event.output_beam2 = beam2
        
        # save to event
        event.luminosity_full = lumi
        event.luminosity_peak = lumi
        event.luminosity_geom = lumi

        return event
        