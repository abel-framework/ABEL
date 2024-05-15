#!/usr/bin/env python3

import CLICopti
import abel

class RFlinac_CLICG(abel.RFlinac):
    def __init__(self, num_rf_cells=24, R05=True, f0_scaleto=11.9942, length=None, num_structures=None, gradient=None, beam_pulse_length=None,beam_current=0.0):
        structure = CLICopti.RFStructure.AccelStructure_CLICG(num_rf_cells, isR05=R05, f0_scaleto=f0_scaleto)

        #TODO: Scaling
        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, gradient=gradient, 
                         beam_pulse_length=beam_pulse_length,beam_current=beam_current)
