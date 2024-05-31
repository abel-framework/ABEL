#!/usr/bin/env python3

import CLICopti
import abel

class RFlinac_CLICG(abel.RFlinac):
    "Class implementing a CLIC_G type RF structure"

    def __init__(self, num_rf_cells=24, R05=True, f0_scaleto=11.9942, length=None, num_structures=None, gradient=None, voltage_total=None, beam_pulse_length=None,beam_current=0.0):
        if type(num_rf_cells) != int:
            raise TypeError("num_rf_cells must be an int")
        self.num_rf_cells = num_rf_cells

        if type(R05) != bool:
            raise TypeError("R05 must be a bool")
        self.R05 = R05

        self.f0 = f0_scaleto

        structure = CLICopti.RFStructure.AccelStructure_CLICG(num_rf_cells, isR05=R05, f0_scaleto=f0_scaleto)

        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, gradient=gradient, voltage_total=voltage_total,
                         beam_pulse_length=beam_pulse_length,beam_current=beam_current)

    def make_structure_title(self):
        tit = "CLIC_G"
        if self.R05:
            tit += "_R05"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.f0:.1f} [GHz]"
        return tit