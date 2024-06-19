#!/usr/bin/env python3

import CLICopti
import abel

class RFAccelerator_TW_CLICG(abel.RFAccelerator_TW):
    "Class implementing a CLIC_G type RF structure"

    def __init__(self, num_rf_cells=24, R05=True, rf_frequency=11.9942e9, length=None, num_structures=None, nom_energy_gain=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
        if type(num_rf_cells) != int:
            raise TypeError("num_rf_cells must be an int")
        self.num_rf_cells = num_rf_cells

        if type(R05) != bool:
            raise TypeError("R05 must be a bool")
        self.R05 = R05

        structure = CLICopti.RFStructure.AccelStructure_CLICG(num_rf_cells, isR05=R05, f0_scaleto=rf_frequency/1e9)

        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain, \
                         bunch_separation=bunch_separation, num_bunches_in_train=num_bunches_in_train, rep_rate_trains=bunch_separation)

    def make_structure_title(self):
        tit = "CLIC_G"
        if self.R05:
            tit += "_R05"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.f0:.1f} [GHz]"
        return tit