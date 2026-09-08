#!/usr/bin/env python3

# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later


import CLICopti
import abel

class RFAccelerator_TW_CLIC502(abel.RFAccelerator_TW):
    """
    Class implementing a CLIC_502 type RF structure

    Parameters:
    ===========

    num_rf_cells : int
        The number of accelerating cells in the modelled RF structure

    rf_frequency : float
        The frequency the structure is operating at [Hz]
    """

    def __init__(self, num_rf_cells=24, rf_frequency=11.9942, \
                 length=None, num_structures=None, nom_energy_gain=None):
        if type(num_rf_cells) != int:
            raise TypeError("num_rf_cells must be an int")

        structure = self._make_structure(num_rf_cells=num_rf_cells, rf_frequency=rf_frequency)

        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain )

    def _make_structure(self, num_rf_cells=None,rf_frequency=None):
        return CLICopti.RFStructure.AccelStructure_CLIC502(num_rf_cells, f0_scaleto=rf_frequency/1e9)
        
    def make_structure_title(self):
        tit = "CLIC502"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.rf_frequency/1e9:.1f} [GHz]"
        return tit