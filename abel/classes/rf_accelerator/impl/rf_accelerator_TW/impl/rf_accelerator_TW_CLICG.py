#!/usr/bin/env python3

import CLICopti
import abel

class RFAccelerator_TW_CLICG(abel.RFAccelerator_TW):
    """
    Class implementing a CLIC_G type RF structure

    Parameters:
    ===========

    num_rf_cells : int
        The number of accelerating cells in the modelled RF structure

    R05 : bool
        To use the newer .5mm milling radius design?

    rf_frequency : float
        The frequency the structure is operating at [Hz]

    """

    def __init__(self, num_rf_cells=24, R05=True, rf_frequency=11.9942e9,
                 length=None, num_structures=None, nom_energy_gain=None):

        if type(num_rf_cells) != int:
            raise TypeError("num_rf_cells must be an int")
        if type(R05) != bool:
            raise TypeError("R05 must be a bool")

        structure = self._make_structure(num_rf_cells, R05, rf_frequency, constructorCalling=True)

        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain )

    def _make_structure(self, num_rf_cells=None, R05=None, rf_frequency=None, constructorCalling=False):
        num_rf_cells = self._checkType_or_getOld(num_rf_cells, "num_rf_cells", typeWanted=int,  nameInCLICopti="N",                firstCall=constructorCalling)
        R05          = self._checkType_or_getOld(R05,          "R05",          typeWanted=bool, nameInCLICopti="isR05",            firstCall=constructorCalling)
        rf_frequency = self._checkType_or_getOld(rf_frequency, "rf_frequency", nameInCLICopti="f0_scaleto", scaleFromCLICopti=1e9, firstCall=constructorCalling)

        structure = CLICopti.RFStructure.AccelStructure_CLICG(num_rf_cells, f0_scaleto=rf_frequency/1e9)

        if not constructorCalling:
            self._initialize_RF_structure(structure)

        return structure

    @property
    def R05(self) -> bool:
        return self._RF_structure.isR05
    @R05.setter
    def R05(self, R05 : bool):
        raise NotImplementedError("Not possible to set directly")

    def make_structure_title(self):
        tit = "CLIC_G"
        if self.R05:
            tit += "_R05"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.rf_frequency/1e9:.1f} [GHz]"
        return tit