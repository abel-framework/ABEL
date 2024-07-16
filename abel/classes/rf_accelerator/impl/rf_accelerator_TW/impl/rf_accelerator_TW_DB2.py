#!/usr/bin/env python3

import CLICopti
import abel

import matplotlib.pyplot as plt

class RFAccelerator_TW_DB2(abel.RFAccelerator_TW):
    """
    Class implementing an RF structure generated from Database v2,
    i.e. a CLIC-type damped high gradient cell, with linear tapering of iris aperture and thickness.

    Database v2 disregarded the P/C limit, using only Sc.

    Parameters:
    ===========

    num_rf_cells : int
        The number of accelerating cells in the modelled RF structure

    rf_frequency : float
        The frequency the structure is operating at [Hz]

    a_n : float
        Average normalized iris aperture a/lambda,
        where a [distance] is the aperture radius and lambda = c/f
    
    a_n_delta : float
        Difference in a_n of first cell and last cell,
        i.e. positive a_n_delta means that the first aperture is larger
    
    d_n : float
        Average normalized iris thickness d/L,
        where d [distance] is the iris thickness and L=lambda*psi/360 the cell length.
    
    d_n_delta : float
        Difference in d_n of first cell and last cell,
        i.e. positive d_n_delta means that the first cell has a thicker iris.

     length, num_structures, nom_energy_gain :
        See `RFAccelerator` class.
    """

    #Database object is static
    default_frequency = 11.9942e9 #[Hz]
    database = CLICopti.CellBase.CellBase_linearInterpolation_freqScaling(CLICopti.CellBase.celldatabase_TD_12GHz_v2,("a_n","d_n"), default_frequency/1e9)

    def __init__(self, num_rf_cells=24,
                 rf_frequency=default_frequency,
                 a_n=0.110022947942206, a_n_delta=0.016003337882503*2,
                 d_n=0.160233420548558, d_n_delta=0.040208386429788*2,
                 length=None, num_structures=None, nom_energy_gain=None):

        structure = self._make_structure(num_rf_cells, a_n, a_n_delta, d_n, d_n_delta, rf_frequency, constructorCalling=True)

        super().__init__(RF_structure=structure, \
                         length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain)

    def _make_structure(self, num_rf_cells=None, a_n=None, a_n_delta=None, d_n=None, d_n_delta=None, rf_frequency=None, constructorCalling=False):
        """
        (re)Initialize the underlying structure object.
        On first call, must set all arguments, on subsequent calls they can be set individually to change parameters.
        If constructorCalling is False, then finalize the initialization in RFAccelerator_TW (use True when calling from child class constructor).
        """

        num_rf_cells = self._checkType_or_getOld(num_rf_cells, "num_rf_cells", typeWanted=int, nameInCLICopti="N",                 firstCall=constructorCalling)
        a_n          = self._checkType_or_getOld(a_n,          "a_n",                                                              firstCall=constructorCalling)
        a_n_delta    = self._checkType_or_getOld(a_n_delta,    "a_n_delta",                                                        firstCall=constructorCalling)
        d_n          = self._checkType_or_getOld(d_n,          "d_n",                                                              firstCall=constructorCalling)
        d_n_delta    = self._checkType_or_getOld(d_n_delta,    "d_n_delta",                                                        firstCall=constructorCalling)
        rf_frequency = self._checkType_or_getOld(rf_frequency, "rf_frequency", nameInCLICopti="f0_scaleto", scaleFromCLICopti=1e9, firstCall=constructorCalling)

        #print("Making new structure, parameters:", num_rf_cells, a_n, a_n_delta, d_n, d_n_delta, rf_frequency)

        structure = CLICopti.RFStructure.AccelStructure_paramSet2_noPsi(
            RFAccelerator_TW_DB2.database,
            num_rf_cells, a_n, a_n_delta, d_n, d_n_delta, rf_frequency/1e9)

        #DB v2 constructed ignoring P/C limit
        structure.uselimit_PC = False

        if not constructorCalling:
            self._initialize_RF_structure(structure)

        return structure

    @property
    def a_n(self) -> float:
        return self._RF_structure.a_n
    @a_n.setter
    def a_n(self,a_n : float):
        self._make_structure(a_n=a_n)

    @property
    def a_n_delta(self) -> float:
        return self._RF_structure.a_n_delta
    @a_n_delta.setter
    def a_n_delta(self,a_n_delta : float):
        self._make_structure(a_n_delta=a_n_delta)

    @property
    def d_n(self) -> float:
        return self._RF_structure.d_n
    @d_n.setter
    def d_n(self,d_n : float):
        self._make_structure(d_n=d_n)

    @property
    def d_n_delta(self) -> float:
        return self._RF_structure.d_n_delta
    @d_n_delta.setter
    def d_n_delta(self,d_n_delta : float):
        self._make_structure(d_n_delta=d_n_delta)

    # For plots etc

    def make_structure_title(self):
        tit = "DBv2 structure"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.rf_frequency/1e9:.1f} [GHz]"
        tit += f", a_n={self.a_n:.3f} delta={self.a_n_delta:.3f}"
        tit += f", d_n={self.d_n:.3f} delta={self.d_n_delta:.3f}"
        return tit
    
    def plot_database_points(self, bgData=None):
        "Plot the cell parameters in the database"
        pointsGrid,_ = self.database.getGrid_meshgrid()

        plt.scatter(pointsGrid[0],pointsGrid[1], marker='*', color='red')
        plt.xlabel(r'$a/\lambda$')
        plt.ylabel(r'$d/h$')

        c1 = self._RF_structure.getCellFirst()
        c2 = self._RF_structure.getCellMid()
        c3 = self._RF_structure.getCellLast()
        a_n = (c1.a_n, c2.a_n, c3.a_n)
        d_n = (c1.d_n, c2.d_n, c3.d_n)

        plt.plot(a_n, d_n, 's')
        plt.annotate("",(a_n[0],d_n[0]), (a_n[1],d_n[1]),arrowprops=dict(arrowstyle="->"))
        plt.annotate("",(a_n[1],d_n[1]), (a_n[2],d_n[2]),arrowprops=dict(arrowstyle="->"))


