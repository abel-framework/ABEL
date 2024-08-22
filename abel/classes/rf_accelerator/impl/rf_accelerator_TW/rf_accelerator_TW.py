#!/usr/bin/env python3

from abc import abstractmethod

import abel

import os
os.environ['CLICOPTI_NOSPLASH'] = 'YES' #Suppress splash
import CLICopti

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SI

class RFAccelerator_TW(abel.RFAccelerator):
    """
    Abstract rf_accelerator implementation for calculating parameters
    using models for Traveling Wave Structures, wrapping the CLICopti library.
    Several classes with different constructors are available
    for generating different structure geometries.
    """
    @abstractmethod
    def __init__(self, RF_structure, length=None, num_structures=None, nom_energy_gain=None):
        """
        Initializes a rf_accelerator_TW object.
        Accepts same arguments rf_accelerator + an RF_structure CLICopti object.

        Parameters
        ----------

        RF_structure : CLICopti.RFstructure.Accelstructure object
            The underlying AccelStructure object, typically provided by the implementing subclass.
            Must be specified.
        """

        #Set for not-handling in base class
        self._structure_length = None

        #Set this to true to always optimize the gradient, number of structures, and overall length on track()
        self.autoOptimize = True
        self.autoOptimize_targetFillFactor = 0.71

        self._num_integration_points = 1000
        self._initialize_RF_structure(RF_structure)

        super().__init__(length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain)

    def _initialize_RF_structure(self, RF_structure):
        "Finalizing the RFstructure initialization, to be called by __init__ or by subclasses as needed"
        if RF_structure == None:
            raise ValueError("Must set RF structure; this is typically done from an implementing daugther class")
        self._RF_structure = RF_structure
        
        self.set_num_integration_points(self.get_num_integration_points())

    def _checkType_or_getOld(self, r, name, typeWanted=float, nameInCLICopti=None, scaleFromCLICopti=None, firstCall=False):
        """
        Helper for _make_structure() in the child classes.
        If r is None return the old value from self._RFstructure,
        otherwise check data type and if possible/needed convert and scale.
        """

        if nameInCLICopti == None:
            nameInCLICopti = name

        if firstCall == True and r == None:
            raise ValueError("Must set " + name + " on first initialization")
        if r == None and firstCall == False:
            r = getattr(self._RF_structure, nameInCLICopti)
            if scaleFromCLICopti != None:
                r *= scaleFromCLICopti
            return r

        if type(r) != typeWanted:
            if typeWanted == float:
                if type(r) == int:
                    r = float(r)
                else:
                    raise TypeError(name + " must be a float (can convert ints)")
            elif typeWanted == int:
                if type(r) != int:
                    raise TypeError(name + " must be an int")
            else:
                raise TypeError("For autoconverting, typeWanted must be float or int, got " + str(typeWanted))
        return r

    @abstractmethod
    def _make_structure(self, num_rf_cells=None, rf_frequency=None):
        """
        Method used by subclasses to (re)generate the _RF_structure object.
        Several arguments are generally provided, at least num_rf_cells [int] and rf_frequency [Hz].
        """
        raise NotImplementedError("Must be implemented in child classes")

    #---------------------------------------------------------------------------------#
    # Override some of the properties and methods from the parent class RFAccelerator #
    #=================================================================================#

    @property
    def structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        #This is related to the number of cells, the phase advance, and the frequency
        return self._RF_structure.getL()
    @structure_length.setter
    def structure_length(self, structure_length : float):
        #Indirectly possible through num_rf_cells.setter
        raise NotImplementedError("Not possible to set directly with RFAccelerator_TW")

    @property
    def rf_frequency(self) -> float:
        "The RF frequency of the RF structures [Hz]"
        return self._RF_structure.getF0()
    @rf_frequency.setter
    def rf_frequency(self, rf_frequency):
        self._make_structure(rf_frequency=rf_frequency)

    def energy_usage(self):
        "Energy usage per bunch [J]"
        return self.get_structure_pulse_energy() * self.get_num_structures() * self.efficiency_wallplug_to_rf / self.num_bunches_in_train

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)

    #---------------------------------------------------------------------#
    # CLICopti-based modelling                                            #
    #=====================================================================#

    @property
    def num_rf_cells(self) -> int:
        "The number of RF structure cells"
        return self._RF_structure.N
    @num_rf_cells.setter
    def num_rf_cells(self, num_rf_cells : int):
        if type(num_rf_cells) != int:
            raise TypeError("num_rf_cells must be an integer")
        self._make_structure(num_rf_cells=num_rf_cells)

    def set_num_integration_points(self,N : int):
        if type(N) != int:
            raise TypeError("Number of integration points must be an integer")
        self._num_integration_points = N
        self._RF_structure.calc_g_integrals(self._num_integration_points)
    def get_num_integration_points(self):
        return self._num_integration_points

    def get_structure_pulse_energy(self) -> float:
        "Get the energy requirements for a single pulse for one structure"
        #Note: Model is assuming a pulse like from plotPowerProfile(),
        #      which assumes a drive beam driven pulse in that pulse end is a mirror of the start,
        #      and that the pusle shape can be easily "reorganized" to a square pulse.
        #      As long as Tfill is short, this doesn't matter.
        return (self._RF_structure.getTrise() + self.get_t_fill() + self.train_duration) * \
            self.get_structure_power()

    def get_structure_power(self) -> float:
        "Get the peak power [W] required for a single RF structure in the given configuration."
        if self.num_bunches_in_train == 1:
            P = self._RF_structure.getPowerUnloaded(self.voltage_structure)
        else:
            P = self._RF_structure.getPowerLoaded(self.voltage_structure, self.average_current_train)
        return P

    def get_RF_efficiency(self) -> float:
        """
        Get the RF->beam efficiency as a number between 0 and 1, including the effect due to pulse shape
        Calculated with different methods depending on wehter num_bunches_in_train = 1 or > 1.
        """
        if self.num_bunches_in_train == 1:
            return self.voltage_structure*self.beam_charge / self.get_structure_pulse_energy()
        else:
            return self._RF_structure.getTotalEfficiency(self.get_structure_power(), self.average_current_train, self.train_duration)

    def get_RF_efficiency_flattop(self) -> float:
        "Get the RF->beam efficiency as a number between 0 and 1, ignoring the fill time etc."
        if self.average_current_train == None:
            raise ValueError("Undefined for single bunch trains")
        return self._RF_structure.getFlattopEfficiency(self.get_structure_power(), self.average_current_train)

    def get_t_fill(self) -> float:
        "Get the filling time [s] of the structure, i.e. the time for a signal to propagate through the whole structure."
        return self._RF_structure.getTfill()

    def get_pulse_length_total(self) -> float:
        "Get the total RF pulse length [s], including rise time, filling time, beam time, and rampdown (drive-beam style)"
        return 2 * (self._RF_structure.getTrise() + self._RF_structure.getTfill()) \
                 + self.train_duration

    def get_pulse_length_max(self) -> float:
        "Calculates the max train duration [s] before exceeding gradient limits, given power and average_current_train [A]"
        if self.num_bunches_in_train == 1:
            return self._RF_structure.getMaxAllowableBeamTime(self.get_structure_power(), 0.0)
        else:
            return self._RF_structure.getMaxAllowableBeamTime(self.get_structure_power(), self.average_current_train)

    def get_structure_voltage_max(self) -> float:
        "Calculates the maximum structure voltage before exceeding breakdown limts, given the currently selected pulse length and beam current"
        Vmax = None
        if self.num_bunches_in_train == 1:
            Pmax = self._RF_structure.getMaxAllowablePower_beamTimeFixed(0.0, self.train_duration)
            Vmax = self._RF_structure.getVoltageUnoaded(Pmax)
        else:
            Pmax = self._RF_structure.getMaxAllowablePower_beamTimeFixed(self.average_current_train, self.train_duration)
            Vmax = self._RF_structure.getVoltageLoaded(Pmax, self.average_current_train)
        return Vmax
    
    def optimize_linac_geometry_and_gradient(self,fill_factor=1.0):
        """
        Find the right structure voltage using get_structure_voltage_max(), then set the number of structures
        and the overall linac length so that the total energy gain is respected.

        Returns (Vmax, Vstruct, Ntotal_int)
        """
        #Get the maximum possible voltage
        Vmax = self.get_structure_voltage_max()
        #TODO: Handle Vmax = 0.0
        Vtotal = self.nom_energy_gain
        Ntotal = Vtotal/Vmax
        
        Ntotal_int = int(np.ceil(Ntotal))
        Vstruct    = Vtotal/Ntotal_int

        length=Ntotal_int*self.structure_length/fill_factor

        #Save it
        self.length          = length
        self.num_structures  = Ntotal_int
        return (Vmax, Vstruct, Ntotal_int)

    #---------------------------------------------------------------------#
    # Plots (CLICopti-based)                                              #
    #=====================================================================#

    def plot_gradient_profile(self) -> float:
        z = self._RF_structure.getZ_all()
        if self.average_current_train == None:
            Ez = self._RF_structure.getEz_unloaded_all(self.get_structure_power())
        else:
            Ez = self._RF_structure.getEz_loaded_all(self.get_structure_power(), self.average_current_train)

        plt.subplots()
        plt.plot(z*1e3,Ez/1e6)
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$E_z$ [MV/m]')

        plt.title(self.make_plot_title())

    def plot_power_profile(self) -> float:
        t = np.linspace(0,self.get_pulse_length_total(), 100)
        P = self._RF_structure.getP_t(t, self.get_structure_power(),\
                                         self.train_duration,\
                                         self.average_current_train )

        plt.subplots()
        plt.plot(t*1e9,P/1e6)
        plt.xlabel('Time [ns]')
        plt.ylabel('$P_{in}$ [MW]')
        plt.title(self.make_plot_title())

    def make_plot_title(self) -> str:
        "Create a standardized title string for plots"
        #tit =  f"{type(self._RF_structure).__name__}"
        tit = self.make_structure_title()
        tit += "\n"
        tit += f"V={self.voltage_structure/1e6:.1f} [MV]"
        tit += f", I={self.average_current_train:.1f} [A]"
        tit += f", <G>={self.gradient_structure/1e6:.1f} [MV/m]"
        return tit

    @abstractmethod
    def make_structure_title(self) -> str:
        "Make a 'personalized' title for the implementing RF structure"
        return f"{type(self._RF_structure).__name__}, V={self.voltage_structure/1e6:.1f} [MV]"

    #---------------------------------------------#
    # Methods for cost modelling                  #
    #=============================================#

    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        return 500e3 #Rough number from memory (KNS)