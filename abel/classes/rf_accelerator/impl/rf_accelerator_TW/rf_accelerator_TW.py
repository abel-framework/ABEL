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
    def __init__(self, RF_structure, length=None, num_structures=None, nom_energy_gain=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
        """
        Initializes a rf_accelerator_TW object.
        Accepts same arguments rf_accelerator + an RF_structure CLICopti object.

        Parameters
        ----------

        RF_structure : CLICopti.RFstructure.Accelstructure object
            The underlying AccelStructure object, typically provided by the implementing subclass.
            Must be specified.
        """
        #Finalizing the RFstructure initialization
        if RF_structure == None:
            raise ValueError("Must set RF structure; this is typically done from an implementing daugther class")
        self._RF_structure = RF_structure
        self._num_integration_points = None
        self.set_num_integration_points(1000)

        #Set for not-handling in base class
        self._structure_length = None

        #Initialize it
        self._beam_charge = None

        super().__init__(length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain, \
                         bunch_separation=bunch_separation, num_bunches_in_train=num_bunches_in_train, rep_rate_trains=rep_rate_trains)

    #---------------------------------------------------------------------#
    # Override some of the properties from the parent class RFAccelerator #
    #=====================================================================#

    @property
    def structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self._RF_structure.getL()
    @structure_length.setter
    def structure_length(self, structure_length : float):
        #This is derived from the number of cells, the phase advance, and the frequency
        raise NotImplementedError("Not possible to set directly with RFAccelerator_TW")

    @property
    def rf_frequency(self) -> float:
        "The RF frequency of the RF structures [1/s]"
        return self._RF_structure.getOmega()*2*np.pi
    @rf_frequency.setter
    def rf_frequency(self, rf_frequency):
        raise NotImplementedError("Not possible to set directly with RFAccelerator_TW")

    def energy_usage(self):
        "Energy usage per bunch [J]"
        return self.get_structure_pulse_energy() * self.get_num_structures() * self.efficiency_wallplug_to_rf / self.num_bunches_in_train

    #---------------------------------------------------------------------#
    # CLICopti-based modelling                                            #
    #=====================================================================#

    def set_num_integration_points(self,N : int):
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
        return (self._RF_structure.getTrise() + self.get_t_fill() + self.get_beam_pulse_length()) * \
            self.get_structure_power()

    @property
    def beam_charge(self) -> float:
        """
        The total charge of the beam [C]
        Can be set directly for power calculations,
          or implicitly by track()
        """
        if self._beam_charge is None:
            raise abel.RFAcceleratorInitializationException("Beam charge not initialized")
        return self._beam_charge
    @beam_charge.setter
    def beam_charge(self,beam_charge : float):
        self._beam_charge = beam_charge

    @property
    def bunch_charge(self) -> float:
        """
        The total charge of one bunch [C]
        Calculated from beam_charge and num_bunches_in_train
        When set, only beam_charge is changed
        """
        return self.beam_charge/self.num_bunches_in_train
    @bunch_charge.setter
    def bunch_charge(self, bunch_charge : float):
        self.beam_charge = bunch_charge*self.num_bunches_in_train

    @property
    def beam_current(self) -> float:
        """
        The beam current of the linac [A].
        Calculated from bunch_charge and get_beam_pulse_length()
        When set, only beam_charge is changed.
        When the number of bunches is train duration is 0 (num bunches = 1),
        the beam_current is effectively 0 for e.g. beam loading purposes.
        """
        if self.get_beam_pulse_length() == 0.0:
            return 0.0
        return self.beam_charge / self.get_beam_pulse_length()
    @beam_current.setter
    def beam_current(self, beam_current : float) -> None:
        self.beam_charge = beam_current * self.get_beam_pulse_length()

    def get_structure_power(self) -> float:
        "Get the peak power [W] required for a single RF structure in the given configuration."
        if self.beam_current == 0.0:
            P = self._RF_structure.getPowerUnloaded(self.voltage_structure)
        else:
            P = self._RF_structure.getPowerLoaded(self.voltage_structure, self.beam_current)
        return P

    def get_RF_efficiency(self) -> float:
        """
        Get the RF->beam efficiency as a number between 0 and 1, including the effect due to pulse shape
        Calculated with different methods depending on wehter num_bunches_in_train = 1 or > 1.
        """
        if self.get_beam_pulse_length() == 0.0:
            return self.voltage_structure*self.beam_charge / self.get_structure_pulse_energy()
        else:
            return self._RF_structure.getTotalEfficiency(self.get_structure_power(), self.beam_current, self.get_beam_pulse_length())

    def get_RF_efficiency_flattop(self) -> float:
        "Get the RF->beam efficiency as a number between 0 and 1, ignoring the fill time etc."
        return self._RF_structure.getFlattopEfficiency(self.get_structure_power(), self.beam_current)

    def get_beam_pulse_length(self) -> float:
        """
        Gets the beam pulse length (=length of RF pulse flat top) [s] for the RF structues,
        or 0.0 if the bunch separation is not set
        """
        if self.bunch_separation is None:
            if self.num_bunches_in_train != 1:
                #TODO Move this to trackable
                raise ValueError("Invalid bunch separation but multiple bunches!")
            return 0.0
        bpl = self.get_train_duration()
        if bpl is None:
            raise abel.RFAcceleratorInitializationException("trackable.get_train_duration returned None")
        return bpl

    def get_t_fill(self) -> float:
        "Get the filling time of the structure, i.e. the time for a signal to propagate through the whole structure."
        return self._RF_structure.getTfill()

    def get_pulse_length_total(self) -> float:
        return 2 * (self._RF_structure.getTrise() + self._RF_structure.getTfill()) \
                 + self.get_beam_pulse_length()

    def get_max_pulse_length(self) -> float:
        "Calculates the max beam_pulse_length before exceeding gradient limits, given power and beam_current"
        return self._RF_structure.getMaxAllowableBeamTime(self.get_structure_power(), self.beam_current)

    def set_pulse_length_max(self) -> float:
        "Sets the pulse length to the maximally achievable given the currently selected gradient and beam current"
        #Obsolete, set_beam_pulse_length is dead. Change the number of bunches from upstream instead!
        self.set_beam_pulse_length( self.get_max_pulse_length() )
        return self.get_beam_pulse_length()

    def get_gradient_max(self) -> float:
        "Calculates the max gradient (and thus voltage) before exceeding breakdown limts, given the currently selected pulse length and beam current"
        return self._RF_structure.getMaxAllowablePower_beamTimeFixed(self.beam_current, self.get_beam_pulse_length())

    def set_gradient_max(self) -> float:
        "Sets the gradient to the maximally achievable given the currently selected pulse length and beam current"
        self.set_gradient( self.get_gradient_max() )
        return self.get_gradient()

    #---------------------------------------------------------------------#
    # Plots (CLICopti-based)                                              #
    #=====================================================================#

    def plot_gradient_profile(self) -> float:
        z = self._RF_structure.getZ_all()
        if self.beam_current == None:
            Ez = self._RF_structure.getEz_unloaded_all(self.get_structure_power())
        else:
            Ez = self._RF_structure.getEz_loaded_all(self.get_structure_power(), self.beam_current)

        plt.subplots()
        plt.plot(z*1e3,Ez/1e6)
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$E_z$ [MV/m]')

        plt.title(self.make_plot_title())

    def plot_power_profile(self) -> float:
        t = np.linspace(0,self.get_pulse_length_total(), 100)
        P = self._RF_structure.getP_t(t, self.get_structure_power(),\
                                         self.get_beam_pulse_length(),\
                                         self.beam_current )

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
        tit += f", I={self.beam_current:.1f} [A]"
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