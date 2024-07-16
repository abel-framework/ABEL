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

        self._num_integration_points = 1000
        self._initialize_RF_structure(RF_structure)

        #Set for not-handling in base class
        self._structure_length = None

        #Initialize the average_current_train and train_duration to None,
        # and get it from Beam once in track().
        # Can also be manually specified.
        # For consistency, it follows how Beam does it even if average current + pulse length
        # is more fundamental for CLICopti modelling.
        self.bunch_charge           = None # [C]
        self.bunch_separation       = None # [s]
        self.num_bunches_in_train   = None # [-]

        #Set this to true to always optimize the gradient, number of structures, and overall length on track()
        self.autoOptimize = True

        super().__init__(length=length, num_structures=num_structures, nom_energy_gain=nom_energy_gain)

    def __str__(self):
        s = abel.RFAccelerator.__str__(self)
        s += f", " + \
             f"bunch_charge={self._bunch_charge}[C], bunch_separation={self._bunch_separation}[s], num_bunches_in_train={self._num_bunches_in_train}"
        return s

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
        
        #Store data used for power-flow modelling
        self._train_duration = beam.train_duration()
        self._average_current_train = np.fabs(beam.average_current_train())
        self._num_bunches_in_train = beam.num_bunches_in_train
        self._bunch_charge = beam.abs_charge()

        if self.autoOptimize:
            #Given the target energy gain, current, and pulse length,
            # set the max gradient and the minimum linac length
            self.optimize_linac_geometry_and_gradient()

        return super().track(beam, savedepth, runnable, verbose)

    #---------------------------------------------------------------------#
    # CLICopti-based modelling                                            #
    #=====================================================================#

    ## Manage caching of average_current_train [A], train_duration [s], num_bunches_in_train [int], and bunch_charge [C]
    # Stored like this confusing mess since average_current_train an train_duration are the most important parameters

    def _ensureFloat(self,r, ensurePos=False) -> float:
        "Little helper function, allowing None or float, autoconverting int to float. If ensurePos is True, then only allow r>0.0."
        if r == None:
            return r
        if type(r) == int:
            #quietly convert int to float
            r = float(r)
        if type(r) != float:
            raise TypeError("must be float, int, or None")
        if ensurePos and r < 0.0:
            raise ValueError("must be >= 0.0")
        return r

    @property
    def bunch_charge(self) -> float:
        """The total charge of one bunch [C]"""
        if self._bunch_charge == None:
            raise abel.RFAcceleratorInitializationException("bunch_charge not yet initialized")
        return self._bunch_charge
    @bunch_charge.setter
    def bunch_charge(self, bunch_charge : float):
        self._bunch_charge = self._ensureFloat(bunch_charge)

    @property
    def bunch_separation(self) -> float:
        "The time [s] between each bunch"
        if self._bunch_separation == None:
            raise abel.RFAcceleratorInitializationException("bunch_separation not yet initialized")
        return self._bunch_separation
    @bunch_separation.setter
    def bunch_separation(self, bunch_separation : float):
        self._bunch_separation = self._ensureFloat(bunch_separation,True)
    
    @property
    def num_bunches_in_train(self) -> int:
        """
        The number of bunches in the train.
        When 1, train_duration is 0.0 and average_current_train is None / undefined.
        """
        if self._num_bunches_in_train == None:
            raise abel.RFAcceleratorInitializationException("num_bunches_in_train not yet initialized")
        return self._num_bunches_in_train
    @num_bunches_in_train.setter
    def num_bunches_in_train(self, num_bunches_in_train : int):
        if num_bunches_in_train == None:
            self._num_bunches_in_train = None
            return
        if type(num_bunches_in_train) != int:
            raise TypeError("num_bunches_in_train must be int or None")
        if num_bunches_in_train <= 0:
            raise ValueError("num_bunches_in_train must be > 0")
        self._num_bunches_in_train = num_bunches_in_train

    @property
    def bunch_frequency(self) -> float:
        if self.num_bunches_in_train == 1:
            raise ValueError("Bunch frequency undefined when num_bunches_in_train == 1")
            return None
        return 1.0/self.bunch_separation
    @bunch_frequency.setter
    def bunch_frequency(self, bunch_frequency):
        raise NotImplementedError("Cannot directly set bunch_frequency")
    
    @property
    def train_duration(self) -> float:
        """
        The train duration [s] = beam pulse length [s] = length of RF pulse flat top length [s] for the RF structures.
        0.0 for single-bunch trains.
        Normally populated from Beam in track() but can be set directly for calculator use.
        """
        if self.num_bunches_in_train == 1:
            return 0.0 
        return self.bunch_separation*(self.num_bunches_in_train-1)
    @train_duration.setter
    def train_duration(self, train_duration : float):
        raise NotImplementedError("Cannot directly set train_duration")
    
    @property
    def average_current_train(self) -> float:
        """
        The beam current of the linac [A].
        """
        return self.bunch_charge*self.bunch_frequency
    @average_current_train.setter
    def average_current_train(self, average_current_train : float) -> None:
        raise NotImplementedError("Cannot directly set average_current_train")

    @property
    def beam_charge(self) -> float:
        """
        The total charge of the beam [C]
        """
        if self._bunch_charge != None and self._num_bunches_in_train != None:
            return self._bunch_charge*self._num_bunches_in_train
        raise abel.RFAcceleratorInitializationException("Beam charge not initialized")
    @beam_charge.setter
    def beam_charge(self,beam_charge : float):
        raise NotImplementedError("Cannot set beam_charge directly")

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