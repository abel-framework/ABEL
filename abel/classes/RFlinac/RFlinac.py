#!/usr/bin/env python3

#from abel import Trackable
import abel

import CLICopti

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from abc import abstractmethod

class RFlinac(abel.Trackable):
    """
    Class modelling a conventional warm(ish) travelling wave linac, typically used for drivers or injectors.
    It's an abstract class, meant to be superseeded by a specific RFlinac implementation, relating to a structure type etc.
    """

    def __init__(self, RF_structure, length=None, num_structures=None, gradient=None, voltage_total=None, beam_pulse_length=None,beam_current=0.0):
        """
        Initialize the RFlinac base class.
        This interfaces the underlying RF_structure, as well as (TODO:) the klystron model, providing a trackable object.

        Parameters
        ----------
        RF_structure : CLICopti.RFstructure.Accelstructure object
            The underlying AccelStructure object, typically provided by the implementing subclass.
            Must be specified.

        length : float
            The length of the entire linac [m], including drift space etc. between structures
            Must be specified

        num_structures : int
            The number of RF structures in the linac.
            Must be specified.

        gradient : float
            The accelerating gradient in the linac RF structures [V/m].
            Either this or voltage_total must be specified.

        voltage_total : float
            The total accelerating voltage (e.g. the energy gain) of the RF structures [V].
            Either this or voltage_total must be specified.

        beam_pulse_length : float
            The length of the beam pulse [s], which is equal to the RF pulse flat top.

        beam_current : float
            The average current during the beam pulse [A].

        """

        #Finalizing the RFstructure initialization
        if RF_structure == None:
            raise ValueError("Must set RF structure; this is typically done from an implementing daugther class")
        self._RF_structure = RF_structure
        self._RF_structure.calc_g_integrals(1000)

        #Setting up RFlinac geometry
        if num_structures == 1 and length == None:
            #Use a single RF structure, RF linac length = structure length
            self.length = self._RF_structure.getL()
            self.num_structures = 1
        elif length==None or num_structures==None:
            raise ValueError("Must specify both linac total length `length` and number of RF structures `num_structures` so that fill factor can be calculated")
        else:
            self.length = length
            self.num_structures = int(num_structures)

        self.fill_factor = self.num_structures * self._RF_structure.getL() / self.length
        if self.fill_factor > 1:
            raise ValueError(f"Fill factor = {self.fill_factor} > 1, this is not physically possible")

        #Setting RF pulse parameters
        if gradient == None and voltage_total == None:
            raise ValueError("Must specify gradient or voltage")
        elif voltage_total == None:
            self.set_gradient(gradient)
        elif gradient == None:
            self.set_voltage_total(voltage_total)

        if self.beam_pulse_length == None:
            raise ValueError("Must set beam pulse length")

        self.beam_pulse_length = self.set_beam_pulse_length(beam_pulse_length)
        self.beam_current      = self.set_beam_current(beam_current)

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        #TODO: Check current etc.
        beam.set_Es(beam.Es()+self.voltage_total)
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        #return super().get_length()
        return self.length

    def survey_object(self):
        #return super().survey_object()
        #TODO: This was copied from source; maybe it would be a decent default implementation too?
        rect = matplotlib.patches.Rectangle((0, -0.5), self.get_length(), 1)
        return rect

    ## RF structure model ##

    # RFlinac geometry
    def get_structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self._RF_structure.getL()

    def get_num_structures(self) -> int:
        "Gets the number of individual RF structures [m]"
        return self.num_structures

    def get_structure_pulse_energy(self) -> float:
        "Get the energy requirements for a single pulse for one structure"
        #Note: Model is assuming a pulse like from plotPowerProfile(),
        #      which assumes a drive beam driven pulse in that pulse end is a mirror of the start,
        #      and that the pusle shape can be easily "reorganized" to a square pulse.
        #      As long as Tfill is short, this doesn't matter.
        return (self._RF_structure.getTrise() + self.get_t_fill() + self.beam_pulse_length) * \
            self.get_structure_power()

    # RF pulse parameters

    def set_gradient(self,gradient : float) -> None:
        "Set the accelerating gradient of the structures [V/m]"
        self.gradient = gradient
        self.voltage_structure = gradient*self._RF_structure.getL()
        self.voltage_total     = self.voltage_structure * self.num_structures

    def get_gradient(self) -> float:
        return self.gradient

    def set_voltage_total(self,voltage_total : float) -> None:
        "Set the total accelerating voltage [V] of the RFlinac"
        self.voltage_total = voltage_total
        self.voltage_structure = voltage_total / self.num_structures
        self.gradient = self.voltage_structure / self.get_structure_length()

    def get_voltage_total(self) -> float:
        "Gets the total accelerating voltage of the RFlinac object [V]"
        return self.voltage_total

    def get_voltage_structure(self) -> float:
        "Gets the accelerating voltage of a single RF structure [V]"
        return self.voltage_structure

    def set_beam_current(self, beam_current : float) -> None:
        "Set the beam current of the linac [A]"
        self.beam_current = beam_current

    def get_beam_current(self) -> float:
        "Sets the beam current of the linac [A]"
        return self.beam_current

    def get_structure_power(self) -> float:
        "Get the peak power [W] required for a single RF structure in the given configuration."
        if self.beam_current == 0.0:
            P = self._RF_structure.getPowerUnloaded(self.voltage_structure)
        else:
            P = self._RF_structure.getPowerLoaded(self.voltage_structure, self.beam_current)
        return P

    def get_RF_efficiency(self) -> float:
        "Get the RF->beam efficiency as a number between 0 and 1, including the effect due to pulse shape"
        return self._RF_structure.getTotalEfficiency(self.get_structure_power(), self.beam_current, self.beam_pulse_length)

    def get_RF_efficiency_flattop(self) -> float:
        "Get the RF->beam efficiency as a number between 0 and 1, ignoring the fill time etc."
        return self._RF_structure.getFlattopEfficiency(self.get_structure_power(), self.beam_current)

    def set_beam_pulse_length(self, beam_pulse_length : float) -> None:
        "Set the beam pulse length (=length of RF pulse flat top) [s] for the RF structures"
        self.beam_pulse_length = beam_pulse_length

    def get_beam_pulse_length(self) -> float:
        "Gets the beam pulse length (=length of RF pulse flat top) [s] for the RF structues"
        return self.beam_pulse_length

    def get_t_fill(self) -> float:
        "Get the filling time of the structure, i.e. the time for a signal to propagate through the whole structure."
        return self._RF_structure.getTfill()

    def get_pulse_length_total(self) -> float:
        return 2 * (self._RF_structure.getTrise() + self._RF_structure.getTfill()) \
                 + self.beam_pulse_length

    def get_max_pulse_length(self) -> float:
        "Calculates the max beam_pulse_length before exceeding gradient limits, given power and beam_current"
        return self._RF_structure.getMaxAllowableBeamTime(self.get_structure_power(), self.get_beam_current())

    def set_pulse_length_max(self) -> float:
        "Sets the pulse length to the maximally achievable given the currently selected gradient and beam current"
        self.set_beam_pulse_length( self.get_max_pulse_length() )
        return self.get_beam_pulse_length()

    def get_gradient_max(self) -> float:
        "Calculates the max gradient (and thus voltage) before exceeding breakdown limts, given the currently selected pulse length and beam current"
        return self._RF_structure.getMaxAllowablePower_beamTimeFixed(self.get_beam_current(), self.get_beam_pulse_length())

    def set_gradient_max(self) -> float:
        "Sets the gradient to the maximally achievable given the currently selected pulse length and beam current"
        self.set_gradient( self.get_gradient_max() )
        return self.get_gradient()

    ## PLOTS ##

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
                                         self.beam_pulse_length,\
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
        tit += f", <G>={self.gradient/1e6:.1f} [MV/m]"
        return tit

    @abstractmethod
    def make_structure_title(self) -> str:
        "Make a 'personalized' title for the implementing RF structure"
        return f"{type(self._RF_structure).__name__}, V={self.voltage_structure/1e6:.1f} [MV]"

    ##Klystron model
    #TODO - simple number for efficiency

    #COST MODEL - learning curve already in simple model

    #Todo: not CamelCase but_it_should_be_underscored
