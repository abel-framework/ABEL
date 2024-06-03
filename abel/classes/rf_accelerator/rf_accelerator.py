#!/usr/bin/env python3

from abc import abstractmethod

import abel

import matplotlib
import matplotlib.pyplot as plt

class RFAccelerator(abel.Trackable):
    """
    Class modelling a RF linac, typically used for drivers or injectors.
    It's an abstract class, meant to be superseeded by a specific RFlinac implementation, relating to a structure type and how it is modelled.
    """

    @abstractmethod
    def __init__(self, length=None, num_structures=None, gradient=None, voltage_total=None,  bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
        """
        Initialize the rf_accelerator base class.
        This can interface an underlying RF structure model including power requirements, and costs.
        It provides a trackable object.

        Note: This constructor should be called AFTER the implementing class is done.

        Parameters
        ----------

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
            Either this or gradient must be specified.

        bunch_separation : float
            The separation of bunches during the beam pulse [s]

        num_bunches_in_train : int
            The number of bunches in the beam pulse

        rep_rate_trains : float
            The repetition rate of the beam & RF pulses

        """

        #We assume that the implementer constructor has ran, so that the RFstructure is ready
        # and can be used to define the linac

        #Setting up RFlinac geometry
        if num_structures == 1 and length == None:
            #Use a single RF structure, RF linac length = structure length
            self.length = self.get_RF_structure_length()
            self.num_structures = 1
        elif length==None or num_structures==None:
            raise ValueError("Must specify both linac total length `length` and number of RF structures `num_structures` so that fill factor can be calculated")
        else:
            self.length = length
            self.num_structures = int(num_structures)

        self.fill_factor = self.num_structures * self.get_RF_structure_length() / self.length
        if self.fill_factor > 1:
            raise ValueError(f"Fill factor = {self.fill_factor} > 1, this is not physically possible")

        #Setting RF pulse parameters
        if gradient == None and voltage_total == None:
            raise ValueError("Must specify gradient or voltage")
        elif voltage_total == None:
            self.set_gradient(gradient)
        elif gradient == None:
            self.set_voltage_total(voltage_total)

        #if self.beam_pulse_length == None:
        #    raise ValueError("Must set beam pulse length")

        if bunch_separation==None:
            #TODO: Check that this correspods to an integer number of 1/frequency
            raise ValueError("Must set bunch separation")

        # bunch train pattern
        self.bunch_separation = bunch_separation # [s]
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains # [Hz]

        #TODO - Legacy, need to merge
        #self.beam_pulse_length = self.set_beam_pulse_length(self.train_duration())
        #self.beam_current      = self.set_beam_current(beam_current)

        #TODO
        self.cost_per_length = 0.22e6 # [ILCU/m] not including klystrons (ILC is 0.24e6 with power)

    #Implement stuff from Trackable

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        #TODO: Check current etc.
        beam.set_Es(beam.Es()+self.voltage_total)
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        return self.length

    def survey_object(self):
        #return super().survey_object()
        #TODO: This was copied from source; maybe it would be a decent default implementation too?
        rect = matplotlib.patches.Rectangle((0, -1), self.get_length(), 2)
        return rect

    #Define and implement RFaccelerator specifics

    # RFlinac geometry
    @abstractmethod
    def get_RF_structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return np.nan

    def get_num_structures(self) -> int:
        "Gets the number of individual RF structures [m]"
        return self.num_structures

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
        "Gets the total accelerating voltage of the whole RFaccelerator object [V]"
        return self.voltage_total
    def get_nom_energy_gain(self) -> float:
        "Alias of get_voltage_total"
        return self.get_voltage_total()

    def get_voltage_structure(self) -> float:
        "Gets the accelerating voltage of a single RF structure [V]"
        return self.voltage_structure

    # Time structure of pulse and beam

    @abstractmethod
    def get_RF_frequency(self):
        "Get the RF frequency of the RF structures [1/s]"
        return np.nan

    def rep_rate_average(self):
        "Average repetition rate of bunches [1/s]"
        return self.num_bunches_in_train * self.rep_rate_trains

    def rep_rate_intratrain(self):
        "Bunch rate [1/s]"
        if self.bunch_separation is not None:
            return 1/self.bunch_separation
        else:
            return None

    def train_duration(self):
        #TODO: Consolidate with beam_pulse_length and fact that bunch_separation is now mandatory
        if self.bunch_separation is not None:
            return self.bunch_separation * (self.num_bunches_in_train-1)
        else:
            return None

    # Energy use and costing

    @abstractmethod
    def energy_usage(self):
        "Calculate the energy usage (per pulse or per bunch?) [unit]"
        pass

    def wallplug_power(self):
        "Calculate the "
        return self.energy_usage() * self.rep_rate_average()

    # Cost model

    def get_cost(self):
        "Cost of the rf_accelerator [ILC units]"
        return self.get_length() * self.cost_per_length