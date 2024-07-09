#!/usr/bin/env python3

from abc import abstractmethod
from matplotlib import patches
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled

import numpy as np
from matplotlib import pyplot as plt

class RFAccelerator(Trackable, CostModeled):
    """
    Class modelling a RF linac, typically used for drivers or injectors.
    It's an abstract class, meant to be superseeded by a specific RFlinac implementation, relating to a structure type and how it is modelled.
    """

    @abstractmethod
    def __init__(self, length=None, structure_length=None, num_structures=None, nom_energy_gain=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
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

        structure_length : float
            The length of the RF structure.
            Must be specified.

        num_structures : int
            The number of RF structures in the RFaccelerator

        nom_energy_gain : float
            The total accelerating voltage (e.g. the energy gain) of the RF structures [V].
            Either this or gradient must be specified.

        bunch_separation : float
            The separation of bunches during the beam pulse [s]

        num_bunches_in_train : int
            The number of bunches in the beam pulse

        rep_rate_trains : float
            The repetition rate of the beam & RF pulses

        """
        
        # set bunch pattern
        super().__init__(num_bunches_in_train=num_bunches_in_train, bunch_separation=bunch_separation, rep_rate_trains=rep_rate_trains)
        
        #Initialize variables through property setters
        #
        #Internally, we use self._length, self._num_structures, and self._structure_length to keep track of the geometry;
        # and self.nom_energy_gain to keep track of RF voltage.
        # The properties nom_accel_gradient, fill_factor, voltage_structure, and gradient_structure are calculated from these
        #If we try to set all the parameter directly without a well-defined set of variables,
        # we easily end up with inconsistencies and infinite recursions, or very complex logic.
        # Here we just set these "basic" parameters, use the property setters/getters to get to the other ones.

        self.length             = length

        if not hasattr(self, "_structure_length"):
            #If already handled by daughter class, don't try to set again
            # E.g in rf_accelerator_TW, the structure_length is a computed parameter
            # and cannot be set directly.
            self.structure_length   = structure_length
        self.num_structures     = num_structures

        self.nom_energy_gain    = nom_energy_gain

        # default settings
        self.num_structures_per_klystron = 1.0
        self.efficiency_wallplug_to_rf   = 0.55

    #-----------------------------------------#
    # Properties related to voltage and power #
    #=========================================#

    @property
    def nom_energy_gain(self) -> float:
        """
        The total nominal accelerating voltage [eV] of the RFAccelerator;
        """
        if self._nom_energy_gain is None:
            raise RFAcceleratorInitializationException("nom_energy_gain not yet initialized")
        return self._nom_energy_gain
    @nom_energy_gain.setter
    def nom_energy_gain(self,nom_energy_gain : float):
        self._nom_energy_gain = nom_energy_gain

    @property
    def voltage_total(self) -> float:
        "Alias of RFAccelerator.nom_energy_gain"
        return self.nom_energy_gain
    @voltage_total.setter
    def voltage_total(self,voltage_total : float):
        self.nom_energy_gain = voltage_total

    @property
    def nom_accel_gradient(self) -> float:
        """
        The nominal accelerating gradient of the whole linac [V/m], ignoring that some of it is "empty" space due to fill_factor.
        On setting, it changes the nom_energy_gain
        Note: the gradient of the structures, e.g. the 100e6 V/m quoted for CLIC structures, are given by gradient_structure.
        """
        return self.nom_energy_gain/self.length
    @nom_accel_gradient.setter
    def nom_accel_gradient(self,nom_accel_gradient : float):
        self.nom_energy_gain = nom_accel_gradient*self.length

    @property
    def voltage_structure(self) -> float:
        """
        The accelerating voltage of a single RF structure [V].
        On setting, it changes the nom_energy_gain.
        """
        return self.nom_energy_gain / self.num_structures
    @voltage_structure.setter
    def voltage_structure(self,voltage_structure : float):
        self.nom_energy_gain = voltage_structure*self.num_structures

    @property
    def gradient_structure(self) -> float:
        """
        The accelerating gradient in the structures [V/m], excluding the "empty" space due to fill_factor.
        On setting, it changes the structure voltage which changes the nom_energy_gain
        """
        return self.voltage_structure / self.structure_length
    @gradient_structure.setter
    def gradient_structure(self,gradient_structure : float):
        self.voltage_structure = gradient_structure*self.structure_length


    #--------------------------------#
    # Properties related to geometry #
    #================================#

    @property
    def length(self) -> float:
        """
        The total length of the RFlinac [m], including the empty space between structures.
        Note: Changing the length does not modify the nom_energy_gain or num_structures, but the gradient and fill_factor is implicitly changed.
        Set length_constgradfill instead for changing the length of the linac so that the gradient is constant.
        """
        if self._length is None:
            raise RFAcceleratorInitializationException("length not yet initialized")
        return self._length
    @length.setter
    def length(self,length : float):
        self._length = length

    @property
    def length_constgradfill(self) -> float:
        """
        Alternative name for length.
        Setting this keeps the nom_accel_gradient and fill_factor constant, modifying the total voltage and fill factor.
        """
        return self._length
    @length_constgradfill.setter
    def length_constgradfill(self,length : float):
        g0 = self.nom_accel_gradient
        f0 = self.fill_factor
        self._length = length
        self.nom_accel_gradient = g0
        self.fill_factor = f0

    @property
    def fill_factor(self) -> float:
        """
        The fill factor of the RFAccelerator, a number > 0 and <= 1, defined as
        the fraction of the linac occupied by active RF accelerating structures.
        On setting, it changes the num_structures, rounding down but not below 1,
        which means that setting and reading back the fill_factor might not give exactly the same answer.
        """
        #TODO: Can get this over 1 by setting num_structures, length, and structure length badly.
        # Should probably send a warning if this is the case.
        return self.num_structures * self.structure_length / self.length
    @fill_factor.setter
    def fill_factor(self,fill_factor : float):
        if fill_factor > 1.0 or fill_factor <= 0.0:
            raise ValueError("Invalid fill_factor outside the half-open interval (0.0,1.0].")
        ns = int(np.floor(fill_factor * self.length / self.structure_length))
        if ns == 0:
            ns = 1
        self.num_structures = ns

    #structure_length overridden by RFAccelerator_TW
    @property
    def structure_length(self) -> float:
        "The length of each individual RF structure [m]"
        if self._structure_length is None:
            raise RFAcceleratorInitializationException("structure_length not yet initialized")
        return self._structure_length
    @structure_length.setter
    def structure_length(self, structure_length : float):
        self._structure_length = structure_length

    @property
    def num_structures(self) -> int:
        "The number of individual RF structures. Must be >= 1"
        if self._num_structures is None:
            raise RFAcceleratorInitializationException("num_structures not yet initialized")
        return self._num_structures
    @num_structures.setter
    def num_structures(self,num_structures : int):
        if num_structures is None:
            self._num_structures = num_structures
            return
        num_structures = int(num_structures)
        if num_structures < 1:
            raise ValueError("num_structures must be >=1")
        self._num_structures = num_structures

    #rf_frequency overridden by RFAccelerator_TW
    @property
    def rf_frequency(self) -> float:
        "The RF frequency of the RF structures [1/s]"
        return self._rf_frequency
    @rf_frequency.setter
    def rf_frequency(self,rf_frequency : float):
        self._rf_frequency = rf_frequency

    #-------------------------------------------#
    # Implement abstract methods from Trackable #
    #===========================================#

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        #TODO: Check current etc.
        beam.set_Es(beam.Es() + self.nom_energy_gain)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        "Returns the linac physical length [m], using the RFaccelerator length property"
        return self.length

    def survey_object(self):
        #return patches.Rectangle((0, -1), self.get_length(), 2)

        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'RF accelerator'
        color = 'blue'
        return x_points, y_points, final_angle, label, color

    #---------------------------------------------#
    # Methods for energy use modelling            #
    #=============================================#

    @abstractmethod
    def energy_usage(self) -> float:
        "Calculate the energy usage per bunch [J]"
        pass

    def wallplug_power(self) -> float:
        "Calculate the wall-plug power [W]"
        return self.energy_usage() * self.get_rep_rate_average()

    def get_klystron_average_power(self):
        "Calculate the average power per klystron [W]"
        return self.wallplug_power() / self.get_num_klystrons()
        
    def get_num_klystrons(self) -> int:
        "Get number of klystrons"
        return int(np.ceil(self.num_structures/self.num_structures_per_klystron))

    #---------------------------------------------#
    # Methods for cost modelling                  #
    #=============================================#

    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        return self.get_length() * CostModeled.cost_per_length_rf_structure

    def get_cost_klystrons(self):
        "Cost of the klystrons and modulators [ILC units]"
        return self.get_num_klystrons() * CostModeled.cost_per_klystron(self.get_num_klystrons(), self.rf_frequency, self.get_klystron_average_power())
        
    def get_cost_breakdown(self):
        "Breakdown of costs"
        breakdown = []
        breakdown.append((f"RF structures ({self.get_num_structures()}x)", self.get_cost_structures()))
        breakdown.append((f"Klystrons and modulators ({self.get_num_klystrons()}x)", self.get_cost_klystrons()))
        return ('RF accelerator', breakdown)

class RFAcceleratorInitializationException(Exception):
    "An Exception class that is raised when trying to access a uninitialized field"
    pass
