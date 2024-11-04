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

    default_num_rf_cells = 20 # [m]
    default_fill_factor = 0.71
    default_rf_frequency = 2e9 # [Hz]
    
    default_operating_temperature = 300
    
    @abstractmethod
    def __init__(self, length=None, nom_energy_gain=None, num_rf_cells=default_num_rf_cells, fill_factor=default_fill_factor, rf_frequency=default_rf_frequency, operating_temperature=default_operating_temperature):
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
            The length of the RF structure [m].
            Must be specified. Might be calculated in subclass from e.g. num RF cells.

        num_structures : int
            The number of RF structures in the RFaccelerator

        nom_energy_gain : float
            The total accelerating voltage (e.g. the energy gain) of the RF structures [V].
            Either this or gradient must be specified.
        """

        #Force them all to None/uninitialized
        super().__init__(num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None)

        #Initialize variables through property setters
        #
        #Internally, we use self._length, self._num_structures, and self._structure_length to keep track of the geometry;
        # and self.nom_energy_gain to keep track of RF voltage.
        # The properties nom_accel_gradient, fill_factor, voltage_structure, and gradient_structure are calculated from these
        #If we try to set all the parameter directly without a well-defined set of variables,
        # we easily end up with inconsistencies and infinite recursions, or very complex logic.
        # Here we just set these "basic" parameters, use the property setters/getters to get to the other ones.

        self.length = length
        self.nom_energy_gain = nom_energy_gain
        self.num_rf_cells = num_rf_cells
        self.fill_factor = fill_factor
        self.rf_frequency = rf_frequency
        self.operating_temperature = operating_temperature

        # default settings
        self.num_structures_per_klystron = 1.0
        self.efficiency_wallplug_to_rf   = 0.623 # from CLIC CDR 2012

        #Initialize the average_current_train and train_duration to None,
        # and get it from Beam once in track().
        # Can also be manually specified.
        # For consistency, it follows how Beam does it even if average current + pulse length
        # is more fundamental for CLICopti modelling.
        self.bunch_charge           = None # [C]

        self.name = 'RF accelerator'

    def __str__(self):
        "Print info"
        s = f"{self.name}: Length={self.length:.1f}[m], L_struct={self.structure_length*1e3:.0f}[mm], N={self.num_structures}, fill={self.fill_factor*100:.3f}[%], " + \
            f"Egain={self.nom_energy_gain/1e9:.3f} [GV], gradient_structure={self.gradient_structure/1e6:.1f}[MV/m], rf_frequency={self.rf_frequency/1e9}[GHz], " + \
            f"bunch_charge={self._bunch_charge}[C], bunch_separation={self._bunch_separation}[s], num_bunches_in_train={self._num_bunches_in_train}"
        return s

    #-----------------------------------------#
    # Properties related to voltage and power #
    #=========================================#

    @property
    def nom_energy_gain(self) -> float:
        """The total nominal accelerating voltage [eV] of the RFAccelerator"""
        #if self._nom_energy_gain is None:
        #    raise RFAcceleratorInitializationException("nom_energy_gain not yet initialized")
        return self._nom_energy_gain
    @nom_energy_gain.setter
    def nom_energy_gain(self,nom_energy_gain : float):
        self._nom_energy_gain = nom_energy_gain
        if hasattr(self, '_nom_accel_gradient'):
            self.length = self.nom_energy_gain/self._nom_accel_gradient
            del self._nom_accel_gradient
    def get_nom_energy_gain(self):
        return self.nom_energy_gain

    @property
    def nom_accel_gradient(self) -> float:
        """
        The nominal accelerating gradient of the whole linac [V/m], ignoring that some of it is "empty" space due to fill_factor.
        On setting, it changes the nom_energy_gain
        Note: the gradient of the structures, e.g. the 100e6 V/m quoted for CLIC structures, are given by gradient_structure.
        """
        if hasattr(self, '_nom_accel_gradient'):
            if self.nom_energy_gain is not None:
                self.length = self.nom_energy_gain/self._nom_accel_gradient
                del self._nom_accel_gradient
            else:
                return self._nom_accel_gradient
        return self.nom_energy_gain/self.length
    @nom_accel_gradient.setter
    def nom_accel_gradient(self, nom_accel_gradient : float):
        if self.nom_energy_gain is not None:
            self.length = self.nom_energy_gain/nom_accel_gradient
        elif self.length is not None:
            self.nom_energy_gain = nom_accel_gradient*self.length
        else:
            self._nom_accel_gradient = nom_accel_gradient

    @property
    def voltage_structure(self) -> float:
        """
        The accelerating voltage of a single RF structure [V].
        On setting, it changes the nom_energy_gain.
        """
        if self.nom_energy_gain is not None and self.num_structures is not None:
            return self.nom_energy_gain / self.num_structures
        else:
            return None
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


    #---------------------------------------#
    # Properties related to train structure #
    #=======================================#

    @property
    def bunch_charge(self) -> float:
        """The total charge of one bunch [C]"""
        if self._bunch_charge == None:
            raise RFAcceleratorInitializationException("bunch_charge not yet initialized")
        return self._bunch_charge
    @bunch_charge.setter
    def bunch_charge(self, bunch_charge : float):
        self._bunch_charge = self._ensureFloat(bunch_charge)

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
        raise RFAcceleratorInitializationException("Beam charge not initialized")
    @beam_charge.setter
    def beam_charge(self,beam_charge : float):
        raise NotImplementedError("Cannot set beam_charge directly")

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
        #if self._length is None:
        #    raise RFAcceleratorInitializationException("length not yet initialized")
        return self._length
    @length.setter
    def length(self,length : float):
        self._length = length

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
        return self._fill_factor
    @fill_factor.setter
    def fill_factor(self, fill_factor : float):
        self._fill_factor = fill_factor

    @property
    def structure_length(self) -> float:
        "The length of each individual RF structure [m]"
        #Note: structure_length overridden by RFAccelerator_TW,
        #      which calculates it from num_rf_cells
        #if self._structure_length is None:
        #    raise RFAcceleratorInitializationException("structure_length not yet initialized")
        return self._structure_length
    @structure_length.setter
    def structure_length(self, structure_length : float):
        self._structure_length = structure_length

    @property
    def num_structures(self) -> int:
        "The number of individual RF structures. Must be >= 1"
        if self.length is not None and self.fill_factor is not None and self.structure_length is not None:
            return self.length*self.fill_factor/self.structure_length
        else:
            return None
    
    @num_structures.setter
    def num_structures(self,num_structures : int):
        self.length = num_structures*self.structure_length/self.fill_factor
    def get_num_structures(self):
        return self.num_structures

    #rf_frequency is overridden by RFAccelerator_TW, which manages it through the CLICopti object
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

    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
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
        "Calculate the wall-plug power total [W]"
        return self.wallplug_power_rf() + self.wallplug_power_cooling()

    def wallplug_power_cooling(self) -> float:
        "Calculate the wall-plug power for cooling [W]"
        return self.energy_usage_cooling() * self.get_rep_rate_average()

    def wallplug_power_rf(self) -> float:
        "Calculate the wall-plug power for klystrons [W]"
        return self.energy_usage_rf() * self.get_rep_rate_average()

    def heat_power_at_cryo_temperature(self) -> float:
        return self.heat_energy_at_cryo_temperature() * self.get_rep_rate_average()
    
    def get_klystron_average_power(self) -> float:
        "Calculate the average power per klystron [W]"
        return self.wallplug_power_rf() / self.get_num_klystrons()

    def get_klystron_peak_power(self):
        return self.get_structure_power() * self.num_structures_per_klystron
        
    def get_num_klystrons(self) -> int:
        "Get number of klystrons"
        return int(np.ceil(self.num_structures/self.num_structures_per_klystron))

    #---------------------------------------------#
    # Methods for cost modelling                  #
    #=============================================#
    
    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        return self.num_structures * self.structure_length * CostModeled.cost_per_length_rf_structure_normalconducting

    def get_cost_remaining_beamline(self):
        "Cost of the beamline between structures [ILC units]"
        return self.length * (1-self.fill_factor) * CostModeled.cost_per_length_instrumented_beamline

    def get_cost_cryo_infrastructure(self):
        "Cost of the cryo infrastructure [ILC units]"
        if self.operating_temperature <= 77 and self.operating_temperature > 40:
            return self.heat_power_at_cryo_temperature() * CostModeled.cost_per_power_reliquification_plant_nitrogen
        elif self.operating_temperature >= 2 and self.operating_temperature <= 4:
            return self.heat_power_at_cryo_temperature() * CostModeled.cost_per_power_reliquification_plant_helium
        else:
            return None
        
    def get_cost_klystrons(self):
        "Cost of the klystrons and modulators [ILC units]"
        return self.get_num_klystrons() * CostModeled.cost_per_klystron(self.get_num_klystrons(), self.rf_frequency, self.get_klystron_average_power(), self.get_klystron_peak_power())
        
    def get_cost_breakdown(self):
        "Breakdown of costs"
        breakdown = []
        breakdown.append((f"Instrumented beamline ({(1-self.fill_factor)*100:.0f}%)", self.get_cost_remaining_beamline()))
        breakdown.append((f"RF structures ({self.get_num_structures():.0f}x)", self.get_cost_structures()))
        breakdown.append((f"Klystrons ({self.get_num_klystrons():.0f}x, {self.get_klystron_peak_power()/1e6:.0f} MW peak, {self.get_klystron_average_power()/1e3:.0f} kW avg)", self.get_cost_klystrons()))
        cooling_cost = self.get_cost_cryo_infrastructure()
        if cooling_cost is not None:
            breakdown.append((f"Cryo plants ({self.heat_power_at_cryo_temperature()/1e6:.1f} MW at {self.operating_temperature:.0f} K)", cooling_cost))
        return (self.name, breakdown)

class RFAcceleratorInitializationException(Exception):
    "An Exception class that is raised when trying to access a uninitialized field"
    pass

