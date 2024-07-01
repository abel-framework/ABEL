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
    def __init__(self, length=None, structure_length=None, nom_accel_gradient=None, nom_energy_gain=None, fill_factor=None, rf_frequency=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
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

        nom_accel_gradient : float
            The accelerating gradient in the linac RF structures [V/m].
            Either this or voltage_total must be specified.

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
        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)
        
        self.nom_energy_gain = nom_energy_gain
        self.nom_accel_gradient = nom_accel_gradient
        self.length = length
        self.fill_factor = fill_factor
        self.rf_frequency = rf_frequency
        self.structure_length = structure_length
        
        # default settings
        self.num_structures_per_klystron = 1.0
        self.efficiency_wallplug_to_rf = 0.55

        self.name = 'RF accelerator'

    
    # implement stuff from Trackable

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        #TODO: Check current etc.
        beam.set_Es(beam.Es() + self.get_nom_energy_gain())
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        if self.length is None:
            return self.get_nom_energy_gain()/self.get_nom_accel_gradient()
        else:
            return self.length

    def get_fill_factor(self):
        if self.fill_factor is not None:
            return self.fill_factor
        else:
            return self.get_num_structures() * self.get_structure_length() / self.get_length()
    
        
    def survey_object(self):
        #return patches.Rectangle((0, -1), self.get_length(), 2)
        
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'RF accelerator'
        color = 'blue'
        return x_points, y_points, final_angle, label, color

    
    # define and implement RFaccelerator specifics
    
    def get_structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self.structure_length

    def get_num_structures(self) -> int:
        "Gets the number of individual RF structures [m]"
        return int(round(self.get_fill_factor()*self.get_length()/self.get_structure_length()))

    
    # RF pulse parameters

    def set_gradient(self, nom_accel_gradient : float) -> None:
        "Set the accelerating gradient of the structures [V/m]"
        self.nom_accel_gradient = nom_accel_gradient
        self.voltage_structure = gradient*self.get_structure_length()
        self.nom_energy_gain     = self.voltage_structure * self.get_num_structures()

    def get_nom_accel_gradient(self) -> float:
        if self.nom_accel_gradient is not None:
            return self.nom_accel_gradient
        else:
            return self.nom_energy_gain/self.length

    def set_nom_energy_gain(self,nom_energy_gain : float) -> None:
        "Set the total accelerating voltage [V] of the RFlinac"
        self.nom_energy_gain = nom_energy_gain
        self.voltage_structure = nom_energy_gain / self.get_num_structures()
        self.nom_accel_gradient = self.voltage_structure / self.get_structure_length()

    def get_nom_energy_gain(self) -> float:
        "Gets the total accelerating voltage of the whole RFaccelerator object [eV]"
        return self.nom_energy_gain
        
    def get_voltage_total(self) -> float:
        "Alias of get_nom_energy_gain [V]"
        return self.get_nom_energy_gain()

    def get_voltage_structure(self) -> float:
        "Gets the accelerating voltage of a single RF structure [V]"
        return self.get_nom_energy_gain() / self.get_num_structures()

    def get_gradient_structure(self) -> float:
        "Gets the accelerating gradient in the structures [V/m]"
        return self.get_voltage_structure() / self.get_structure_length()

    # time structure of pulse and beam
    
    def get_rf_frequency(self):
        "Get the RF frequency of the RF structures [1/s]"
        return self.rf_frequency
    

    # Energy use and costing

    @abstractmethod
    def energy_usage(self) -> float:
        "Calculate the energy usage per bunch [J]"
        pass

    def wallplug_power(self) -> float:
        "Calculate the wall-plug power total [W]"
        return self.wallplug_power_cooling() + self.wallplug_power_klystrons()

    def wallplug_power_cooling(self) -> float:
        "Calculate the wall-plug power for cooling [W]"
        return self.energy_usage_cooling() * self.get_rep_rate_average()

    def wallplug_power_klystrons(self) -> float:
        "Calculate the wall-plug power for klystrons [W]"
        return self.energy_usage_klystrons() * self.get_rep_rate_average()

    def get_klystron_average_power(self) -> float:
        "Calculate the average power per klystron [W]"
        return self.wallplug_power_klystrons() / self.get_num_klystrons()
        
    def get_num_klystrons(self) -> int:
        "Get number of klystrons"
        return int(np.ceil(self.get_num_structures()/self.num_structures_per_klystron))

    # costs

    @abstractmethod
    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        pass

    def get_cost_klystrons(self):
        "Cost of the klystrons and modulators [ILC units]"
        return self.get_num_klystrons() * CostModeled.cost_per_klystron(self.get_num_klystrons(), self.get_rf_frequency(), self.get_klystron_average_power())
        
    def get_cost_breakdown(self):
        "Breakdown of costs"
        breakdown = []
        breakdown.append((f"RF structures ({self.get_num_structures()}x)", self.get_cost_structures()))
        breakdown.append((f"Klystrons and modulators ({self.get_num_klystrons()}x, {round(self.get_klystron_average_power()/1e3,0)} kW avg per)", self.get_cost_klystrons()))
        return (self.name, breakdown)
        