# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.cost_modeled import CostModeled
import scipy.constants as SI
import copy
import numpy as np
from types import SimpleNamespace

class RFAcceleratorBasic(RFAccelerator):

    default_rf_frequency = RFAccelerator.default_rf_frequency
    default_fill_factor = RFAccelerator.default_fill_factor
    default_num_rf_cells = RFAccelerator.default_num_rf_cells
    default_operating_temperature = RFAccelerator.default_operating_temperature
    
    def __init__(self, length=None, nom_energy_gain=None, fill_factor=default_fill_factor, rf_frequency=default_rf_frequency, num_rf_cells=default_num_rf_cells, operating_temperature=default_operating_temperature):

        # run base class constructor
        super().__init__(length=length, nom_energy_gain=nom_energy_gain, num_rf_cells=num_rf_cells, fill_factor=fill_factor, rf_frequency=rf_frequency, operating_temperature=operating_temperature)

        self.structure = SimpleNamespace()
        self.structure.length = None
        self.structure.rise_time = None
        self.structure.fill_time = None
        self.structure.pulse_length_total = None
        self.structure.power = None
        self.structure.rf_efficiency = None

        self.wallplug_energy_per_bunch_rf = None
        self.wallplug_energy_per_bunch_cooling = None
        self.heat_energy_per_bunch_heating = None

        # output Twiss
        self.beta_x = None
        self.beta_y = None
        
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # store data used power-flow/power use/etc modelling
        self.bunch_charge = beam0.abs_charge()
        if self.bunch_separation is None:
            self.bunch_separation = beam0.bunch_separation
        if self.num_bunches_in_train is None:
            self.num_bunches_in_train = beam0.num_bunches_in_train

        # calculate the number of klystrons required
        self.calculate_structure()

        # perform energy increase
        beam = copy.deepcopy(beam0)
        beam.set_Es(beam0.Es() + self.nom_energy_gain)

        # set output Twiss if given
        if self.beta_x is not None and self.beta_y is not None:
            
            # transport phase spaces to waist (in each plane)
            ds_x = beam.alpha_x()/beam.gamma_x()
            ds_y = beam.alpha_y()/beam.gamma_y()
            
            # find waist beta functions (in each plane)
            Rx = np.eye(4)
            Rx[0,1] = ds_x
            Rx[2,3] = ds_x
            beamx = copy.deepcopy(beam)
            beamx.set_transverse_vector(np.dot(Rx, beamx.transverse_vector()))
    
            Ry = np.eye(4)
            Ry[0,1] = ds_y
            Ry[2,3] = ds_y
            beamy = copy.deepcopy(beam)
            beamy.set_transverse_vector(np.dot(Ry, beamy.transverse_vector()))
            
            # scale the waist phase space by beta functions
            X = beamx.transverse_vector()
            Y = beamy.transverse_vector()
            X[0,:] = X[0,:] * np.sqrt(self.beta_x/beamx.beta_x())
            X[1,:] = X[1,:] / np.sqrt(self.beta_x/beamx.beta_x())
            X[2,:] = Y[2,:] * np.sqrt(self.beta_y/beamy.beta_y())
            X[3,:] = Y[3,:] / np.sqrt(self.beta_y/beamy.beta_y()) 
            beam.set_transverse_vector(X)
        
        return super().track(beam, savedepth, runnable, verbose)


    @RFAccelerator.structure_length.getter
    def structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self.structure.length
        
    def get_structure_power(self) -> float:
        "Get the peak power [W] required for a single RF structure in the given configuration."
        return self.structure.power


    def calculate_structure(self):
        
        # angular frequency
        omega = 2*np.pi*self.rf_frequency
        
        # cavity R/Q per length (shape dependent only)
        norm_R_upon_Q = (1/6)*SI.mu_0*omega # TODO: implement that this could be different

        # quality factor (temperature and frequency dependent)
        Q = 3.59e7*np.exp(max(5.38, 7.39-0.627*np.log(self.operating_temperature))) / np.sqrt(omega)
        
        # structure length
        #phase_advance_cell = 2/3*np.pi
        phase_advance_cell = 1/2*np.pi # TODO: implement that this could be different
        cell_length = phase_advance_cell*SI.c/omega
        self.structure.length = cell_length*self.num_rf_cells
        
        # power lost to wall losses and beam loading (per length)
        dP_dz_losses = self.gradient_structure**2 / (norm_R_upon_Q * Q)
        if self.bunch_separation is not None and self.bunch_separation > 0:
            dP_dz_beamloading = self.gradient_structure * (self.bunch_charge / self.bunch_separation)
        else:
            dP_dz_beamloading = 0.0
        dP_dz_total = dP_dz_losses + dP_dz_beamloading

        # power input required
        self.structure.power = dP_dz_total * self.structure_length
        
        # energy per length in structures (W)
        structure_energy_per_length = self.gradient_structure**2/(omega*norm_R_upon_Q)

        # cavity rise and fill time (approx.)
        self.structure.rise_time = self.structure_length/SI.c
        self.structure.fill_time = structure_energy_per_length * self.structure_length / self.structure.power
        
        # beam loading efficiency
        self.structure.pulse_length_total = self.structure.rise_time + self.structure.fill_time + self.train_duration
        
        energy_per_train_rf = self.structure.pulse_length_total * self.structure.power * self.num_structures
        wallplug_energy_per_train_rf = energy_per_train_rf / self.efficiency_wallplug_to_rf
        self.wallplug_energy_per_bunch_rf = wallplug_energy_per_train_rf / self.num_bunches_in_train

        # rf efficiency
        self.structure.rf_efficiency = self.num_bunches_in_train*self.bunch_separation/self.structure.pulse_length_total
        
        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 300 # [K]
        efficiency_cooling_fraction_of_carnot = 0.44 # C3 estimate is 0.44 @ 77K, while ILC estimate is 0.22 @2K
        inv_efficiency_cooling = max(0.0, (room_temperature-self.operating_temperature)/(efficiency_cooling_fraction_of_carnot*self.operating_temperature))

        # total energy lost into the wall
        energy_per_train_heating = dP_dz_losses * self.structure.pulse_length_total * self.structure_length * self.num_structures
        self.heat_energy_per_bunch_heating = energy_per_train_heating / self.num_bunches_in_train
        self.wallplug_energy_per_bunch_cooling = self.heat_energy_per_bunch_heating * inv_efficiency_cooling

    
    # implement required abstract methods
    
    def heat_energy_at_cryo_temperature(self) -> float:
        "Heat energy dissipated per bunch at cryo temperature [J]"
        return self.heat_energy_per_bunch_heating
    
    def energy_usage_rf(self) -> float:
        "Energy usage per bunch for RF [J]"
        return self.wallplug_energy_per_bunch_rf

    def energy_usage_cooling(self) -> float:
        "Energy usage per bunch for cooling [J]"
        return self.wallplug_energy_per_bunch_cooling

    def energy_usage(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per bunch total [J]"
        
        # return energy usage per bunch
        return self.energy_usage_rf() + self.energy_usage_cooling()
        
