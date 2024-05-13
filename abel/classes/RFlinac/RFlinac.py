#!/usr/bin/env python3

from abel import Trackable

import CLICopti

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class RFlinac(Trackable):
    """
    Class modelling a conventional warm(ish) travelling wave linac, typically used for drivers or injectors.
    TODO: It's an abstract class, meant to be superseeded by an implementation defining a specific RFstructure
    """

    def __init__(self, length=None, num_structures=None, gradient=None, beam_pulse_length=None,beam_current=0.0, RF_structure=None):
        "Initialize the base class, computing fillFactor and total voltage."

        if RF_structure == None:
            #TODO TEMP: Just define the CLIC_G as the default and ignore everything else
            self._RF_structure = CLICopti.RFStructure.AccelStructure_CLICG(24)
        self._RF_structure.calc_g_integrals(1000)

        if length==None or num_structures==None or gradient==None:
            raise ValueError("Must specify all 3 from length, num_structures, gradient so that fill factor can be properly calculated")

        #TODO: Decide on the meaning of beam_pulse_length = None - maybe just auto?
        #      Also add check_pulse_length?

        self.length = length
        self.num_structures = num_structures
        self.fill_factor = self.num_structures * self._RF_structure.getL() / self.length
        if self.fill_factor > 1:
            raise ValueError(f"Fill factor = {self.fill_factor} > 1, this is not physically possible")
        self.gradient = gradient
        self.voltage_structure = gradient*self._RF_structure.getL()
        self.voltage = self.voltage_structure*self.num_structures

        self.beam_pulse_length = beam_pulse_length
        self.beam_current = beam_current

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        #TODO: Check current etc.
        beam.set_Es(beam.Es()+self.voltage)
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self):
        #return super().get_length()
        return self.length
    
    def survey_object(self):
        #return super().survey_object()
        #TODO: This was copied from source; maybe it would be a decent default implementation too?
        rect = matplotlib.patches.Rectangle((0, -0.5), self.get_length(), 1)
        return rect
    

    ## RF structure model
    def getStructurePower(self):
        "Get the peak power required for a single RF structure"
        if self.beam_current == 0.0:
            P = self._RF_structure.getPowerUnloaded(self.voltage_structure)
        else:
            P = self._RF_structure.getPowerLoaded(self.voltage_structure, self.beam_current)
        return P
    
    def getStructurePulseEnergy(self):
        "Get the energy requirements for a single pulse for one structure"
        #Note: Model is assuming a pulse like from plotPowerProfile(),
        #      which assumes drive beam in that pulse end is a mirror of the start
        return (self._RF_structure.getTrise() + self._RF_structure.getTfill() + self.beam_pulse_length) * \
            self.getStructurePower()

    def getRFEfficiency(self):
        return self._RF_structure.getTotalEfficiency(self.getStructurePower(), self.beam_current, self.beam_pulse_length)
    
    def getPulseLengthTotal(self):
        if self.beam_pulse_length == None:
            raise ValueError("Must set beam pulse length")
        return 2 * (self._RF_structure.getTrise() + self._RF_structure.getTfill()) \
                 + self.beam_pulse_length

    def getMaxPulseLength(self):
        "Calculates the max beam_pulse_length before exceeding gradient limits"
        return self._RF_structure.getMaxAllowableBeamTime_dT(self.getStructurePower(), self.beam_current, useLoadedField=False)


    def plotGradientProfile(self):
        z = self._RF_structure.getZ_all()
        tit = f"{type(self._RF_structure).__name__}, V={self.voltage_structure/1e6:.1f} [MV]"
        if self.beam_current == None:
            Ez = self._RF_structure.getEz_unloaded_all(self.getStructurePower())
        else:
            Ez = self._RF_structure.getEz_loaded_all(self.getStructurePower(), self.beam_current)
            tit += f", I={self.beam_current:.1f} [A]"
        
        plt.subplots()
        plt.plot(z*1e3,Ez/1e6)
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$E_z$ [MV/m]')

        plt.title(tit)
    
    def plotPowerProfile(self):
        t = np.linspace(0,self.getPulseLengthTotal(), 100)
        P = np.empty_like(t)
        breakoverPower = self._RF_structure.getBreakoverPower(self.getStructurePower(),self.beam_current)
        for (iT, T) in enumerate(t):
            P[iT] = self._RF_structure.getP_t(T,self.getStructurePower(),self.beam_pulse_length,self.beam_current,breakoverPower)
        
        plt.subplots()
        plt.plot(t*1e9,P/1e6)
        plt.xlabel('Time [ns]')
        plt.ylabel('$P_{in}$ [MW]')
        tit = f"{type(self._RF_structure).__name__}, V={self.voltage_structure/1e6:.1f} [MV]"
        if self.beam_current != None:
            tit += f", I={self.beam_current:.1f} [A]"
        plt.title(tit)
    
    ##Klystron model
    #TODO

