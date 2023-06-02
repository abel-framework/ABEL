from opal import Runnable, Linac, BeamDeliverySystem, InteractionPoint, Event
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from copy import deepcopy
from matplotlib import lines
from datetime import datetime

class Collider(Runnable):
    
    # constructor
    def __init__(self, linac1, ip, linac2=None):
        
        # check element classes, then assemble
        assert(isinstance(linac1, Linac))
        assert(isinstance(ip, InteractionPoint))
        
        self.linac1 = linac1
        self.linac2 = linac2
        self.ip = ip
        
        # extract driver linac
        if len(linac1.stages):
            self.driverLinac = linac1.stages[0].driverSource
        else:
            self.driverLinac = None
        
        self.costPerLength = 2e5 # [LCU/m]
        self.costPerEnergy = 0.15/3.6e6 # [LCU/J]
        self.targetIntegratedLuminosity = 1e46 # [m^-2] or 1/ab
    
    
    # calculate energy usage (per bunch crossing)
    def energyUsage(self):
        return self.linac1.energyUsage() + self.linac2.energyUsage()
    
    # full luminosity per crossing [m^-2]
    def fullLuminosityPerCrossing(self):
        files = self.ip.runData()
        Nevents = len(files)
        lumi_full = np.empty(Nevents)
        for i in range(Nevents):
            event = Event.load(files[i], loadBeams=False)
            lumi_full[i] = event.fullLuminosity()
        return np.median(lumi_full)
    
    # peak luminosity per crossing [m^-2]
    def peakLuminosityPerCrossing(self):
        files = self.ip.runData()
        Nevents = len(files)
        lumi_peak = np.empty(Nevents)
        for i in range(Nevents):
            event = Event.load(files[i], loadBeams=False)
            lumi_peak[i] = event.peakLuminosity()
        return np.median(lumi_peak)
        
    # full luminosity per power [m^-2/J]
    def fullLuminosityPerPower(self):
        return self.fullLuminosityPerCrossing() / self.energyUsage()
    
    # peak luminosity per power [m^-2/J]
    def peakLuminosityPerPower(self):
        return self.peakLuminosityPerCrossing() / self.energyUsage()
    
    # integrated energy usage (to reach target integrated luminosity)
    def integratedEnergyUsage(self):
        return self.targetIntegratedLuminosity / self.peakLuminosityPerPower()
    
    # integrated cost of energy
    def runningCost(self):
        return self.integratedEnergyUsage() * self.costPerEnergy
      
    # total length of linacs (TODO: add driver production length)
    def totalLength(self):
        Ltot = self.linac1.length() + self.linac2.length()
        if self.driverLinac is not None:
            Ltot += self.driverLinac.length()
        return Ltot
      
    # cost of construction
    def constructionCost(self):
        return self.totalLength() * self.costPerLength
    
    # total cost of construction and running
    def totalCost(self):
        return self.constructionCost() + self.runningCost()
    
    # run simulation
    def run(self, runname=None, shots=1, savedepth=2, verbose=True, overwrite=False, overwriteIP=False):
        
        # copy second arm if undefined
        if self.linac2 is None:
            self.linac2 = deepcopy(self.linac1)
        
        # define run name (generate if not given)
        if runname is None:
            self.runname = "collider_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
        
        # declare shots list
        self.shots = []
        
        # make base folder and clear tracking directory
        if not exists(self.runPath()):
            mkdir(self.runPath())
        
        # run first linac arm
        if verbose:
            print(">> LINAC #1")
        beam1 = self.linac1.run(self.runname + "/linac1", shots, savedepth, verbose, overwrite)
        
        # run second linac arm
        if verbose:
            print(">> LINAC #2")
        beam2 = self.linac2.run(self.runname + "/linac2", shots, savedepth, verbose, overwrite)
        
        # simulate collisions
        if verbose:
            print(">> INTERACTION POINT")
        event = self.ip.run(self.linac1, self.linac2, self.runname + "/ip", allByAll=True, overwrite=(overwrite or overwriteIP))
        
        # return beams from last shot
        return beam1, beam2
    
    
    # plot the distribution of luminosity per power
    def plotLuminosityPerPower(self, perPower=False):
        self.plotLuminosity(perPower=True)
    
    
    # plot the luminosity distribution
    def plotLuminosity(self, perPower=False):
        
        if perPower:
            norm = 1e4 * self.energyUsage()/1e6 # [100 J]
            normLabel = 'per power (cm$^{-2}$ s$^{-1}$ MW$^{-1}$)'
        else:
            norm = 1
            normLabel = 'per crossing (m$^{-2}$)'
            
        # load luminosities
        files = self.ip.runData()
        Nevents = len(files)
        lumi_geom = np.empty(Nevents)
        lumi_full = np.empty(Nevents)
        lumi_peak = np.empty(Nevents)
        for i in range(Nevents):
            event = Event.load(files[i], loadBeams=False)
            lumi_geom[i] = event.geometricLuminosity() / norm
            lumi_full[i] = event.fullLuminosity() / norm
            lumi_peak[i] = event.peakLuminosity() / norm
        
        # calculate fraction of events at zero luminosity
        frac_zero_geom = np.mean(lumi_geom==0)
        frac_zero_full = np.mean(lumi_full==0)
        frac_zero_peak = np.mean(lumi_peak==0)
        
        # prepare figure
        fig, axs = plt.subplots(1,2)
        fig.set_figwidth(18)
        fig.set_figheight(3.5)
        Nbins = max(10, int(np.sqrt(Nevents)*10))
        logbins = np.logspace(30, 37, Nbins)/norm
        
        # plot full luminosity
        axs[0].hist(lumi_geom, bins=logbins, color='aliceblue', label="Geometric ("+str(round(frac_zero_geom*100))+"% no lumi.)")
        axs[0].hist(lumi_full, bins=logbins, label="Full ("+str(round(frac_zero_full*100))+"% no lumi.)")
        axs[0].set_xscale("log")
        axs[0].set_xlabel(f"Full luminosity {normLabel}")
        axs[0].set_ylabel('Count per bin')
        axs[0].set_xlim([min(logbins), max(logbins)])
        
        # plot peak luminosity
        axs[1].hist(lumi_geom, bins=logbins, color='aliceblue', label="Geometric ("+str(round(frac_zero_geom*100))+"% no lumi.)")
        axs[1].hist(lumi_peak, bins=logbins, label="Peak 1% ("+str(round(frac_zero_peak*100))+"% no lumi.)")
        axs[1].set_xscale("log")
        axs[1].set_xlabel(f"Peak luminosity {normLabel}")
        axs[1].set_ylabel('Count per bin')
        axs[1].legend(loc='upper left')
        axs[1].set_xlim([min(logbins), max(logbins)])
        
        # add ILC (250 GeV) comparison
        if perPower:
            valILC250full = 6.15e31 # [cm^-2 s^-1 MW^-1] wall-plug power = 122 MW
            valILC250peak = 5.35e31 # [cm^-2 s^-1 MW^-1]
            valILC500full = 1.10e32 # [cm^-2 s^-1 MW^-1] wall-plug power = 163 MW
            valILC500peak = 6.41e31 # [cm^-2 s^-1 MW^-1]
            valCLIC380full = 2.09e32 # [cm^-2 s^-1 MW^-1] wall-plug power ≈ 110 MW
            valCLIC380peak = 1.18e32 # [cm^-2 s^-1 MW^-1]
            valCLIC3000full = 1.01e32 # [cm^-2 s^-1 MW^-1] wall-plug power = 582 MW
            valCLIC3000peak = 3.43e31 # [cm^-2 s^-1 MW^-1]
        else:
            valILC250full = 1.14e34 # [m^-2]
            valILC250peak = 9.96e33 # [m^-2]
            valILC500full = 2.74e34 # [m^-2]
            valILC500peak = 1.60e34 # [m^-2]
            valCLIC380full = 1.3e34 # [m^-2]
            valCLIC380peak = 7.39e33 # [m^-2]
            valCLIC3000full = 3.78e34 # [m^-2]
            valCLIC3000peak = 1.28e34 # [m^-2]
        axs[0].add_artist(lines.Line2D([valILC250full, valILC250full], axs[0].get_ylim(), linestyle='-', color='lightgreen', label='ILC 250'))
        axs[1].add_artist(lines.Line2D([valILC250peak, valILC250peak], axs[1].get_ylim(), linestyle='-', color='lightgreen', label='ILC 250'))
        axs[0].add_artist(lines.Line2D([valILC500full, valILC500full], axs[0].get_ylim(), linestyle='-', color='forestgreen', label='ILC 500'))
        axs[1].add_artist(lines.Line2D([valILC500peak, valILC500peak], axs[1].get_ylim(), linestyle='-', color='forestgreen', label='ILC 500'))
        axs[0].add_artist(lines.Line2D([valCLIC380full, valCLIC380full], axs[0].get_ylim(), linestyle='-', color='lightcoral', label='CLIC 380'))
        axs[1].add_artist(lines.Line2D([valCLIC380peak, valCLIC380peak], axs[1].get_ylim(), linestyle='-', color='lightcoral', label='CLIC 380'))
        axs[0].add_artist(lines.Line2D([valCLIC3000full, valCLIC3000full], axs[0].get_ylim(), linestyle='-', color='indianred', label='CLIC 3000'))
        axs[1].add_artist(lines.Line2D([valCLIC3000peak, valCLIC3000peak], axs[1].get_ylim(), linestyle='-', color='indianred', label='CLIC 3000'))
        axs[0].legend(loc='upper left')
        
        
                                     
    