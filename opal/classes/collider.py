from opal import Runnable, Linac, BeamDeliverySystem, InteractionPoint, Event
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from copy import deepcopy

class Collider(Runnable):
    
    # constructor
    def __init__(self, linac1=None, bds1=None, ip=None, linac2=None, bds2=None):
        
        # check element classes, then assemble
        assert(isinstance(linac1, Linac))
        assert(isinstance(bds1, BeamDeliverySystem))
        assert(isinstance(ip, InteractionPoint))
        
        self.linac1 = linac1
        self.linac2 = linac2
        self.bds1 = bds1
        self.bds2 = bds2
        self.ip = ip
    
    
    # run simulation
    def run(self, runname=None, shots=1, savedepth=2, verbose=True, overwrite=False):
        
        # copy second arm if undefined
        if self.linac2 is None:
            self.linac2 = deepcopy(self.linac1)
        if self.bds2 is None:
            self.bds2 = deepcopy(self.bds1)
        
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
        event = self.ip.run(self.linac1, self.linac2, self.runname + "/ip", allByAll=True)
        
        # return beams from last shot
        return beam1, beam2, event
    
    
    # plot the luminosity distribution
    def plotLuminosity(self):
        
        # load luminosities
        files = self.ip.runData()
        Nevents = len(files)
        lumi_geom = np.empty(Nevents)
        lumi_full = np.empty(Nevents)
        lumi_peak = np.empty(Nevents)
        for i in range(Nevents):
            event = Event.load(files[i], loadBeams=False)
            lumi_geom[i] = event.geometricLuminosity()
            lumi_full[i] = event.fullLuminosity()
            lumi_peak[i] = event.peakLuminosity()
        
        # calculate fraction of events at zero luminosity
        frac_zero_full = np.mean(lumi_full==0)
        frac_zero_peak = np.mean(lumi_peak==0)
        
        # prepare figure
        fig, axs = plt.subplots(1,3)
        fig.set_figwidth(16)
        fig.set_figheight(3)
        logbins = np.logspace(32, 36, int(np.sqrt(Nevents)*3))
        
        # plot full luminosity
        axs[0].hist(lumi_full, bins=logbins, label=str(round(frac_zero_full*100))+"% no lumi.")
        axs[0].set_xscale("log")
        axs[0].set_xlabel('Full luminosity per crossing (m$^{-2}$)')
        axs[0].set_ylabel('Count per bin')
        axs[0].legend()
        
        # plot peak luminosity
        axs[1].hist(lumi_peak, bins=logbins, label=str(round(frac_zero_peak*100))+"% no lumi.")
        axs[1].set_xscale("log")
        axs[1].set_xlabel('Peak luminosity per crossing (m$^{-2}$)')
        axs[1].set_ylabel('Count per bin')
        axs[1].legend()
        
        # plot geometric luminosity
        axs[2].hist(lumi_geom, bins=logbins)
        axs[2].set_xscale("log")
        axs[2].set_xlabel('Geometric luminosity per crossing (m$^{-2}$)')
        axs[2].set_ylabel('Count per bin')
        
        
                                     
    