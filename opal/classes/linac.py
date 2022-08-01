from opal import Beam, Trackable, Source, Stage
from opal.utilities import SI
import copy
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from matplotlib import pyplot as plt
import numpy as np

class Linac(Trackable):
    
    def __init__(self, source = None, stage = None, interstage = None, Nstages = 1):
        
        self.trackingfolder = 'trackdata/'
        
        self.trackables = [None] * (2*Nstages)
        
        # add source
        self.trackables[0] = source
        
        for i in range(Nstages):
            
            # add stages
            self.trackables[1+2*i] = copy.deepcopy(stage)
            
            # add interstages
            if i < Nstages-1:
                self.trackables[2+2*i] = copy.deepcopy(interstage)
                self.trackables[2+2*i].E0 = source.energy() + (i+1) * stage.energyGain()
                
                
    def track(self, beam = None):
        
        # make and clear tracking directory
        if not exists(self.trackingfolder):
            mkdir(self.trackingfolder)
        self.clearTrackingData()
            
        # perform tracking
        for trackable in self.trackables:
            beam = trackable.track(beam)
            beam.save(self)
        
        return beam
            
    def length(self):
        L = 0
        for trackable in self.trackables:
            L += trackable.length()
        return L
    
            
    ## ENERGY CONSIDERATIONS
    
    def targetEnergy(self):
        E = 0
        Es = np.array([]);
        for trackable in self.trackables:
            if isinstance(trackable, Source):
                E += trackable.energy()
            elif isinstance(trackable, Stage):
                E += trackable.energyGain()
            Es = np.append(Es, E)
        return E, Es
    
    def effectiveGradient(self):
        return self.targetEnergy()/self.length()
    
    ## SURVEY
    
    # make survey rectangles
    def plotObject(self):
        objs = []
        for trackable in self.trackables:
            objs.append(trackable.plotObject())
        return objs
    
    # plot survey    
    def plotSurvey(self):
         
        # setup figure
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(1)
        ax.set_xlabel('s (m)')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(-1, self.length()+1)
        ax.set_ylim(-1.5, 1.5)
        
        # add objects
        s = 0
        objs = self.plotObject()
        for obj in objs:
            
            # add objects
            obj.set_x(s)
            ax.add_patch(obj)
            
            # add to location
            s += obj.get_width()
        
        plt.show()
        
        
    ## TRACKING DATA
    
    # list tracking data
    def trackingData(self):
        files = [f for f in listdir(self.trackingfolder) if isfile(join(self.trackingfolder, f))]    
        files.sort()
        return files
    
    # clear tracking data
    def clearTrackingData(self):
        for file in self.trackingData():
            remove(self.trackingfolder + file)
    
    
    
    ## PLOT EVOLUTION
    
    # apply function to all beam files
    def evolutionFcn(self, fcns):
        
        # find tracking data
        files = self.trackingData()
        
        # declare data structure
        vals = np.empty((len(files), len(fcns)))
        stageNumbers = np.empty(len(files))
        
        # go through files
        for i in range(len(files)):

            # load phase space
            beam = Beam.load(self.trackingfolder + files[i])

            # find stage number
            stageNumbers[i] = beam.stageNumber
            
            # apply all functions
            for j in range(len(fcns)):
                vals[i,j] = fcns[j](beam)
                
        return vals, stageNumbers
 

    # apply waterfall function to all beam files
    def waterfallFcn(self, fcns, edges, args = None):
        
        # find tracking data
        files = self.trackingData()
        
        # declare data structure
        waterfalls = [None] * len(fcns)
        bins = [None] * len(fcns)
        for j in range(len(fcns)):
            waterfalls[j] = np.empty((len(edges[j])-1, len(files)))
        stageNumbers = np.empty(len(files))
        
        # go through files
        for i in range(len(files)):

            # load phase space
            beam = Beam.load(self.trackingfolder + files[i])

            # find stage number
            stageNumbers[i] = beam.stageNumber
            
            # get all waterfalls
            for j in range(len(fcns)):
                if args[j] is None:
                    waterfalls[j][:,i], bins[j] = fcns[j](beam, bins=edges[j])
                else:
                    waterfalls[j][:,i], bins[j] = fcns[j](beam, args[j][i], bins=edges[j])
                
        return waterfalls, stageNumbers, bins
             
        
    def plotEvolution(self):
        
        # calculate values
        vals, stageNums = self.evolutionFcn([Beam.absCharge, \
                                             Beam.energy, Beam.relEnergySpread, \
                                             Beam.bunchLength, Beam.offsetZ, \
                                             Beam.normEmittanceX, Beam.normEmittanceY, \
                                             Beam.beamSizeX, Beam.beamSizeY, \
                                             Beam.offsetX, Beam.offsetY])
        Qs = vals[:,0]
        Es = vals[:,1]
        sigdeltas = vals[:,2]
        sigzs = vals[:,3]
        z0s = vals[:,4]
        emnxs = vals[:,5]
        emnys = vals[:,6]
        sigxs = vals[:,7]
        sigys = vals[:,8]
        x0s = vals[:,9]
        y0s = vals[:,10]
        
        # target energies
        _, Es_target = self.targetEnergy()
        deltas = Es/Es_target - 1
        
        # initial charge
        Q0 = Qs[0]
        
        # plot evolution
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        
        axs[0,0].plot(stageNums, Es_target / 1e9, ':', color='gray')
        axs[0,0].plot(stageNums, Es / 1e9, '-')
        axs[0,0].set_xlabel('Stage number')
        axs[0,0].set_ylabel('Energy (GeV)')
        
        axs[1,0].plot(stageNums, sigdeltas * 100, '-')
        axs[1,0].set_xlabel('Stage number')
        axs[1,0].set_ylabel('Energy spread (%)')
        axs[1,0].set_yscale('log')
        
        axs[2,0].plot(stageNums, np.zeros(deltas.shape), ':', color='gray')
        axs[2,0].plot(stageNums, deltas * 100, '-')
        axs[2,0].set_xlabel('Stage number')
        axs[2,0].set_ylabel('Energy offset (%)')
        
        axs[0,1].plot(stageNums, Q0 * np.ones(Qs.shape) * 1e9, ':', color='gray')
        axs[0,1].plot(stageNums, Qs * 1e9, '-')
        axs[0,1].set_xlabel('Stage number')
        axs[0,1].set_ylabel('Charge (nC)')
        
        axs[1,1].plot(stageNums, sigzs*1e6, '-')
        axs[1,1].set_xlabel('Stage number')
        axs[1,1].set_ylabel('Bunch length (um)')
        
        axs[2,1].plot(stageNums, z0s*1e6, '-')
        axs[2,1].set_xlabel('Stage number')
        axs[2,1].set_ylabel('Longitudinal offset (um)')
        
        axs[0,2].plot(stageNums, emnxs*1e6, '-', stageNums, emnys*1e6, '-')
        axs[0,2].set_xlabel('Stage number')
        axs[0,2].set_ylabel('Emittance, rms (mm mrad)')
        
        axs[1,2].plot(stageNums, sigxs*1e6, '-', stageNums, sigys*1e6, '-')
        axs[1,2].set_xlabel('Stage number')
        axs[1,2].set_ylabel('Transverse beam size, rms (um)')
        
        axs[2,2].plot(stageNums, np.zeros(x0s.shape), ':', color='gray')
        axs[2,2].plot(stageNums, x0s*1e6, '-', stageNums, y0s*1e6, '-')
        axs[2,2].set_xlabel('Stage number')
        axs[2,2].set_ylabel('Transverse offset (um)')
        
        plt.show()
    
    
    
    # density plots
    def plotWaterfalls(self):
        
        # calculate values
        tedges = np.arange(-100e-6, 0e-6, 1e-6) / SI.c
        deltaedges = np.arange(-0.05, 0.05, 0.002)
        _, E0s = self.targetEnergy()
        waterfalls, stageNums, bins = self.waterfallFcn([Beam.currentProfile, Beam.relEnergyProfile], \
                                                        [tedges, deltaedges], \
                                                        [None, E0s])
        
        # prepare figure
        fig, axs = plt.subplots(1,2)
        fig.set_figwidth(14)
        fig.set_figheight(4)
        
        # current profile
        Is = waterfalls[0]
        ts = bins[0]
        c0 = axs[0].pcolor(stageNums, ts*SI.c*1e6, -Is/1e3, cmap='GnBu')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_xlabel('Stage number')
        axs[0].set_ylabel('Longitudinal position (um)')
        cbar0.ax.set_ylabel('Beam current (kA)')
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(stageNums, deltas*1e2, -dQddeltas*1e7, cmap='GnBu')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_xlabel('Stage number')
        axs[1].set_ylabel('Energy offset (%)')
        cbar1.ax.set_ylabel('Spectral density (nC/%)')
        
        plt.show()