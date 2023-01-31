from opal import Runnable, Beam, Beamline, Source, Stage, Interstage, BeamDeliverySystem
from opal.utilities import SI
from copy import deepcopy
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

class Linac(Beamline):
    
    def __init__(self, source=None, stage=None, interstage=None, bds=None, Nstages=1, alternatingInterstagePolarity=False):
        
        # check element classes, then assemble
        assert(isinstance(source, Source))
        if stage is not None:
            assert(isinstance(stage, Stage))
        if interstage is not None:
            assert(isinstance(interstage, Interstage))
        if bds is not None:
            assert(isinstance(bds, BeamDeliverySystem))
        
        # save as variables
        self.source = source
        self.bds = bds
        self.stages = [None]*Nstages
        self.interstages = [None]*(Nstages-1)
        
        # declare list of trackables
        trackables = [None] * (1 + Nstages + max(0,Nstages-1) + int(bds is not None))
        
        # add source
        trackables[0] = source
        
        # add stages
        for i in range(Nstages):
            
            # add stages
            stage_instance = deepcopy(stage)
            trackables[1+2*i] = stage_instance
            self.stages[i] = stage_instance
            
            # add interstages
            if i < Nstages-1:
                interstage_instance = deepcopy(interstage)
                interstage_instance.E0 = source.energy() + (i+1) * stage.energyGain()
                if alternatingInterstagePolarity:
                    interstage_instance.Bdip = (2*(i%2)-1)*interstage_instance.Bdip
                trackables[2+2*i] = interstage_instance
                self.interstages[i] = interstage_instance
        
        # add beam delivery system
        if bds is not None:
            bds.E0 = source.energy() + Nstages * stage.energyGain()
            assert(isinstance(bds, BeamDeliverySystem))
            trackables[max(1,2*Nstages)] = bds
        
        # run linac constructor
        super().__init__(trackables)
                
                 
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
    
    
    def energyUsage(self):
        Etot = self.source.energyUsage()
        for stage in self.stages:
            Etot += stage.energyUsage()
        return Etot
    
    def energyEfficiency(self):
        Etot_beam = self.finalBeam().totalEnergy()
        return Etot_beam/self.energyUsage()
        
    
    ## BEAMS
    
    # initial beam
    def initialBeam(self, shot=0):
        files = self.runData()
        files = files[shot-1]
        return Beam.load(files[0])
    
    # final beam
    def finalBeam(self, shot=0):
        files = self.runData()
        files = files[shot-1]
        return Beam.load(files[-1])

    
    ## PLOT EVOLUTION
    
    # apply function to all beam files
    def evolutionFcn(self, fcns):
        
        # find tracking data
        files = self.runData()
        
        # declare data structure
        Nsteps = len(files[0])
        stageNumbers = np.empty(Nsteps)
        ss = np.empty(Nsteps)
        vals_mean = np.empty((Nsteps, len(fcns)))
        vals_std = np.empty((Nsteps, len(fcns)))
        
        # go through files
        for i in range(Nsteps):
            
            # load beams and apply functions
            vals = np.empty((len(files), len(fcns)))
            for j in range(len(files)):
                beam = Beam.load(files[j][i])
                for k in range(len(fcns)):
                    vals[j,k] = fcns[k](beam)
            
            # calculate mean and standard dev
            for k in range(len(fcns)):
                vals_mean[i,k] = np.mean(vals[:,k])
                vals_std[i,k] = np.std(vals[:,k])
            
            # find stage number
            stageNumbers[i] = beam.stageNumber
            ss[i] = beam.location
        
        return ss, vals_mean, vals_std, stageNumbers
 

    # apply waterfall function to all beam files
    def waterfallFcn(self, fcns, edges, args = None):
        
        # find tracking data
        files = self.runData()
        files = files[-1]
        
        # declare data structure
        waterfalls = [None] * len(fcns)
        bins = [None] * len(fcns)
        for j in range(len(fcns)):
            waterfalls[j] = np.empty((len(edges[j])-1, len(files)))
        trackableNumbers = np.empty(len(files))
        
        # go through files
        for i in range(len(files)):

            # load phase space
            beam = Beam.load(files[i])

            # find stage number
            trackableNumbers[i] = beam.trackableNumber
            
            # get all waterfalls (apply argument is it exists)
            for j in range(len(fcns)):
                if args[j] is None:
                    waterfalls[j][:,i], bins[j] = fcns[j](beam, bins=edges[j])
                else:
                    waterfalls[j][:,i], bins[j] = fcns[j](beam, args[j][i], bins=edges[j])
                
        return waterfalls, trackableNumbers, bins
             
        
    def plotEvolution(self, useStageNumbers=False):
        
        # calculate values
        ss, vals_mean, vals_std, stageNums = self.evolutionFcn([Beam.absCharge, \
                                             Beam.energy, Beam.relEnergySpread, \
                                             Beam.bunchLength, Beam.offsetZ, \
                                             Beam.normEmittanceX, Beam.normEmittanceY, \
                                             Beam.betaX, Beam.betaY, \
                                             Beam.offsetX, Beam.offsetY])
        
        if useStageNumbers:
            long_axis = stageNums
            long_label = 'Stage number'
        else:
            long_axis = ss
            long_label = 'Location (m)'
        
        # mean values
        Qs = vals_mean[:,0]
        Es = vals_mean[:,1]
        sigdeltas = vals_mean[:,2]
        sigzs = vals_mean[:,3]
        z0s = vals_mean[:,4]
        emnxs = vals_mean[:,5]
        emnys = vals_mean[:,6]
        betaxs = vals_mean[:,7]
        betays = vals_mean[:,8]
        x0s = vals_mean[:,9]
        y0s = vals_mean[:,10]
        
        # errors
        Qs_error = vals_std[:,0]
        Es_error = vals_std[:,1]
        sigdeltas_error = vals_std[:,2]
        sigzs_error = vals_std[:,3]
        z0s_error = vals_std[:,4]
        emnxs_error = vals_std[:,5]
        emnys_error = vals_std[:,6]
        betaxs_error = vals_std[:,7]
        betays_error = vals_std[:,8]
        x0s_error = vals_std[:,9]
        y0s_error = vals_std[:,10]
        
        # target energies
        _, Es_target = self.targetEnergy()
        deltas = Es/Es_target - 1
        deltas_error = Es_error/Es_target
        
        # initial charge
        Q0 = Qs[0]
        
        # line format
        fmt = "-"
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.2
        
        # plot evolution
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        
        axs[0,0].plot(long_axis, Es_target / 1e9, ':', color=col0)
        axs[0,0].plot(long_axis, Es / 1e9, color=col1)
        axs[0,0].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((Es+Es_error, np.flip(Es-Es_error))) / 1e9, color=col1, alpha=af)
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_ylabel('Energy (GeV)')
        
        axs[1,0].plot(long_axis, sigdeltas * 100, color=col1)
        axs[1,0].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigdeltas+sigdeltas_error, np.flip(sigdeltas-sigdeltas_error))) * 100, color=col1, alpha=af)
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_ylabel('Energy spread (%)')
        axs[1,0].set_yscale('log')
        
        axs[2,0].plot(long_axis, np.zeros(deltas.shape), ':', color=col0)
        axs[2,0].plot(long_axis, deltas * 100, color=col1)
        axs[2,0].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((deltas+deltas_error, np.flip(deltas-deltas_error))) * 100, color=col1, alpha=af)
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_ylabel('Energy offset (%)')
        
        axs[0,1].plot(long_axis, Q0 * np.ones(Qs.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(long_axis, Qs * 1e9, color=col1)
        axs[0,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((Qs+Qs_error, np.flip(Qs-Qs_error))) * 1e9, color=col1, alpha=af)
        axs[0,1].set_xlabel(long_label)
        axs[0,1].set_ylabel('Charge (nC)')
        
        axs[1,1].plot(long_axis, sigzs*1e6, color=col1)
        axs[1,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigzs+sigzs_error, np.flip(sigzs-sigzs_error))) * 1e6, color=col1, alpha=af)
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_ylabel('Bunch length (um)')
        
        axs[2,1].plot(long_axis, z0s*1e6, color=col1)
        axs[2,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((z0s+z0s_error, np.flip(z0s-z0s_error))) * 1e6, color=col1, alpha=af)
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_ylabel('Longitudinal offset (um)')
        
        axs[0,2].plot(long_axis, np.ones(len(long_axis))*emnxs[0]*1e6, ':', color=col0)
        axs[0,2].plot(long_axis, np.ones(len(long_axis))*emnys[0]*1e6, ':', color=col0)
        axs[0,2].plot(long_axis, emnxs*1e6, color=col1)
        axs[0,2].plot(long_axis, emnys*1e6, color=col2)
        axs[0,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((emnxs+emnxs_error, np.flip(emnxs-emnxs_error))) * 1e6, color=col1, alpha=af)
        axs[0,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((emnys+emnys_error, np.flip(emnys-emnys_error))) * 1e6, color=col2, alpha=af)
        axs[0,2].set_xlabel(long_label)
        axs[0,2].set_ylabel('Emittance, rms (mm mrad)')
        axs[0,2].set_yscale('log')
        
        axs[1,2].plot(long_axis, np.sqrt(Es_target/Es_target[0])*betaxs[0]*1e3, ':', color=col0)
        axs[1,2].plot(long_axis, betaxs*1e3, color=col1)
        axs[1,2].plot(long_axis, betays*1e3, color=col2)
        axs[1,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((betaxs+betaxs_error, np.flip(betaxs-betaxs_error))) * 1e3, color=col1, alpha=af)
        axs[1,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((betays+betays_error, np.flip(betays-betays_error))) * 1e3, color=col2, alpha=af)
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_ylabel('Beta function (mm)')
        axs[1,2].set_yscale('log')
        
        axs[2,2].plot(long_axis, np.zeros(x0s.shape), ':', color=col0)
        axs[2,2].plot(long_axis, x0s*1e6, color=col1)
        axs[2,2].plot(long_axis, y0s*1e6, color=col2)
        axs[2,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((x0s+x0s_error, np.flip(x0s-x0s_error))) * 1e6, color=col1, alpha=af)
        axs[2,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((y0s+y0s_error, np.flip(y0s-y0s_error))) * 1e6, color=col2, alpha=af)
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_ylabel('Transverse offset (um)')
        
        plt.show()
    
    
    
    # density plots
    def plotWaterfalls(self):
        
        # calculate values
        beam0 = self.initialBeam()
        Nbins = int(np.sqrt(beam0.Npart()*2))
        nsig = 5
        tedges = np.arange(-100e-6, 0e-6, 1e-6) / SI.c
        deltaedges = np.linspace(-0.05, 0.05, Nbins)
        xedges = (nsig*beam0.beamSizeX() + abs(beam0.offsetX()))*np.linspace(-1, 1, Nbins)
        yedges = (nsig*beam0.beamSizeY() + abs(beam0.offsetY()))*np.linspace(-1, 1, Nbins)
        _, E0s = self.targetEnergy()
        waterfalls, trackableNumbers, bins = self.waterfallFcn([Beam.currentProfile, Beam.relEnergyProfile, Beam.transverseProfileX, Beam.transverseProfileY], \
                                                        [tedges, deltaedges, xedges, yedges], \
                                                        [None, E0s, None, None])
        
        # prepare figure
        fig, axs = plt.subplots(4,1)
        fig.set_figwidth(8)
        fig.set_figheight(11)
        
        # current profile
        Is = waterfalls[0]
        ts = bins[0]
        c0 = axs[0].pcolor(trackableNumbers, ts*SI.c*1e6, -Is/1e3, cmap='GnBu')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_ylabel('Longitudinal position (um)')
        cbar0.ax.set_ylabel('Beam current (kA)')
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(trackableNumbers, deltas*1e2, -dQddeltas*1e7, cmap='GnBu')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_ylabel('Energy offset (%)')
        cbar1.ax.set_ylabel('Spectral density (nC/%)')
        
        densityX = waterfalls[2]
        xs = bins[2]
        c2 = axs[2].pcolor(trackableNumbers, xs*1e6, -densityX*1e3, cmap='GnBu')
        cbar2 = fig.colorbar(c2, ax=axs[2])
        axs[2].set_ylabel('Horizontal position (um)')
        cbar2.ax.set_ylabel('Charge density (nC/um)')
        
        densityY = waterfalls[3]
        ys = bins[3]
        c3 = axs[3].pcolor(trackableNumbers, ys*1e6, -densityY*1e3, cmap='GnBu')
        cbar3 = fig.colorbar(c3, ax=axs[3])
        axs[3].set_xlabel('Trackable element number')
        axs[3].set_ylabel('Vertical position (um)')
        cbar3.ax.set_ylabel('Charge density (nC/um)')
        
        plt.show()