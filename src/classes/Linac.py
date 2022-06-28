from src.classes.Trackable import *
from src.classes.Beam import *
from src.classes.Source import *
from src.classes.Stage import *
import copy
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from matplotlib import pyplot as plt

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
            
    def length(self):
        L = 0
        for trackable in self.trackables:
            L += trackable.length()
        return L
    
            
    ## ENERGY CONSIDERATIONS
    
    def targetEnergy(self):
        E = 0
        for trackable in self.trackables:
            if isinstance(trackable, Source):
                E += trackable.energy()
            elif isinstance(trackable, Stage):
                E += trackable.energyGain()
        return E
    
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
        stageNumbers = np.empty((len(files), 1))
        
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
          
        
    def plotBeamEvolution(self):
        
        # calculate values
        vals, stageNums = self.evolutionFcn([Beam.charge, Beam.energy, Beam.relEnergySpread, Beam.bunchLength, Beam.normEmittanceX, Beam.normEmittanceY, Beam.beamSizeX, Beam.beamSizeY])
        Qs = vals[:,0]
        Es = vals[:,1]
        sigdeltas = vals[:,2]
        sigzs = vals[:,3]
        emnxs = vals[:,4]
        emnys = vals[:,5]
        sigxs = vals[:,6]
        sigys = vals[:,7]
        
        # plot evolution
        fig, axs = plt.subplots(2,3)
        fig.set_figwidth(20)
        fig.set_figheight(8)
        
        axs[0,0].plot(stageNums, Es / 1e9, '-')
        axs[0,0].set_xlabel('Stage number')
        axs[0,0].set_ylabel('Energy (GeV)')
        
        axs[0,1].plot(stageNums, sigdeltas * 100, '-')
        axs[0,1].set_xlabel('Stage number')
        axs[0,1].set_ylabel('Energy spread (%)')
        axs[0,1].set_yscale('log')
        
        axs[0,2].plot(stageNums, emnxs*1e6, '-', stageNums, emnys*1e6, '-')
        axs[0,2].set_xlabel('Stage number')
        axs[0,2].set_ylabel('Emittance, rms (mm mrad)')
        
        axs[1,0].plot(stageNums, Qs * 1e9, '-')
        axs[1,0].set_xlabel('Stage number')
        axs[1,0].set_ylabel('Charge (nC)')
        
        axs[1,1].plot(stageNums, sigzs*1e6, '-')
        axs[1,1].set_xlabel('Stage number')
        axs[1,1].set_ylabel('Bunch length (um)')
        
        axs[1,2].plot(stageNums, sigxs*1e6, '-', stageNums, sigys*1e6, '-')
        axs[1,2].set_xlabel('Stage number')
        axs[1,2].set_ylabel('Beam size, rms (um)')
        
        plt.show()
    