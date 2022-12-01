from opal import CONFIG, Beam, Trackable
from opal.utilities import SI
import copy
from os import listdir, remove, mkdir
from os.path import isfile, join, exists
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

class Linac(Trackable):
    
    def __init__(self, trackables, runname=None):
        
        # populate trackables
        self.trackables = trackables
        
        # make run name if not given
        if runname is None:
            self.runname = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = runname
            
    
    
    # get cumulative length
    def length(self):
        L = 0
        for trackable in self.trackables:
            L += trackable.length()
        return L
    
    
    # generate track path
    def trackPath(self):
        return CONFIG.trackdata_path + self.runname + "/"
    
    
    # perform tracking
    def track(self, beam=None, quiet=True):
        
        # make and clear tracking directory
        if not exists(self.trackPath()):
            mkdir(self.trackPath())
        self.clearTrackData()
            
        # perform tracking
        for trackable in self.trackables:
            beam = trackable.track(beam)
            beam.save(self)
            if not quiet:
                print('Tracked element ' + str(beam.trackableNumber) + ' (' + type(trackable).__name__ + ', stage ' + str(beam.stageNumber) + ')')
        
        return beam
    
    
    # get tracking data
    def trackData(self):
        files = [f for f in listdir(self.trackPath()) if isfile(join(self.trackPath(), f))]    
        files.sort()
        return files
    
    
    # clear tracking data
    def clearTrackData(self):
        for file in self.trackData():
            remove(self.trackPath() + file)
    
    
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
        
        