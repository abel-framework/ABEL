from abel import CONFIG, Beam, Trackable, Runnable
import scipy.constants as SI
import copy
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

class Beamline(Trackable, Runnable):
    
    def __init__(self, trackables=None):
        self.trackables = trackables
        
    
    def assemble_trackables(self):
        pass
    
    
    # get cumulative length
    def get_length(self):
        if self.trackables is None:
            self.assemble_trackables()
        L = 0
        for trackable in self.trackables:
            L += trackable.get_length()
        return L
    
    
    # perform tracking
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # assemble the trackables
        self.assemble_trackables()
        
        # perform element-wise tracking
        for trackable in self.trackables:
            beam = trackable.track(beam, savedepth-1, runnable, verbose)
        
        return beam
    
    
    
    ## SURVEY
    
    # make survey rectangles
    def survey_object(self):
        if self.trackables is None:
            self.assemble_trackables()
        objs = []
        for trackable in self.trackables:
            objs.append(trackable.survey_object())
        return objs
    
    # plot survey    
    def plot_survey(self):
         
        # setup figure
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(1)
        ax.set_xlabel('s (m)')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(-1, self.get_length()+1)
        ax.set_ylim(-1.5, 1.5)
        
        # add objects
        s = 0
        objs = self.survey_object()
        for obj in objs:
            
            # add objects
            obj.set_x(s)
            ax.add_patch(obj)
            
            # add to location
            s += obj.get_width()
        
        plt.show()
        
        