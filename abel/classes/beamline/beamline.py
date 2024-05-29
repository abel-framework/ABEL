from abc import abstractmethod
from abel import CONFIG, Beam, Trackable, Runnable
import scipy.constants as SI
import copy
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

class Beamline(Trackable, Runnable):
    
    def __init__(self, trackables=None, rep_rate=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
        self.trackables = trackables

        # bunch train pattern
        self.bunch_separation = bunch_separation # [s]
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains # [Hz]

        # cost
        self.cost_per_length_tunnel = 0.06e6 # [ILCU/m] FCC cost estimate
        
    
    @abstractmethod
    def assemble_trackables(self):
        pass

    
    # perform tracking
    def track(self, beam=None, savedepth=0, runnable=None, verbose=False):
        
        # assemble the trackables
        if self.trackables is None:                          # enable setting interstage.nom_energy individually
            self.assemble_trackables()
        
        # perform element-wise tracking
        for trackable in self.trackables:
            beam = trackable.track(beam, savedepth-1, runnable, verbose)
        
        return beam

    
    ## BUNCH TRAIN PATTERN
    
    def get_rep_rate_trains(self):
        return self.rep_rate_trains

    def rep_rate_average(self):
        if self.rep_rate_trains is not None:
            return self.num_bunches_in_train * self.rep_rate_trains
        else:
            return None

    def rep_rate_intratrain(self):
        if self.bunch_separation is not None:
            return 1/self.bunch_separation
        else:
            return None

    def train_duration(self):
        if self.bunch_separation is not None:
            return self.bunch_separation * (self.num_bunches_in_train-1)
        else:
            return None
    
    @abstractmethod
    def energy_usage(self):
        pass
    
    def wallplug_power(self):
        if self.rep_rate_average() is not None:
            return self.energy_usage() * self.rep_rate_average()
        else:
            return None
    
    ## LENGTH
    
    # get cumulative length
    def get_length(self):
        if self.trackables is None:
            self.assemble_trackables()
        L = 0
        for trackable in self.trackables:
            L += trackable.get_length()
        return L


    ## COST
    
    # get cumulative cost
    def get_cost(self):
        if self.trackables is None:
            self.assemble_trackables()

        # add all elements
        cost = 0
        for trackable in self.trackables:
            cost += trackable.get_cost()

        # add cost of tunnel
        cost += self.get_length() * self.cost_per_length_tunnel
        
        return cost
    
    
    
    
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
    def plot_survey(self, save_fig=False):
         
        # setup figure
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(1)
        ax.set_xlabel('s [m]')
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

        if save_fig:
            plot_path = self.run_path() + 'plots/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'survey' + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)
        