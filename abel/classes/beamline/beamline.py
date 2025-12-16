# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abc import abstractmethod
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam
from abel.classes.trackable import Trackable
from abel.classes.runnable import Runnable
from abel.classes.cost_modeled import CostModeled
import scipy.constants as SI
import copy
import numpy as np

class Beamline(Trackable, Runnable, CostModeled):
    
    def __init__(self, num_bunches_in_train=1, bunch_separation=None, rep_rate_trains=None, trackables=None):
        
        # set bunch pattern
        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)

        # set trackables
        self.trackables = trackables
        
    
    @abstractmethod
    def assemble_trackables(self):
        
        # apply the bunch pattern to all trackables
        for i in range(len(self.trackables)):
            if self.bunch_separation is not None:
                assert self.trackables[i].bunch_separation is None or self.trackables[i].bunch_separation==self.bunch_separation, 'Mismatched bunch separation'
                self.trackables[i].bunch_separation = self.bunch_separation
            if self.num_bunches_in_train is not None:
                assert self.trackables[i].num_bunches_in_train is None or self.trackables[i].num_bunches_in_train==self.num_bunches_in_train, 'Mismatched number of bunches in train'
                self.trackables[i].num_bunches_in_train = self.num_bunches_in_train
            if self.rep_rate_trains is not None:
                assert self.trackables[i].rep_rate_trains is None or self.trackables[i].rep_rate_trains==self.rep_rate_trains, 'Mismatched train rep rate'
                self.trackables[i].rep_rate_trains = self.rep_rate_trains

    
    # perform tracking
    def track(self, beam=None, savedepth=0, runnable=None, verbose=False):
        
        # assemble the trackables
        if self.trackables is None:
            print("ASSEMBLE!", flush=True)
            self.assemble_trackables()
            print("ASSEMBLED.", flush=True)
            
        # perform element-wise tracking
        for trackable in self.trackables:
            print("trackable = ", trackable, flush=True)
            beam = trackable.track(beam, savedepth-1, runnable, verbose)
        print("Done with trackables.", flush=True)
        
        return beam

    
    ## BUNCH TRAIN PATTERN
    
    @abstractmethod
    def energy_usage(self):
        pass
    
    def wallplug_power(self):
        if self.get_rep_rate_average() is not None:
            return self.energy_usage() * self.get_rep_rate_average()
        else:
            return None

    @abstractmethod
    def get_nom_beam_power(self):
        pass
    
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

    def get_cost_breakdown_civil_construction(self):
        breakdown = []
        for trackable in self.trackables:
            breakdown.append((trackable.name, trackable.get_cost_civil_construction()))
        return ('Civil construction', breakdown)

    
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

        from matplotlib import pyplot as plt
        import os
        
        # setup figure
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(4)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        
        # initialize parameters
        x = 0
        y = 0
        angle = 0
        labels = []

        # get objects
        objs = self.survey_object()

        # extract secondary objects
        second_objs = None
        connect_to = None
        if isinstance(objs, tuple) and len(objs)==2 and isinstance(objs[1], tuple):
            second_objs = objs[1][0]
            connect_to = objs[1][1]
            objs = objs[0]

        # plot the objects
        for i, obj in enumerate(objs):
            
            xs0, ys0, final_angle, label, color = obj
            
            xs = x + xs0*np.cos(angle) + ys0*np.sin(angle)
            ys = y - xs0*np.sin(angle) + ys0*np.cos(angle)

            if label in labels:
                label = None
            else:
                labels.append(label)
                
            ax.plot(xs, ys, '-', color=color, label=label, linewidth=2)
            
            # add secondary objects
            if connect_to == i:
                x2 = x
                y2 = y
                angle2 = angle
                
                # get and iterate through objects
                second_objs.reverse()
                for second_obj in second_objs:
        
                    xs0_2, ys0_2, final_angle2, label, color = second_obj
                    
                    xs0_2 -= xs0_2[-1]
                    ys0_2 -= ys0_2[-1]
                    angle2 = angle2 + final_angle2
        
                    xs2 = x2 + xs0_2*np.cos(angle2) + ys0_2*np.sin(angle2)
                    ys2 = y2 - xs0_2*np.sin(angle2) + ys0_2*np.cos(angle2)

                    x2 = xs2[0]
                    y2 = ys2[0]
                    
                    if label in labels:
                        label = None
                    else:
                        labels.append(label)
                    ax.plot(xs2, ys2, '-', color=color, label=label, linewidth=2)
            
            x = xs[-1]
            y = ys[-1]
            angle = angle + final_angle
            

        ax.legend()
        plt.show()

        if save_fig:
            plot_path = self.run_path() + 'plots/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'survey' + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)


class NotAssembledError(Exception):
    "Exception class for ``Beamline`` to throw if ``self.trackables`` have not been assembled."
    pass
        
