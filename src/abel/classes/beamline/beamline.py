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
            self.assemble_trackables()
        
        # perform element-wise tracking
        for trackable in self.trackables:
            beam = trackable.track(beam, savedepth-1, runnable, verbose)
        
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


    def plot_evolution_recursive(self, shot=None):

        """
        Plot the full evolution of various beam parameters.
        """

        from matplotlib import pyplot as plt
        from types import SimpleNamespace

        # select the shots
        if shot is None:
            shots = range(self.num_shots)
        else:
            shots = [shot]

        # set up evolution data structure
        evol = SimpleNamespace()
        evol.location = np.array([])

        # first calculate the locations
        for trackable in self[shots[0]].trackables:
            if hasattr(trackable, 'evolution') and hasattr(trackable.evolution, 'location'):
                evol.location = np.append(evol.location, trackable.evolution.location)

        # declare the data structure to contain all the data
        vals = np.zeros((8,len(shots),len(evol.location)))

        # extract all the data
        for i, shot in enumerate(shots):
            ind0 = 0
            for trackable in self[shot].trackables:
                if hasattr(trackable, 'evolution') and hasattr(trackable.evolution, 'location'):
                    inds = range(ind0, ind0+len(np.array(trackable.evolution.location)))
                    ind0 = max(inds)+1
                    vals[0,i,inds] = np.array(trackable.evolution.energy)
                    vals[1,i,inds] = np.array(trackable.evolution.charge)
                    vals[2,i,inds] = np.array(trackable.evolution.emit_nx)
                    vals[3,i,inds] = np.array(trackable.evolution.emit_ny)
                    vals[4,i,inds] = np.array(trackable.evolution.rel_energy_spread)
                    vals[5,i,inds] = np.array(trackable.evolution.bunch_length)
                    vals[6,i,inds] = np.array(trackable.evolution.beam_size_x)
                    vals[7,i,inds] = np.array(trackable.evolution.beam_size_y)

        # get the mean
        val_mean = np.mean(vals, axis=1)
        evol.energy = val_mean[0,:]
        evol.charge = val_mean[1,:]
        evol.emit_nx = val_mean[2,:]
        evol.emit_ny = val_mean[3,:]
        evol.rel_energy_spread = val_mean[4,:]
        evol.bunch_length = val_mean[5,:]
        evol.beam_size_x = val_mean[6,:]
        evol.beam_size_y = val_mean[7,:]
        
        # get the standard deviation
        val_std = np.std(vals, axis=1)
        evol_std = SimpleNamespace()
        evol_std.energy = val_std[0,:]
        evol_std.charge = val_std[1,:]
        evol_std.emit_nx = val_std[2,:]
        evol_std.emit_ny = val_std[3,:]
        evol_std.rel_energy_spread = val_std[4,:]
        evol_std.bunch_length = val_std[5,:]
        evol_std.beam_size_x = val_std[6,:]
        evol_std.beam_size_y = val_std[7,:]
        
        # prepare plot
        fig = plt.figure(figsize=(16, 11))
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.2
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]
        
        # define a 4x2 grid layout
        gs = fig.add_gridspec(nrows=4, ncols=2)
        
        # plot energy
        ax = fig.add_subplot(gs[0, 0])
        ax.fill_between(evol.location, (evol.energy + evol_std.energy)/1e9, (evol.energy-evol_std.energy)/1e9, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.energy / 1e9, color=col1)
        ax.set_ylabel('Energy [GeV]')
        ax.set_xlim(long_limits)
        
        # plot charge
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(evol.location, abs(evol.charge[0]) * np.ones(evol.location.shape) * 1e9, ':', color=col0)
        ax.fill_between(evol.location, (evol.charge + evol_std.charge)*1e9, (evol.charge-evol_std.charge)*1e9, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, abs(evol.charge) * 1e9, color=col1)
        ax.set_ylabel('Charge [nC]')
        ax.set_xlim(long_limits)
        ax.set_ylim(0, abs(evol.charge[0]) * 1.1 * 1e9)
        
        # plot energy spread
        ax = fig.add_subplot(gs[1, 0])
        ax.fill_between(evol.location, (evol.rel_energy_spread + evol_std.rel_energy_spread)*1e2, (evol.rel_energy_spread-evol_std.rel_energy_spread)*1e2, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.rel_energy_spread*1e2, color=col1)
        ax.set_ylabel('Energy spread, rms [%]')
        ax.set_xlim(long_limits)
        ax.set_yscale('log')
        
        # plot bunch length
        ax = fig.add_subplot(gs[1, 1])
        ax.fill_between(evol.location, (evol.bunch_length + evol_std.bunch_length)*1e6, (evol.bunch_length-evol_std.bunch_length)*1e6, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.bunch_length*1e6, color=col1)
        ax.set_xlim(long_limits)
        ax.set_ylabel(r'Bunch length [$\mathrm{\mu}$m]')
        
        # plot beam size
        ax = fig.add_subplot(gs[2, :])
        ax.fill_between(evol.location, (evol.beam_size_x + evol_std.beam_size_x)*1e6, (evol.beam_size_x-evol_std.beam_size_x)*1e6, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.beam_size_x*1e6, color=col1, label=r'$\sigma_x$')
        ax.fill_between(evol.location, (evol.beam_size_y + evol_std.beam_size_y)*1e6, (evol.beam_size_y-evol_std.beam_size_y)*1e6, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.beam_size_y*1e6, color=col2, label=r'$\sigma_y$')
        ax.set_xlim(long_limits)
        ax.set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        ax.set_yscale('log')
        ax.legend(ncols=2)
        
        # plot normalized emittance
        ax = fig.add_subplot(gs[3, :])
        ax.plot(evol.location, abs(evol.emit_ny[0]) * np.ones(evol.location.shape) * 1e6, ':', color=col2)
        ax.plot(evol.location, abs(evol.emit_nx[0]) * np.ones(evol.location.shape) * 1e6, ':', color=col1)
        ax.fill_between(evol.location, (evol.emit_ny + evol_std.emit_ny)*1e6, (evol.emit_ny-evol_std.emit_ny)*1e6, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.emit_ny*1e6, color=col2, label=r'$\epsilon_y$')
        ax.fill_between(evol.location, (evol.emit_nx + evol_std.emit_nx)*1e6, (evol.emit_nx-evol_std.emit_nx)*1e6, color=col1, alpha=af, ec=None)
        ax.plot(evol.location, evol.emit_nx*1e6, color=col1, label=r'$\epsilon_x$')
        ax.set_ylabel('Emittance, rms [mm mrad]')
        ax.set_xlim(long_limits)
        ax.set_yscale('log')
        ax.set_xlabel(long_label)
        ax.legend(ncols=2)
        
        plt.show()
        

class NotAssembledError(Exception):
    "Exception class for ``Beamline`` to throw if ``self.trackables`` have not been assembled."
    pass
        