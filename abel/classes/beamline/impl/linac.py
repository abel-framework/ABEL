from abel import Beam, Beamline, Source, Stage, Interstage, BeamDeliverySystem
import scipy.constants as SI
import copy, os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation

class Linac(Beamline):
    
    def __init__(self, source=None, stage=None, interstage=None, bds=None, num_stages=0, first_stage=None, alternate_interstage_polarity=False):
        
        self.source = source
        self.stage = stage
        self.interstage = interstage
        self.bds = bds
        self.first_stage = first_stage
        self.num_stages = num_stages
        self.alternate_interstage_polarity = alternate_interstage_polarity
        
        super().__init__()
        
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes
        assert(isinstance(self.source, Source))
        if self.stage is not None:
            assert(isinstance(self.stage, Stage))
        if self.interstage is not None:
            assert(isinstance(self.interstage, Interstage))
        if self.bds is not None:
            assert(isinstance(self.bds, BeamDeliverySystem))
        if self.first_stage is not None:
            assert(isinstance(self.first_stage, Stage))
        
        # prepare for multiplication of stages and interstages
        self.stages = [None]*self.num_stages
        self.interstages = [None]*max(0,self.num_stages-1)
        
        # declare list of trackables
        self.trackables = [None] * (1 + self.num_stages + max(0,self.num_stages-1) + int(self.bds is not None))
        
        # add source
        self.trackables[0] = self.source
        
        # add stages and interstages
        if (self.stage is not None) and (self.interstage is not None):
            for i in range(self.num_stages):

                # add stages
                if i == 0 and self.first_stage is not None:
                    stage_instance = self.first_stage
                else:
                    stage_instance = copy.deepcopy(self.stage)
                self.trackables[1+2*i] = stage_instance
                self.stages[i] = stage_instance

                # add interstages
                if i < self.num_stages-1:
                    interstage_instance = copy.deepcopy(self.interstage)
                    interstage_instance.nom_energy = self.source.get_energy() + np.sum([stg.get_nom_energy_gain() for stg in self.stages[:(i+1)]])
                    if self.alternate_interstage_polarity:
                        interstage_instance.dipole_field = (2*(i%2)-1)*interstage_instance.dipole_field
                    self.trackables[2+2*i] = interstage_instance
                    self.interstages[i] = interstage_instance
        
        # add beam delivery system
        if self.bds is not None:
            self.bds.nom_energy = self.source.get_energy()
            self.bds.nom_energy += np.sum([stg.get_nom_energy_gain() for stg in self.stages])
            self.trackables[max(1,2*self.num_stages)] = self.bds
        
    
    ## ENERGY CONSIDERATIONS
    
    def nom_energy(self):
        return max(self.nom_stage_energies())
    
    def nom_stage_energies(self):
        E = 0
        Es = np.array([]);
        for trackable in self.trackables:
            if isinstance(trackable, Source):
                E += trackable.get_energy()
            elif isinstance(trackable, Stage):
                E += trackable.get_nom_energy_gain()
            Es = np.append(Es, E)
        return Es
    
    
    def effective_gradient(self):
        return self.nom_energy()/self.get_length()
    
    
    def energy_usage(self):
        if self.trackables is None:
            self.assemble_trackables()
        Etot = self.source.energy_usage()
        for stage in self.stages:
            Etot += stage.energy_usage()
        return Etot
    
    def energy_efficiency(self):
        Etot_beam = self.final_beam().total_energy()
        return Etot_beam/self.energy_usage()
    
    
    ## PLOT EVOLUTION
    
    # apply function to all beam files
    def __evolution_fcn(self, fcns, shot=None):
        
        # declare data structure
        num_outputs = self.num_outputs()
        stage_numbers = np.empty(num_outputs)
        ss = np.empty(num_outputs)
        vals_mean = np.empty((num_outputs, len(fcns)))
        vals_std = np.empty((num_outputs, len(fcns)))

        if shot is None:
            shots = range(self.num_shots)
        else:
            shots = shot
        
        # go through files
        for index in range(num_outputs):
            
            # load beams and apply functions
            vals = np.empty((self.num_shots, len(fcns)))
            for shot in shots:
                beam = self.get_beam(index=index, shot=shot)
                for k in range(len(fcns)):
                    vals[shot,k] = fcns[k](beam)
            
            # calculate mean and standard dev
            for k in range(len(fcns)):
                vals_mean[index,k] = np.mean(vals[:,k])
                vals_std[index,k] = np.std(vals[:,k])
            
            # find stage number
            stage_numbers[index] = beam.stage_number
            ss[index] = beam.location
        
        return ss, vals_mean, vals_std, stage_numbers
 

    # apply waterfall function to all beam files
    def __waterfall_fcn(self, fcns, edges, args=None, shot=0):
        
        # find number of beam outputs to plot
        num_outputs = self.num_outputs(shot)
        
        # declare data structure
        bins = [None] * len(fcns)
        waterfalls = [None] * len(fcns)
        for j in range(len(fcns)):
            waterfalls[j] = np.empty((len(edges[j])-1, num_outputs))
        trackable_numbers = np.empty(num_outputs)
        
        # go through files
        for index in range(num_outputs):

            # load phase space
            beam = self.get_beam(index=index, shot=shot)

            # find trackable number
            trackable_numbers[index] = beam.trackable_number
            
            # get all waterfalls (apply argument is it exists)
            for j in range(len(fcns)):
                if args[j] is None:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, bins=edges[j])
                else:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, args[j][index], bins=edges[j])
                
        return waterfalls, trackable_numbers, bins
             
        
    def plot_evolution(self, use_stage_nums=False, shot=None):
        
        if self.trackables is None:
            self.assemble_trackables()
            
        # TODO: filter shots by step
        
        # calculate values
        ss, vals_mean, vals_std, stage_nums = self.__evolution_fcn([Beam.abs_charge, \
                                             Beam.energy, Beam.rel_energy_spread, \
                                             Beam.bunch_length, Beam.z_offset, \
                                             Beam.norm_emittance_x, Beam.norm_emittance_y, \
                                             Beam.beta_x, Beam.beta_y, \
                                             Beam.x_offset, Beam.y_offset], shot)
        
        if use_stage_nums:
            long_axis = stage_nums
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
        
        # nominal energies
        Es_nom = self.nom_stage_energies()
        deltas = Es/Es_nom - 1
        deltas_error = Es_error/Es_nom
        
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
        
        axs[0,0].plot(long_axis, Es_nom / 1e9, ':', color=col0)
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
        
        axs[1,2].plot(long_axis, np.sqrt(Es_nom/Es_nom[0])*betaxs[0]*1e3, ':', color=col0)
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
    def plot_waterfalls(self, shot=None):
        
        if self.trackables is None:
            self.assemble_trackables()
            
        # select shot
        if shot is None:
            if hasattr(self, 'shot') and self.shot is not None:
                shot = self.shot
            else:
                shot = 0
        
        # calculate values
        beam0 = self.initial_beam(shot=shot)
        num_bins = int(np.sqrt(len(beam0)*2))
        nsig = 5
        tedges = np.arange(-100e-6, 0e-6, 1e-6) / SI.c
        deltaedges = np.linspace(-0.05, 0.05, num_bins)
        xedges = (nsig*beam0.beam_size_x() + abs(beam0.x_offset()))*np.linspace(-1, 1, num_bins)
        yedges = (nsig*beam0.beam_size_y() + abs(beam0.y_offset()))*np.linspace(-1, 1, num_bins)
        E0s = self.nom_stage_energies()
        waterfalls, trackable_numbers, bins = self.__waterfall_fcn([Beam.current_profile, Beam.rel_energy_spectrum, Beam.transverse_profile_x, Beam.transverse_profile_y], [tedges, deltaedges, xedges, yedges], [None, E0s, None, None], shot)
        
        # prepare figure
        fig, axs = plt.subplots(4,1)
        fig.set_figwidth(8)
        fig.set_figheight(11)
        
        # current profile
        Is = waterfalls[0]
        ts = bins[0]
        c0 = axs[0].pcolor(trackable_numbers, ts*SI.c*1e6, -Is/1e3, cmap='GnBu', shading='auto')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_ylabel('Longitudinal position (um)')
        cbar0.ax.set_ylabel('Beam current (kA)')
        axs[0].set_title('Shot ' + str(shot+1))
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(trackable_numbers, deltas*1e2, -dQddeltas*1e7, cmap='GnBu', shading='auto')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_ylabel('Energy offset (%)')
        cbar1.ax.set_ylabel('Spectral density (nC/%)')
        
        densityX = waterfalls[2]
        xs = bins[2]
        c2 = axs[2].pcolor(trackable_numbers, xs*1e6, -densityX*1e3, cmap='GnBu', shading='auto')
        cbar2 = fig.colorbar(c2, ax=axs[2])
        axs[2].set_ylabel('Horizontal position (um)')
        cbar2.ax.set_ylabel('Charge density (nC/um)')
        
        densityY = waterfalls[3]
        ys = bins[3]
        c3 = axs[3].pcolor(trackable_numbers, ys*1e6, -densityY*1e3, cmap='GnBu', shading='auto')
        cbar3 = fig.colorbar(c3, ax=axs[3])
        axs[3].set_xlabel('Trackable element number')
        axs[3].set_ylabel('Vertical position (um)')
        cbar3.ax.set_ylabel('Charge density (nC/um)')
        
        plt.show()
        