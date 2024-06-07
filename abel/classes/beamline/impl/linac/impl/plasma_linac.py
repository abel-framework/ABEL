from abel import Beam, Linac, Source, DriverComplex, Stage, Interstage, BeamDeliverySystem, CONFIG
import scipy.constants as SI
import copy, os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import ticker as mticker

class PlasmaLinac(Linac):
    
    def __init__(self, source=None, driver_complex=None, stage=None, interstage=None, bds=None, num_stages=None, nom_energy=None, first_stage=None, last_stage=None, last_interstage=None, alternate_interstage_polarity=False, bunch_separation=None, num_bunches_in_train=None, rep_rate_trains=None):
        
        self.source = source
        self.driver_complex = driver_complex
        self.stage = stage
        self.interstage = interstage
        self.bds = bds
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.last_interstage = last_interstage
        self.num_stages = num_stages
        self.alternate_interstage_polarity = alternate_interstage_polarity
        
        super().__init__(nom_energy, num_bunches_in_train, bunch_separation, rep_rate_trains)
        
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # set default number of stages
        if self.num_stages is None:
            if self.stage is not None:
                self.num_stages = 1
            else:
                self.num_stages = 0

        # figure out the nominal energy gain if not set
        if self.nom_energy is None:
            self.nom_energy = self.source.energy + self.num_stages * self.stage.nom_energy_gain
        else:
            self.stage.nom_energy_gain = (self.nom_energy - self.source.energy) / self.num_stages
        
        # get or set the driver complex
        if self.driver_complex is not None:

            # check type
            assert(isinstance(self.driver_complex, DriverComplex))

            # set as stage driver
            if self.stage is not None:
                self.stage.driver_source = self.driver_complex
                
        else:
            if self.stage is not None and isinstance(self.stage.driver_source, DriverComplex):
                self.driver_complex = self.stage.driver_source
        
        # set the number of drivers required
        if self.driver_complex is not None:
            self.driver_complex.num_drivers = self.num_stages
            
            # set the rep rate of the driver complex (different to the main beam)
            if self.num_bunches_in_train is not None and self.rep_rate_trains is not None:
                
                # driver complex rep rate (one driver per stage)
                self.driver_complex.num_bunches_in_train = self.num_bunches_in_train*self.num_stages
                self.driver_complex.bunch_separation = self.bunch_separation/self.num_stages
                self.driver_complex.rep_rate_trains = self.rep_rate_trains
        
        # declare list of trackables, stages and interstages
        self.trackables = []
        self.stages = []
        self.interstages = []
        
        # add source
        assert(isinstance(self.source, Source))
        self.trackables.append(self.source)
        
        # add stages and interstages
        if self.stage is not None:

            # check types
            assert(isinstance(self.stage, Stage))
            if self.first_stage is not None:
                assert(isinstance(self.first_stage, Stage))
            if self.last_stage is not None:
                assert(isinstance(self.last_stage, Stage))
            if self.interstage is not None:
                assert(isinstance(self.interstage, Interstage))
            if self.last_interstage is not None:
                assert(isinstance(self.last_interstage, Interstage))
                    
            # instantiate many stages
            for i in range(self.num_stages):
                
                # add stages
                if i == 0 and self.first_stage is not None:
                    stage_instance = self.first_stage
                elif i == (self.num_stages-1) and self.last_stage is not None:
                    stage_instance = self.last_stage
                elif i == 0:
                    stage_instance = self.stage
                else:
                    stage_instance = copy.deepcopy(self.stage)
                    
                # reassign the same driver complex
                if self.driver_complex is not None:
                    stage_instance.driver_source = self.driver_complex
                    
                self.trackables.append(stage_instance)
                self.stages.append(stage_instance)

                # add interstages
                if (self.interstage is not None) and (i < self.num_stages-1):
                    if i == self.num_stages-2 and self.last_interstage is not None:
                        interstage_instance = self.last_interstage
                    else:
                        interstage_instance = copy.deepcopy(self.interstage)
                    if interstage_instance.nom_energy is None:
                        interstage_instance.nom_energy = self.source.get_energy() + np.sum([stg.get_nom_energy_gain() for stg in self.stages[:(i+1)]])
                    if self.alternate_interstage_polarity:
                        interstage_instance.dipole_field = (2*(i%2)-1)*interstage_instance.dipole_field
                    self.trackables.append(interstage_instance)
                    self.interstages.append(interstage_instance)
            
            # populate first/last stage properties
            if self.first_stage is None:
                self.first_stage = self.stages[0]
            if self.last_stage is None:
                self.last_stage = self.stages[-1]
                    
        # add beam delivery system
        if self.bds is not None:
            
            # check type
            assert(isinstance(self.bds, BeamDeliverySystem) or isinstance(self.bds, Interstage))

            # set the nominal energy and length
            self.bds.length = None
            self.bds.nom_energy = self.source.get_energy() + np.sum([stg.get_nom_energy_gain() for stg in self.stages])
            self.bds.length = self.bds.get_length()

            # add to trackables
            self.trackables.append(self.bds)
        
        # set the bunch train pattern etc.
        super().assemble_trackables()

    
    # survey object
        
    def survey_object(self):
        "Survey objects for the plasma linac (adds the driver complex)"
        objs = super().survey_object()
        if self.driver_complex is not None:
            return objs, (self.driver_complex.survey_object(), 1)
        else:
            return objs
    
    ## ENERGY CONSIDERATIONS
    
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
    
    def energy_usage(self):
        
        if self.trackables is None:
            self.assemble_trackables()
        Etot = self.source.energy_usage()
        if self.driver_complex is not None:
            Etot += self.driver_complex.energy_usage()
        else:
            for stage in self.stages:
                Etot += stage.energy_usage()
        return Etot
    
    def get_energy_efficiency(self):
        Etot_beam = self.final_beam.total_energy()
        return Etot_beam/self.energy_usage()

    # Enable setting the interstage nominal energy individually
    def set_interstage_nom_energy(self, interstage_num, nom_energy):
        if self.trackables is None:                                                    
            self.assemble_trackables()                                                 
        self.interstages[interstage_num].set_nom_energy(nom_energy)

    # Enable setting the stage nominal energy gain (and succeeding interstage nominal energy) individually
    def set_stage_nom_energy_gain(self, stage_num, nom_energy_gain):                          
        if self.trackables is None:                                                    
            self.assemble_trackables()                                                 
        self.stages[stage_num].set_nom_energy_gain(nom_energy_gain)
        nom_energies = self.nom_stage_energies()
        if 2*stage_num+1 < len(nom_energies)-1:
            nom_energy = nom_energies[2*stage_num+1]
            self.set_interstage_nom_energy(interstage_num=stage_num, nom_energy=nom_energy)

    # Enable setting the interstage plasma lens offset individually
    def set_interstage_lens_offset(self, interstage_num, lens_x_offset=0.0, lens_y_offset=0.0):
        if self.trackables is None:                                                    
            self.assemble_trackables()                                                 
        self.interstages[interstage_num].set_lens_offset(lens_x_offset, lens_y_offset)


    # cost
    
    def get_cost_breakdown(self):
        "Cost breakdown for the plasma linac [ILC units]"
        
        breakdown = []
        
        if self.driver_complex is not None:
            breakdown.append(self.driver_complex.get_cost_breakdown())
            
        breakdown.append(self.source.get_cost_breakdown())

        stage_costs = 0
        for stage in self.stages:
            stage_costs += stage.get_cost()
        breakdown.append((f"Plasma stages ({self.num_stages}x)", stage_costs))

        interstage_costs = 0
        for interstage in self.interstages:
            interstage_costs += interstage.get_cost()
        breakdown.append(('Interstages', interstage_costs))

        if self.bds is not None:
            breakdown.append(self.bds.get_cost_breakdown())

        breakdown.append(self.get_cost_breakdown_civil_construction())

        return ('Plasma linac', breakdown)

    
    ## PLOT EVOLUTION
    
    # apply function to all beam files
    def evolution_fcn(self, fcns, shot=None):
        
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
            for shot in range(self.num_shots):
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
             
        
    def plot_evolution(self, use_stage_nums=False, shot=None, save_fig=False):
        
        if self.trackables is None:
            self.assemble_trackables()
            
        # TODO: filter shots by step
        
        # calculate values
        ss, vals_mean, vals_std, stage_nums = self.evolution_fcn([Beam.abs_charge, \
                                             Beam.energy, Beam.rel_energy_spread, \
                                             Beam.bunch_length, Beam.z_offset, \
                                             Beam.norm_emittance_x, Beam.norm_emittance_y, \
                                             Beam.beam_size_x, Beam.beam_size_y, \
                                             Beam.x_offset, Beam.y_offset, Beam.angular_momentum], shot)
        
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
        sigxs = vals_mean[:,7]
        sigys = vals_mean[:,8]
        x0s = vals_mean[:,9]
        y0s = vals_mean[:,10]
        Lzs = vals_mean[:,11]
        
        # errors
        Qs_error = vals_std[:,0]
        Es_error = vals_std[:,1]
        sigdeltas_error = vals_std[:,2]
        sigzs_error = vals_std[:,3]
        z0s_error = vals_std[:,4]
        emnxs_error = vals_std[:,5]
        emnys_error = vals_std[:,6]
        sigxs_error = vals_std[:,7]
        sigys_error = vals_std[:,8]
        x0s_error = vals_std[:,9]
        y0s_error = vals_std[:,10]
        Lzs_error = vals_std[:,11]
        
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
        axs[0,0].set_ylabel('Energy [GeV]')
        
        axs[1,0].plot(long_axis, sigdeltas * 100, color=col1)
        axs[1,0].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigdeltas+sigdeltas_error, np.flip(sigdeltas-sigdeltas_error))) * 100, color=col1, alpha=af)
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_ylabel('Energy spread [%]')
        axs[1,0].set_yscale('log')
        
        axs[2,0].plot(long_axis, np.zeros(deltas.shape), ':', color=col0)
        axs[2,0].plot(long_axis, deltas * 100, color=col1)
        axs[2,0].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((deltas+deltas_error, np.flip(deltas-deltas_error))) * 100, color=col1, alpha=af)
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_ylabel('Energy offset [%]')

        axs[0,1].plot(long_axis, Q0 * np.ones(Qs.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(long_axis, Qs * 1e9, color=col1)
        axs[0,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((Qs+Qs_error, np.flip(Qs-Qs_error))) * 1e9, color=col1, alpha=af)
        axs[0,1].set_xlabel(long_label)
        axs[0,1].set_ylabel('Charge [nC]')
        #axs[0,1].set_ylim(0, Q0 * 1.3 * 1e9)
        
        axs[1,1].plot(long_axis, sigzs*1e6, color=col1)
        axs[1,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigzs+sigzs_error, np.flip(sigzs-sigzs_error))) * 1e6, color=col1, alpha=af)
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_ylabel(r'Bunch length [$\mathrm{\mu}$m]')
        
        axs[2,1].plot(long_axis, z0s*1e6, color=col1)
        axs[2,1].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((z0s+z0s_error, np.flip(z0s-z0s_error))) * 1e6, color=col1, alpha=af)
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_ylabel(r'Longitudinal offset [$\mathrm{\mu}$m]')
        
        axs[0,2].plot(long_axis, np.ones(len(long_axis))*emnxs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[0,2].plot(long_axis, np.ones(len(long_axis))*emnys[0]*1e6, ':', color=col0)
        axs[0,2].plot(long_axis, emnxs*1e6, color=col1, label=r'$\varepsilon_{\mathrm{n}x}$')
        axs[0,2].plot(long_axis, emnys*1e6, color=col2, label=r'$\varepsilon_{\mathrm{n}y}$')
        axs[0,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((emnxs+emnxs_error, np.flip(emnxs-emnxs_error))) * 1e6, color=col1, alpha=af)
        axs[0,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((emnys+emnys_error, np.flip(emnys-emnys_error))) * 1e6, color=col2, alpha=af)
        #if Lzs.max() > (min(emnxs.min(), emnys.min()))*1e-2:
        #    axs[0,2].plot(long_axis, Lzs*1e6, color=col0)
        #    axs[0,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((Lzs+Lzs_error, np.flip(Lzs-Lzs_error))) * 1e6, color=col0, alpha=af)
        axs[0,2].set_xlabel(long_label)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_yscale('log')
        axs[0,2].set_ylim([0.7e6*min(emnxs.min(), emnys.min()), 1.4e6*max(emnxs.max(), emnys.max())])
        axs[0,2].legend()
        
        axs[1,2].plot(long_axis, (Es_nom[0]/Es_nom)**(1/4)*sigxs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[1,2].plot(long_axis, (Es_nom[0]/Es_nom)**(1/4)*sigys[0]*1e6, ':', color=col0)
        axs[1,2].plot(long_axis, sigxs*1e6, color=col1, label=r'$\sigma_x$')
        axs[1,2].plot(long_axis, sigys*1e6, color=col2, label=r'$\sigma_y$')
        axs[1,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigxs+sigxs_error, np.flip(sigxs-sigxs_error))) * 1e6, color=col1, alpha=af)
        axs[1,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((sigys+sigys_error, np.flip(sigys-sigys_error))) * 1e6, color=col2, alpha=af)
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[1,2].set_yscale('log')
        axs[1,2].legend()
        
        axs[2,2].plot(long_axis, np.zeros(x0s.shape), ':', color=col0)
        axs[2,2].plot(long_axis, x0s*1e6, color=col1, label=r'$\langle x\rangle$')
        axs[2,2].plot(long_axis, y0s*1e6, color=col2, label=r'$\langle y\rangle$')
        axs[2,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((x0s+x0s_error, np.flip(x0s-x0s_error))) * 1e6, color=col1, alpha=af)
        axs[2,2].fill(np.concatenate((long_axis, np.flip(long_axis))), np.concatenate((y0s+y0s_error, np.flip(y0s-y0s_error))) * 1e6, color=col2, alpha=af)
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,2].legend()
        
        plt.show()

        if save_fig:
            plot_path = self.run_path() + 'plots/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'evolution' + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)
    
    
    
    # density plots
    def plot_waterfalls(self, shot=None, save_fig=False):
        
        if self.trackables is None:
            self.assemble_trackables()
            
        # select shot
        if shot is None:
            if hasattr(self, 'shot') and self.shot is not None:
                shot = self.shot
            else:
                shot = 0
        
        # calculate values
        beam0 = self.get_beam(0,shot=shot)
        num_bins = int(np.sqrt(len(beam0)*2))
        nsig = 5
        tedges = (beam0.z_offset(clean=True) + nsig*beam0.bunch_length(clean=True)*np.linspace(-1, 1, num_bins)) / SI.c
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
        c0 = axs[0].pcolor(trackable_numbers, ts*SI.c*1e6, -Is/1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_ylabel(r'Longitudinal position [$\mathrm{\mu}$m]')
        cbar0.ax.set_ylabel('Beam current [kA]')
        axs[0].set_title('Shot ' + str(shot+1))
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(trackable_numbers, deltas*1e2, -dQddeltas*1e7, cmap=CONFIG.default_cmap, shading='auto')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_ylabel('Energy offset [%]')
        cbar1.ax.set_ylabel('Spectral density [nC/%]')
        
        densityX = waterfalls[2]
        xs = bins[2]
        c2 = axs[2].pcolor(trackable_numbers, xs*1e6, -densityX*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar2 = fig.colorbar(c2, ax=axs[2])
        axs[2].set_ylabel(r'Horizontal position [$\mathrm{\mu}$m]')
        cbar2.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')
        
        densityY = waterfalls[3]
        ys = bins[3]
        c3 = axs[3].pcolor(trackable_numbers, ys*1e6, -densityY*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar3 = fig.colorbar(c3, ax=axs[3])
        axs[3].set_xlabel('Trackable element number')
        axs[3].set_ylabel(r'Vertical position [$\mathrm{\mu}$m]')
        cbar3.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')
        
        plt.show()

        if save_fig:
            plot_path = self.run_path() + 'plots/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'waterfalls_shot' + str(self.shot) + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)

    
    # animate the longitudinal phase space
    def animate_lps(self, rel_energy_window=0.06):
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = self.get_beam(0)
        dQdzdE0, zs0, Es0 = beam_init.density_lps()
        Is0, ts_ = beam_init.current_profile(bins=zs0/SI.c)
        dQdE0, Es_ = beam_init.energy_spectrum(bins=Es0)

        # get final beam
        beam_final = self.get_beam(-1)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        dQdE_final, _ = beam_final.energy_spectrum(bins=Es0)

        # nominal energies
        Es_nom = self.nom_stage_energies()
        
        # prepare centroid arrays
        z0s = []
        deltas = []
        sigzs = []
        sigdeltas = []
        ss = []
        Emeans = []

        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i):

            # get beam for this frame
            beam = self.get_beam(i)
            
            # plot mean energy evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in linac [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(0,beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot energy spread and bunch length evolution
            sigzs.append(beam.bunch_length())
            sigdeltas.append(beam.rel_energy_spread())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(sigdeltas)*1e2, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, sigdeltas[-1]*1e2, 'o', color=col1)
            axs[0,1].set_ylim(min([min(sigdeltas)*0.8e2, 1e-1]), max([max(sigdeltas)*1.2e2, 10]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, sigzs[0]*0.7e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.3e6]))
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Energy spread [%]')
            axs[0,1].set_yscale('log')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            
            # plot LPS
            axs[1,0].cla()
            Ebins = Es_nom[i]*np.linspace(1-rel_energy_window, 1+rel_energy_window, 2*Es0.size)
            dQdzdE, zs, Es = beam.phase_space_density(beam.zs, beam.Es, hbins=zs0, vbins=Ebins)
            cax = axs[1,0].pcolor(zs*1e6, Es/1e9, -dQdzdE*1e15, cmap=CONFIG.default_cmap, shading='auto', clim=[0, abs(dQdzdE).max()*1e15])
            axs[1,0].set_ylabel('Energy [GeV]')
            axs[1,0].set_title('Longitudinal phase space')
            
            # plot current profile
            axs[2,0].cla()
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot energy spectrum
            dQdE, Es2 = beam.energy_spectrum(bins=Ebins)
            deltas2 = Es2/Es_nom[i]-1
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdE, np.zeros(dQdE.size)))*1e18, np.concatenate((deltas2, np.flip(deltas2)))*100, alpha=af, color=col1)
            axs[1,1].plot(-dQdE*1e18, deltas2*100)
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].set_ylabel('Rel. energy [%]')
            axs[1,1].set_ylim([-rel_energy_window*100, rel_energy_window*100])
            axs[1,1].set_xlim([0, max(-dQdE)*1.1e18])
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel('dQ/dE [nC/GeV]')
            
            # plot E and z centroid evolution
            z0s.append(beam.z_offset())
            deltas.append(beam.energy()/Es_nom[i]-1)
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(deltas)*100, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, deltas[-1]*100, 'o', color=col1)
            axs[2,1].set_ylim([-rel_energy_window*0.5e2, rel_energy_window*0.5e2])
            axs[2,1].set_xlim([(z0s[0]-sigzs[0]/2)*1e6, (z0s[0]+sigzs[0]/2)*1e6])
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel('Energy offset [%]')
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')


            return cax

        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(self.num_outputs()), repeat=False, interval=100)

        # save the animation as a GIF
        plot_path = self.run_path() + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'lps_shot' + str(self.shot) + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename


    # animate the horizontal phase space
    def animate_phasespace_x(self):
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = self.get_beam(0)
        dQdxdpx0, xs0, pxs0 = beam_init.phase_space_density(beam_init.xs, beam_init.pxs)
        dQdx0, xs_ = beam_init.projected_density(beam_init.xs, bins=xs0)
        dQdpx0, pxs_ = beam_init.projected_density(beam_init.pxs, bins=pxs0)

        # get final beam
        beam_final = self.get_beam(-1)
        dQdxdpx_final, xs_final, pxs_final = beam_final.phase_space_density(beam_final.xs, beam_final.pxs)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        dQdpx_final, _ = beam_final.projected_density(beam_final.pxs, bins=pxs0)

        # calculate limits
        pxlim = max(max(-pxs0.min(), pxs0.max()), max(-pxs_final.min(), pxs_final.max()))
        
        # prepare centroid arrays
        x0s = []
        xp0s = []
        sigxs = []
        sigxps = []
        ss = []
        emitns = []

        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i):

            # get beam for this frame
            beam = self.get_beam(i)
            
            # plot emittance evolution
            ss.append(beam.location)
            emitns.append(beam.norm_emittance_x())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(emitns)*1e6, '-', color=col0)
            axs[0,0].plot(ss[-1], emitns[-1]*1e6, 'o', color=col1)
            axs[0,0].set_xlabel('Position in linac [m]')
            axs[0,0].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(beam_init.norm_emittance_x()*0.5e6,beam_final.norm_emittance_x()*2e6)
            axs[0,0].set_yscale('log')
            axs[0,0].yaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot beam size and divergence evolution
            sigxs.append(beam.beam_size_x())
            sigxps.append(beam.divergence_x())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigxs)*1e6, np.array(sigxps)*1e3, '-', color=col0)
            axs[0,1].plot(sigxs[-1]*1e6, sigxps[-1]*1e3, 'o', color=col1)
            axs[0,1].set_ylim(min([min(sigxps)*0.9e3, beam_final.divergence_x()*0.8e3]), max([max(sigxps)*1.1e3, sigxs[0]*1.2e3]))
            axs[0,1].set_xlim(min([min(sigxs)*0.9e6, beam_final.beam_size_x()*0.8e6]), max([max(sigxs)*1.1e6, sigxs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Beam size [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Divergence [mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdxdpx, xs, pxs = beam.phase_space_density(beam.xs, beam.pxs, hbins=xs0, vbins=pxs_final)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(xs*1e6, pxs*1e-6*SI.c/SI.e, -dQdxdpx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel("Momentum, $p_x$ [MeV/c]")
            axs[1,0].set_title('Horizontal phase space')
            
            # plot position projection
            af = 0.15
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((xs2, np.flip(xs2)))*1e6, -np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, alpha=af, color=col1)
            axs[2,0].plot(xs2*1e6, -dQdx*1e3, color=col1)
            axs[2,0].set_xlim([min(xs0)*1e6, max(xs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-dQdx0), max(-dQdx_final)])*1.2e3])
            axs[2,0].set_xlabel(r'Transverse position, $x$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$dQ/dx$ [nC/$\mathrm{\mu}$m]')
            
            # plot angular projection
            dQdpx, pxs2 = beam.projected_density(beam.pxs, bins=pxs_final)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdpx, np.zeros(dQdpx.size)))*1e9/(1e-6*SI.c/SI.e), np.concatenate((pxs2, np.flip(pxs2)))*1e-6*SI.c/SI.e, alpha=af, color=col1)
            axs[1,1].plot(-dQdpx*1e9/(1e-6*SI.c/SI.e), pxs2*1e-6*SI.c/SI.e, color=col1)
            axs[1,1].set_ylim([-pxlim*1e-6*SI.c/SI.e, pxlim*1e-6*SI.c/SI.e])
            axs[1,1].set_xlim([0, max([max(-dQdpx0), max(-dQdpx_final)])*1e9/(1e-6*SI.c/SI.e)])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel("$dQ/dp_x$ [nC c/MeV]")
            axs[1,1].set_ylabel("Momentum, $p_x$ [MeV/c]")
            
            # plot centroid evolution
            x0s.append(beam.x_offset())
            xp0s.append(beam.x_angle())
            axs[2,1].cla()
            axs[2,1].plot(np.array(x0s)*1e6, np.array(xp0s)*1e6, '-', color=col0)
            axs[2,1].plot(x0s[-1]*1e6, xp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'Centroid offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'Centroid angle [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(min(-max(x0s)*1.1,-0.1*sigxs[0])*1e6, max(max(x0s)*1.1,0.1*sigxs[0])*1e6)
            axs[2,1].set_ylim(min(-max(xp0s)*1.1,-0.1*sigxps[0])*1e6, max(max(xp0s)*1.1,0.1*sigxps[0])*1e6)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')

            return cax

        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(self.num_outputs()), repeat=False, interval=100)

        # save the animation as a GIF
        plot_path = self.run_path() + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'phasespace_x_shot' + str(self.shot) + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename



    # animate the vertical phase space
    def animate_phasespace_y(self):
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = self.get_beam(0)
        dQdydpy0, ys0, pys0 = beam_init.phase_space_density(beam_init.ys, beam_init.pys)
        dQdy0, ys_ = beam_init.projected_density(beam_init.ys, bins=ys0)
        dQdpy0, pys_ = beam_init.projected_density(beam_init.pys, bins=pys0)

        # get final beam
        beam_final = self.get_beam(-1)
        dQdydpy_final, ys_final, pys_final = beam_final.phase_space_density(beam_final.ys, beam_final.pys)
        dQdy_final, _ = beam_final.projected_density(beam_final.ys, bins=ys0)
        dQdpy_final, _ = beam_final.projected_density(beam_final.pys, bins=pys0)

        # calculate limits
        pylim = max(max(-pys0.min(), pys0.max()), max(-pys_final.min(), pys_final.max()))
        
        # prepare centroid arrays
        y0s = []
        yp0s = []
        sigys = []
        sigyps = []
        ss = []
        emitns = []

        # set the colors and transparency
        col0 = "#f5d9c1"
        col1 = "tab:orange"
        
        # frame function
        def frameFcn(i):

            # get beam for this frame
            beam = self.get_beam(i)
            
            # plot emittance evolution
            ss.append(beam.location)
            emitns.append(beam.norm_emittance_y())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(emitns)*1e6, '-', color=col0)
            axs[0,0].plot(ss[-1], emitns[-1]*1e6, 'o', color=col1)
            axs[0,0].set_xlabel('Position in linac [m]')
            axs[0,0].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(beam_init.norm_emittance_y()*0.5e6,beam_final.norm_emittance_y()*2e6)
            axs[0,0].set_yscale('log')
            axs[0,0].yaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot beam size and divergence evolution
            sigys.append(beam.beam_size_y())
            sigyps.append(beam.divergence_y())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigys)*1e6, np.array(sigyps)*1e3, '-', color=col0)
            axs[0,1].plot(sigys[-1]*1e6, sigyps[-1]*1e3, 'o', color=col1)
            axs[0,1].set_ylim(min([min(sigyps)*0.9e3, beam_final.divergence_y()*0.8e3]), max([max(sigyps)*1.1e3, sigys[0]*1.2e3]))
            axs[0,1].set_xlim(min([min(sigys)*0.9e6, beam_final.beam_size_y()*0.8e6]), max([max(sigys)*1.1e6, sigys[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Beam size [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Divergence [mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdydpy, ys, pys = beam.phase_space_density(beam.ys, beam.pys, hbins=ys0, vbins=pys_final)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(ys*1e6, pys*1e-6*SI.c/SI.e, -dQdydpy, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel("Momentum, $p_y$ [MeV/c]")
            axs[1,0].set_title('Vertical phase space')
            
            # plot position projection
            af = 0.15
            dQdy, ys2 = beam.projected_density(beam.ys, bins=ys0)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ys2, np.flip(ys2)))*1e6, -np.concatenate((dQdy, np.zeros(dQdy.size)))*1e3, alpha=af, color=col1)
            axs[2,0].plot(ys2*1e6, -dQdy*1e3, color=col1)
            axs[2,0].set_xlim([min(ys0)*1e6, max(ys0)*1e6])
            axs[2,0].set_ylim([0, max([max(-dQdy0), max(-dQdy_final)])*1.2e3])
            axs[2,0].set_xlabel(r'Transverse position, $y$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$dQ/dy$ [nC/$\mathrm{\mu}$m]')
            
            # plot angular projection
            dQdpy, pys2 = beam.projected_density(beam.pys, bins=pys_final)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdpy, np.zeros(dQdpy.size)))*1e9/(1e-6*SI.c/SI.e), np.concatenate((pys2, np.flip(pys2)))*1e-6*SI.c/SI.e, alpha=af, color=col1)
            axs[1,1].plot(-dQdpy*1e9/(1e-6*SI.c/SI.e), pys2*1e-6*SI.c/SI.e, color=col1)
            axs[1,1].set_ylim([-pylim*1e-6*SI.c/SI.e, pylim*1e-6*SI.c/SI.e])
            axs[1,1].set_xlim([0, max([max(-dQdpy0), max(-dQdpy_final)])*1e9/(1e-6*SI.c/SI.e)])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel("$dQ/dp_y$ [nC c/MeV]")
            axs[1,1].set_ylabel("Momentum, $p_y$ [MeV/c]")
            
            # plot centroid evolution
            y0s.append(beam.y_offset())
            yp0s.append(beam.y_angle())
            axs[2,1].cla()
            axs[2,1].plot(np.array(y0s)*1e6, np.array(yp0s)*1e6, '-', color=col0)
            axs[2,1].plot(y0s[-1]*1e6, yp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'Centroid offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'Centroid angle [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(min(-max(y0s)*1.1,-0.1*sigys[0])*1e6, max(max(y0s)*1.1,0.1*sigys[0])*1e6)
            axs[2,1].set_ylim(min(-max(yp0s)*1.1,-0.1*sigyps[0])*1e6, max(max(yp0s)*1.1,0.1*sigyps[0])*1e6)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')

            return cax

        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(self.num_outputs()), repeat=False, interval=100)

        # save the animation as a GIF
        plot_path = self.run_path() + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'phasespace_y_shot' + str(self.shot) + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename

    
    # animate the horizontal sideview (top view)
    def animate_sideview_x(self):
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = self.get_beam(0)
        dQdzdx0, zs0, xs0 = beam_init.phase_space_density(beam_init.zs, beam_init.xs)
        dQdx0, xs_ = beam_init.projected_density(beam_init.xs, bins=xs0)
        Is0, _ = beam_init.current_profile(bins=zs0/SI.c)

        # get final beam
        beam_final = self.get_beam(-1)
        dQdzdx_final, zs_final, xs_final = beam_final.phase_space_density(beam_final.zs, beam_final.xs)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        # prepare centroid arrays
        x0s = []
        z0s = []
        Emeans = []
        sigzs = []
        sigxs = []
        ss = []
        emitns = []

        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i):

            # get beam for this frame
            beam = self.get_beam(i)
            
            # plot emittance evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in linac [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(0,beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_x()) # TODO: update to normalized amplitude
            sigzs.append(beam.bunch_length())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(min([min(emitns)*0.9e6, beam_final.norm_emittance_x()*0.8e6]), max([max(emitns)*1.1e6, emitns[0]*1.2e6]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, beam_final.bunch_length()*0.8e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel(r'Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdx, zs, xs = beam.phase_space_density(beam.zs, beam.xs, hbins=zs0, vbins=xs0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, xs*1e6, -dQdzdx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse offset, $x$ [$\mathrm{\mu}$m]")
            axs[1,0].set_title('Horizontal sideview (top view)')

            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot position projection
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, np.concatenate((xs2, np.flip(xs2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdx*1e3, xs2*1e6, color=col1)
            axs[1,1].set_ylim([min(xs0)*1e6, max(xs0)*1e6])
            axs[1,1].set_xlim([0, max([max(-dQdx), max(-dQdx_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dx$ [nC/$\mathrm{\mu}$m]")
            axs[1,1].set_ylabel(r"$x$ [$\mathrm{\mu}$m]")
            
            # plot centroid evolution
            z0s.append(beam.z_offset())
            sigxs.append(beam.beam_size_x())
            x0s.append(beam.x_offset())
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(x0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, x0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'$x$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_xlim(min([min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, max([max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            axs[2,1].set_ylim(min(-max(x0s)*1.1,-0.1*sigxs[0])*1e6, max(max(x0s)*1.1,0.1*sigxs[0])*1e6)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')

            return cax

        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(self.num_outputs()), repeat=False, interval=100)

        # save the animation as a GIF
        plot_path = self.run_path() + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'sideview_x_shot' + str(self.shot) + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename


    
    # animate the vertical sideview
    def animate_sideview_y(self):
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = self.get_beam(0)
        dQdzdy0, zs0, ys0 = beam_init.phase_space_density(beam_init.zs, beam_init.ys)
        dQdy0, ys_ = beam_init.projected_density(beam_init.ys, bins=ys0)
        Is0, _ = beam_init.current_profile(bins=zs0/SI.c)

        # get final beam
        beam_final = self.get_beam(-1)
        dQdzdy_final, zs_final, ys_final = beam_final.phase_space_density(beam_final.zs, beam_final.ys)
        dQdy_final, _ = beam_final.projected_density(beam_final.ys, bins=ys0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        # prepare centroid arrays
        y0s = []
        z0s = []
        Emeans = []
        sigzs = []
        sigys = []
        ss = []
        emitns = []

        # set the colors and transparency
        col0 = "#f5d9c1"
        col1 = "tab:orange"
        
        # frame function
        def frameFcn(i):

            # get beam for this frame
            beam = self.get_beam(i)
            
            # plot emittance evolution
            ss.append(beam.location)
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Position in linac [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(beam_init.location,beam_final.location)
            axs[0,0].set_ylim(0,beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_y()) # TODO: update to normalized amplitude
            sigzs.append(beam.bunch_length())
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(min([min(emitns)*0.9e6, beam_final.norm_emittance_y()*0.8e6]), max([max(emitns)*1.1e6, emitns[0]*1.2e6]))
            axs[0,1].set_xlim(min([min(sigzs)*0.9e6, beam_final.bunch_length()*0.8e6]), max([max(sigzs)*1.1e6, sigzs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel(r'Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdy, zs, ys = beam.phase_space_density(beam.zs, beam.ys, hbins=zs0, vbins=ys0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, ys*1e6, -dQdzdy, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse offset, $y$ [$\mathrm{\mu}$m]")
            axs[1,0].set_title('Vertical sideview')

            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3, color=col1)
            axs[2,0].set_xlim([min(zs0)*1e6, max(zs0)*1e6])
            axs[2,0].set_ylim([0, max([max(-Is0), max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel('$I$ [kA]')
            
            # plot position projection
            dQdy, ys2 = beam.projected_density(beam.ys, bins=ys0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdy, np.zeros(dQdy.size)))*1e3, np.concatenate((ys2, np.flip(ys2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdy*1e3, ys2*1e6, color=col1)
            axs[1,1].set_ylim([min(ys0)*1e6, max(ys0)*1e6])
            axs[1,1].set_xlim([0, max([max(-dQdy), max(-dQdy_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dy$ [nC/$\mathrm{\mu}$m]")
            axs[1,1].set_ylabel(r"$y$ [$\mathrm{\mu}$m]")
            
            # plot centroid evolution
            z0s.append(beam.z_offset())
            sigys.append(beam.beam_size_y())
            y0s.append(beam.y_offset())
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(y0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, y0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'y offset [$\mathrm{\mu}$m]')
            axs[2,1].set_xlim(min([min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, max([max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            axs[2,1].set_ylim(min(-max(y0s)*1.1,-0.1*sigys[0])*1e6, max(max(y0s)*1.1,0.1*sigys[0])*1e6)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')

            return cax

        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(self.num_outputs()), repeat=False, interval=100)

        # save the animation as a GIF
        plot_path = self.run_path() + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'sideview_y_shot' + str(self.shot) + '.gif'
        animation.save(filename, writer="pillow", fps=10)

        # hide the figure
        plt.close()

        return filename

        