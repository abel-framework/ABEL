# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.target.sfqed import TargetSFQED
import os, uuid, shutil
from abel.CONFIG import CONFIG
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SI

class TargetSFQEDPtarmigan(TargetSFQED):
    
    def __init__(self, laser_a0=None, laser_waist_size=None, laser_duration=None, laser_wavelength=800e-9, laser_polarization='circular', collision_angle_deg=0.0, increase_pair_rate_by=1e4):
        
        super().__init__(laser_a0=laser_a0, laser_waist_size=laser_waist_size, laser_duration=laser_duration, laser_wavelength=laser_wavelength, laser_polarization=laser_polarization, collision_angle_deg=collision_angle_deg)

        # simulation flags
        self.increase_pair_rate_by = increase_pair_rate_by

        # output
        self.output = None
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        from abel.wrappers.ptarmigan.ptarmigan_wrapper import ptarmigan_write_inputs, ptarmigan_run, ptarmigan_extract_outputs

        ## PREPARE TEMPORARY FOLDER
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.makedirs(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + os.sep
        
        # make directory
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
            
        # define input file
        filename_input = 'ptarmigan.yml'
        path_input = os.path.join(tmpfolder, filename_input)
        
        # make input file
        ptarmigan_write_inputs(path_input, beam, self.laser_a0, self.laser_wavelength, self.laser_duration, self.laser_waist_size, self.laser_polarization, self.collision_angle_deg, self.increase_pair_rate_by)

        # perform ptarmigan simulation
        ptarmigan_run(path_input)

        # extract the information from the H5 file
        filename_output = 'ptarmigan_particles.h5'
        path_output = os.path.join(tmpfolder, filename_output)
        self.output = ptarmigan_extract_outputs(path_output)

        # delete the simulation folder
        shutil.rmtree(tmpfolder)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def peak_chi(self):
        if self.output is not None:
            return max(self.output.photon.parent_chi)
        else:
            return None

    def mean_chi(self):
        if self.output is not None:
            return sum(self.output.photon.parent_chi*self.output.photon.weights)/sum(self.output.photon.weights)
        else:
            return None

    def charge_noninteracting(self):
        if self.output is not None:
            beam_mask = self.output.electron.ids < self.output.beam.num_particles
            n_gamma_mask = self.output.electron.n_gamma == 0
            both_mask = np.logical_and(beam_mask, n_gamma_mask)
            Q_non = sum(self.output.electron.weights[both_mask]) * SI.e
            return Q_non
        else:
            return None

    def interaction_fraction(self):
        if self.output is not None:
            beam_mask = self.output.electron.ids < self.output.beam.num_particles
            Q_tot = sum(self.output.electron.weights[beam_mask]) * SI.e
            Q_non = self.charge_noninteracting()
            return (Q_tot-Q_non)/Q_tot
        else:
            return None

    
    def plot_beam_spectrum(self, show_num_photons=True, log_plot=True):
        if self.output is not None:

            # mask to select only the beam electrons
            beam_mask = self.output.electron.ids < self.output.beam.num_particles

            # number of bins in the histogram
            num_bins = round(np.sqrt(self.output.beam.num_particles))
            Ebins = np.linspace(0, max(self.output.electron.pz), num_bins)
            
            # set up figure
            fig, ax = plt.subplots(1, 1)
            fig.set_figwidth(CONFIG.plot_width_default*0.8)
            fig.set_figheight(CONFIG.plot_width_default*0.5)
            
            if show_num_photons:

                import matplotlib.colors as mcolors
                tab_colors = list(mcolors.TABLEAU_COLORS)

                hf = []
                hw = []
                labels = []
                cols = []
                n = 0
                nmax = round(max(self.output.electron.n_gamma))
                for i in reversed(range(nmax)):
                    n_gamma_mask = self.output.electron.n_gamma == i
                    both_mask = np.logical_and(beam_mask, n_gamma_mask)
                    if sum(both_mask) == 0:
                        continue
                    hf = hf + [self.output.electron.pz[both_mask]/1e9]
                    hw = hw + [self.output.electron.weights[both_mask]]
                    if i == 1:
                        labels = labels + [f"{i} photon"]
                    else:
                        labels = labels + [f"{i} photons"]
                    cols = cols + [tab_colors[nmax-n-1]]
                    n = n + 1
                
                ax.hist(hf, weights=hw, bins=Ebins/1e9, stacked=True, histtype='bar', fill=True, label=labels, color=cols)
                ax.legend(reverse=True)
                
            else:

                Es = self.output.electron.pz[beam_mask]
                weights = self.output.electron.weights[beam_mask]
                ax.hist(Es/1e9, weights=weights, bins=Ebins/1e9)

            ax.set_xlabel('Energy (GeV)')
            ax.set_ylabel('Spectral density (a.u.)')
            ax.set_title('Beam electron spectrum')
            
            # make log scale
            if log_plot:
                ax.set_yscale('log')
                
        else:
            raise Exception('No output data (simulation not run)')

    
    def plot_pair_spectrum(self):
        if self.output is not None:
            
            num_bins = round(np.sqrt(self.output.beam.num_particles))
            bins = np.linspace(0, max(self.output.electron.pz)/1e9, num_bins)

            pair_mask = self.output.electron.ids > self.output.beam.num_particles
            weights_e = self.output.electron.weights[pair_mask]
            weights_p = self.output.positron.weights
            
            fig, ax = plt.subplots(1, 1)
            fig.set_figwidth(CONFIG.plot_width_default*0.8)
            fig.set_figheight(CONFIG.plot_width_default*0.5)

            charge_e = sum(weights_e)*SI.e
            charge_p = sum(weights_p)*SI.e
            
            ax.hist(self.output.electron.pz[pair_mask]/1e9, weights=weights_e, bins=bins, color='tab:blue', label=f'Pair electrons ({charge_e/1e-9:.1e} pC)')
            ax.hist(self.output.positron.pz/1e9, weights=weights_p, bins=bins, color='tab:orange', label=f'Pair positrons ({charge_p/1e-9:.1e} pC)')
            ax.set_ylabel('Spectral density (a.u.)')
            ax.set_xlabel('Energy (GeV)')
            ax.legend()
            ax.set_title('Electron—positron pair spectrum')
            
        else:
            raise Exception('No output data (simulation not run)')

    
    def plot_photon_chis(self):
        if self.output is not None:
            
            num_bins = round(np.sqrt(len(self.output.photon.parent_chi)))

            fig, ax = plt.subplots(1, 1)
            fig.set_figwidth(CONFIG.plot_width_default*0.8)
            fig.set_figheight(CONFIG.plot_width_default*0.5)
            
            ax.hist(self.output.photon.parent_chi, weights=self.output.photon.weights, bins=num_bins, color='tab:green')
            ax.set_xlabel('χ at production')
            ax.set_ylabel('Frequency (per bin)')
            ax.set_title('Photon production')
            
        else:
            raise Exception('No output data (simulation not run)')
    