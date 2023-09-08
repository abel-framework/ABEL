from abel import Runnable, Linac, BeamDeliverySystem, InteractionPoint, Event
from matplotlib import pyplot as plt
import numpy as np
import os, copy
from matplotlib import lines
from datetime import datetime

class Collider(Runnable):
    
    # constructor
    def __init__(self, linac1=None, linac2=None, ip=None):
        self.linac1 = linac1
        self.linac2 = linac2
        self.ip = ip
        
        self.cost_per_length = 2e5 # [LCU/m]
        self.cost_per_energy = 0.15/3.6e6 # [LCU/J]
        self.targetIntegratedLuminosity = 1e46 # [m^-2] or 1/ab
    
    
    # calculate energy usage (per bunch crossing)
    def energy_usage(self):
        return self.linac1.energy_usage() + self.linac2.energy_usage()
    
    # full luminosity per crossing [m^-2]
    def full_luminosity_per_crossing(self):
        files = self.ip.run_data()
        num_events = len(files)
        lumi_full = np.empty(num_events)
        for i in range(num_events):
            event = Event.load(files[i], load_beams=False)
            lumi_full[i] = event.full_luminosity()
        return np.median(lumi_full)
    
    # peak luminosity per crossing [m^-2]
    def peak_luminosity_per_crossing(self):
        files = self.ip.run_data()
        num_events = len(files)
        lumi_peak = np.empty(num_events)
        for i in range(num_events):
            event = Event.load(files[i], load_beams=False)
            lumi_peak[i] = event.peak_luminosity()
        return np.median(lumi_peak)
        
    # full luminosity per power [m^-2/J]
    def full_luminosity_per_power(self):
        return self.full_luminosity_per_crossing() / self.energy_usage()
    
    # peak luminosity per power [m^-2/J]
    def peak_luminosity_per_power(self):
        return self.peak_luminosity_per_crossing() / self.energy_usage()
    
    # integrated energy usage (to reach target integrated luminosity)
    def integrated_energy_usage(self):
        return self.target_integrated_luminosity / self.peak_luminosity_per_power()
    
    # integrated cost of energy
    def running_cost(self):
        return self.integrated_energy_usage() * self.cost_per_energy
      
    # total length of linacs (TODO: add driver production length)
    def total_length(self):
        Ltot = self.linac1.get_length() + self.linac2.get_length()
        if hasattr(self.linac1.stage, 'driver_source'):
            Ltot += self.linac1.stage.driver_source.get_length()
        return Ltot
      
    # cost of construction
    def construction_cost(self):
        return self.total_length() * self.cost_per_length
    
    # total cost of construction and running
    def total_cost(self):
        return self.construction_cost() + self.running_cost()
    
    
    # overwrite run function
    def run(self, run_name=None, num_shots=1, savedepth=2, verbose=True, overwrite=False, overwrite_ip=False, parallel=False, max_cores=16):
        
        # check element classes, then assign
        if hasattr(self, 'linac') and self.linac1 is None:
            self.linac1 = self.linac
        
        # copy second arm if undefined
        if self.linac2 is None:
            self.linac2 = copy.deepcopy(self.linac1)
            
        assert(isinstance(self.linac1, Linac))
        assert(isinstance(self.linac2, Linac))
        assert(isinstance(self.ip, InteractionPoint))
        
        # define run name (generate if not given)
        if run_name is None:
            self.run_name = "collider_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_name = run_name
        
        # make base folder and clear tracking directory
        if not os.path.exists(self.run_path()):
            os.makedirs(self.run_path())
        
        # run first linac arm
        if verbose:
            print(">> LINAC #1")
        beam1 = self.linac1.run(self.run_name + "/linac1", num_shots=num_shots, savedepth=savedepth, verbose=verbose, overwrite=overwrite, parallel=parallel, max_cores=max_cores)
        
        # run second linac arm
        if verbose:
            print(">> LINAC #2")
        beam2 = self.linac2.run(self.run_name + "/linac2", num_shots=num_shots, savedepth=savedepth, verbose=verbose, overwrite=overwrite)
        
        # simulate collisions
        if verbose:
            print(">> INTERACTION POINT")
        event = self.ip.run(self.linac1, self.linac2, self.run_name + "/ip", all_by_all=True, overwrite=(overwrite or overwrite_ip))
        
        # return beams from first shot
        return beam1, beam2
    
    
    # plot the distribution of luminosity per power
    def plot_luminosity_per_power(self):
        self.plot_luminosity(per_power=True)
    
    
    # plot the luminosity distribution
    def plot_luminosity(self, per_power=False):
        
        if per_power:
            norm = 1e4 * self.energy_usage()/1e6 # [100 J]
            normLabel = 'per power (cm$^{-2}$ s$^{-1}$ MW$^{-1}$)'
        else:
            norm = 1
            normLabel = 'per crossing (m$^{-2}$)'
            
        # load luminosities
        files = self.ip.run_data()
        num_events = len(files)
        lumi_geom = np.empty(num_events)
        lumi_full = np.empty(num_events)
        lumi_peak = np.empty(num_events)
        for i in range(num_events):
            event = Event.load(files[i], load_beams=False)
            lumi_geom[i] = event.geometric_luminosity() / norm
            lumi_full[i] = event.full_luminosity() / norm
            lumi_peak[i] = event.peak_luminosity() / norm
        
        # calculate fraction of events at zero luminosity
        frac_zero_geom = np.mean(lumi_geom==0)
        frac_zero_full = np.mean(lumi_full==0)
        frac_zero_peak = np.mean(lumi_peak==0)
        
        # prepare figure
        fig, axs = plt.subplots(1,2)
        fig.set_figwidth(18)
        fig.set_figheight(3.5)
        num_bins = max(10, int(np.sqrt(num_events)*10))
        logbins = np.logspace(30, 37, num_bins)/norm
        
        # plot full luminosity
        axs[0].hist(lumi_geom, bins=logbins, color='aliceblue', label="Geometric ("+str(round(frac_zero_geom*100))+"% no lumi.)")
        axs[0].hist(lumi_full, bins=logbins, label="Full ("+str(round(frac_zero_full*100))+"% no lumi.)")
        axs[0].set_xscale("log")
        axs[0].set_xlabel(f"Full luminosity {normLabel}")
        axs[0].set_ylabel('Count per bin')
        axs[0].set_xlim([min(logbins), max(logbins)])
        
        # plot peak luminosity
        axs[1].hist(lumi_geom, bins=logbins, color='aliceblue', label="Geometric ("+str(round(frac_zero_geom*100))+"% no lumi.)")
        axs[1].hist(lumi_peak, bins=logbins, label="Peak 1% ("+str(round(frac_zero_peak*100))+"% no lumi.)")
        axs[1].set_xscale("log")
        axs[1].set_xlabel(f"Peak luminosity {normLabel}")
        axs[1].set_ylabel('Count per bin')
        axs[1].legend(loc='upper left')
        axs[1].set_xlim([min(logbins), max(logbins)])
        
        # plot ILC and CLIC comparisons
        if per_power:
            val_ilc_250_full = 6.15e31 # [cm^-2 s^-1 MW^-1] wall-plug power = 122 MW
            val_ilc_250_peak = 5.35e31 # [cm^-2 s^-1 MW^-1]
            val_ilc_500_full = 1.10e32 # [cm^-2 s^-1 MW^-1] wall-plug power = 163 MW
            val_ilc_500_peak = 6.41e31 # [cm^-2 s^-1 MW^-1]
            val_clic_380_full = 2.09e32 # [cm^-2 s^-1 MW^-1] wall-plug power ≈ 110 MW
            val_clic_380_peak = 1.18e32 # [cm^-2 s^-1 MW^-1]
            val_clic_3000_full = 1.01e32 # [cm^-2 s^-1 MW^-1] wall-plug power = 582 MW
            val_clic_3000_peak = 3.43e31 # [cm^-2 s^-1 MW^-1]
        else:
            val_ilc_250_full = 1.14e34 # [m^-2]
            val_ilc_250_peak = 9.96e33 # [m^-2]
            val_ilc_500_full = 2.74e34 # [m^-2]
            val_ilc_500_peak = 1.60e34 # [m^-2]
            val_clic_380_full = 1.3e34 # [m^-2]
            val_clic_380_peak = 7.39e33 # [m^-2]
            val_clic_3000_full = 3.78e34 # [m^-2]
            val_clic_3000_peak = 1.28e34 # [m^-2]
        axs[0].add_artist(lines.Line2D([val_ilc_250_full, val_ilc_250_full], axs[0].get_ylim(), linestyle='-', color='lightgreen', label='ILC 250'))
        axs[1].add_artist(lines.Line2D([val_ilc_250_peak, val_ilc_250_peak], axs[1].get_ylim(), linestyle='-', color='lightgreen', label='ILC 250'))
        axs[0].add_artist(lines.Line2D([val_ilc_500_full, val_ilc_500_full], axs[0].get_ylim(), linestyle='-', color='forestgreen', label='ILC 500'))
        axs[1].add_artist(lines.Line2D([val_ilc_500_peak, val_ilc_500_peak], axs[1].get_ylim(), linestyle='-', color='forestgreen', label='ILC 500'))
        axs[0].add_artist(lines.Line2D([val_clic_380_full, val_clic_380_full], axs[0].get_ylim(), linestyle='-', color='lightcoral', label='CLIC 380'))
        axs[1].add_artist(lines.Line2D([val_clic_380_peak, val_clic_380_peak], axs[1].get_ylim(), linestyle='-', color='lightcoral', label='CLIC 380'))
        axs[0].add_artist(lines.Line2D([val_clic_3000_full, val_clic_3000_full], axs[0].get_ylim(), linestyle='-', color='indianred', label='CLIC 3000'))
        axs[1].add_artist(lines.Line2D([val_clic_3000_peak, val_clic_3000_peak], axs[1].get_ylim(), linestyle='-', color='indianred', label='CLIC 3000'))
        axs[0].legend(loc='upper left')
        
        
                                     
    