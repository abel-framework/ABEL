from abel.classes.runnable import Runnable
from abel.classes.cost_modeled import CostModeled
from abel.classes.event import Event
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.bds.bds import BeamDeliverySystem
from abel.classes.ip.ip import InteractionPoint
from matplotlib import pyplot as plt
import numpy as np
import os, copy
from matplotlib import lines
from datetime import datetime
from types import SimpleNamespace

class Collider(Runnable, CostModeled):
    
    # constructor
    def __init__(self, linac1=None, linac2=None, ip=None, com_energy=None, energy_asymmetry=None):
        
        self.linac1 = linac1
        self.linac2 = linac2
        self.ip = ip
        self.com_energy = com_energy
        self.energy_asymmetry = energy_asymmetry
        
        self.cost_per_length = 2e5 # [LCU/m]
        self.cost_per_energy = 0.2/(3600*1000)# ILCU/J (0.2 ILCU per kWh)
        self.target_integrated_luminosity = 1e46 # [m^-2] or 1/ab

        # cost overheads
        self.overheads = SimpleNamespace()
        self.overheads.design_and_development = 0.1
        self.overheads.controls_and_cabling = 0.15
        self.overheads.installation_and_commissioning = 0.15
        self.overheads.management_inspection = 0.12
        self.overheads.construction = self.overheads.design_and_development + self.overheads.controls_and_cabling + self.overheads.installation_and_commissioning + self.overheads.management_inspection

        # power overhead
        self.overheads.power = 0.25

        # maintenance costs
        self.maintenance_labor_per_construction_cost = 100/1e9 # [FTE/ILCU/year] # people required for maintaining the machine
        self.cost_per_labor = 0.07e6 # [ILCU/FTE] 
        self.uptime_percentage = 0.7

        # emissions
        self.emissions_per_tunnel_length = 6.38 # [ton CO2e/m] from 6.38 kton/km (CLIC estimate)
        self.emissions_per_energy_usage = 20/(1e9*3600) # [ton CO2e/J] from 20 ton/GWh
        self.cost_carbon_tax_per_emissions = 800 # [ILCU per ton CO2e] 800 from European Investment Bank estimate for 2050

    
    def __str__(self):
        s = f"Collider: {self.com_energy/1e9:.0f} GeV c.o.m. ({self.com_energy/2/self.energy_asymmetry/1e9:.0f} + {self.com_energy/2*self.energy_asymmetry/1e9:.0f} GeV)"
        try:
            s += f", {self.wallplug_power()/1e6:.0f} MW"
        except:
            pass
        try:
            s += f", {self.get_cost()/1e9:.1f} BILCU"
        except:
            pass

        return s
    
     # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assign
        if hasattr(self, 'linac') and self.linac1 is None:
            self.linac1 = self.linac

        # copy second arm if undefined
        if self.linac2 is None:
            self.linac2 = copy.deepcopy(self.linac1)

        # check type
        assert(isinstance(self.linac1, Linac))
        assert(isinstance(self.linac2, Linac))
        assert(isinstance(self.ip, InteractionPoint))
        
        if self.com_energy is not None:
            if self.energy_asymmetry is not None:
                self.linac1.nom_energy = self.com_energy/2 * self.energy_asymmetry
                self.linac2.nom_energy = self.com_energy/2 / self.energy_asymmetry
            elif self.linac1.get_nom_energy() is not None:
                self.energy_asymmetry = 2*self.linac1.get_nom_energy()/self.com_energy
                self.linac2.nom_energy = self.com_energy/2 / self.energy_asymmetry
            elif self.linac2.get_nom_energy() is not None:
                self.energy_asymmetry = 1/(2*self.linac2.nom_energy/self.com_energy)
                self.linac1.nom_energy = self.com_energy/2 * self.energy_asymmetry
            else:
                self.energy_asymmetry = 1
                self.linac1.nom_energy = self.com_energy/2
                self.linac2.nom_energy = self.com_energy/2
        else:
            self.com_energy = self.get_com_energy()
            self.energy_asymmetry = self.get_energy_asymmetry()

        # assign bunch train pattern
        self.linac1.bunch_separation = self.bunch_separation
        self.linac1.num_bunches_in_train = self.num_bunches_in_train
        self.linac1.rep_rate_trains = self.rep_rate_trains
        self.linac2.bunch_separation = self.bunch_separation
        self.linac2.num_bunches_in_train = self.num_bunches_in_train
        self.linac2.rep_rate_trains = self.rep_rate_trains

        # assemble the linacs
        self.linac1.assemble_trackables()
        self.linac2.assemble_trackables()
        

    def get_com_energy(self):
        return 2 * np.sqrt(self.linac1.get_nom_energy() * self.linac2.get_nom_energy())

    def get_energy_asymmetry(self):
        return np.sqrt(self.linac1.get_nom_energy() / self.linac2.get_nom_energy())
        
    # calculate energy usage (per bunch crossing)
    def energy_usage(self):
        return (self.linac1.energy_usage() + self.linac2.energy_usage()) * (1+self.overheads.power)

    def power_overhead(self):
        return (self.linac1.wallplug_power() + self.linac2.wallplug_power()) * self.overheads.power
        
    # calculate wallplug power
    def wallplug_power(self):
        return self.linac1.wallplug_power() + self.linac2.wallplug_power() + self.power_overhead()
        
    
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

    def peak_luminosity(self):
        return self.peak_luminosity_per_power() * self.wallplug_power()
    
    # integrated energy usage (to reach target integrated luminosity)
    def integrated_energy_usage(self):
        return self.target_integrated_luminosity / self.peak_luminosity_per_power()
       
    # total length of linacs (TODO: add driver production length)
    def total_tunnel_length(self):
        Ltot = 0
        for linac in [self.linac1, self.linac2]:
            Ltot += linac.get_length()
            if hasattr(linac, 'driver_complex'):
                Ltot += linac.driver_complex.get_length()
            if hasattr(linac, 'damping_ring'):
                Ltot += linac.damping_ring.get_circumference()
        return Ltot

    # run time
    
    def integrated_runtime(self, in_years=False) -> float:
        "Total integrated run time of the collider to gather enough data [s]/[years]"
        runtime = self.target_integrated_luminosity / self.peak_luminosity()
        if in_years:
            runtime = runtime/(365*24*3600)
        return runtime

    def programme_duration(self, in_years=False):
        "Total programme duration, also accounting for downtime [s]/[years]"
        return self.integrated_runtime(in_years) / self.uptime_percentage

    # costs
    
    def energy_cost(self):
        "Integrated cost of energy [ILC units]"
        return self.integrated_energy_usage() * self.cost_per_energy
        
    def construction_cost(self):
        "Cost of construction [ILC units]"
        return self.linac1.get_cost() + self.linac2.get_cost() + self.ip.get_cost()
    
    # construction overhead costs
    def overhead_cost(self):
        return self.overhead_cost_design_and_development() + self.overhead_cost_controls_and_cabling() + self.overhead_cost_installation_and_commissioning() + self.overhead_cost_management_inspection()
    def overhead_cost_design_and_development(self):
        return self.construction_cost() * self.overheads.design_and_development
    def overhead_cost_controls_and_cabling(self):
        return self.construction_cost() * self.overheads.controls_and_cabling
    def overhead_cost_installation_and_commissioning(self):
        return self.construction_cost() * self.overheads.installation_and_commissioning
    def overhead_cost_management_inspection(self):
        return self.construction_cost() * self.overheads.management_inspection
    
    def total_project_cost(self):
        "Total project cost / ITF cost (US accounting) [ILC units]"
        return self.construction_cost() + self.overhead_cost()

    # required labor force for maintenance (FTEs/year)
    def maintenance_labor(self):
        return self.maintenance_labor_per_construction_cost * self.construction_cost()
        
    def maintenance_cost(self):
        return self.maintenance_labor() * self.cost_per_labor * self.programme_duration(in_years=True)

    def construction_emissions(self):        
        return self.total_tunnel_length() * self.emissions_per_tunnel_length
        
    def energy_emissions(self):        
        return self.integrated_energy_usage() * self.emissions_per_energy_usage

    def total_emissions(self):
        return self.construction_emissions() + self.energy_emissions()
        
    def carbon_tax_cost(self):
        return self.total_emissions() * self.cost_carbon_tax_per_emissions

    def get_cost_breakdown_construction(self):
        breakdown = []
        breakdown.append(self.linac1.get_cost_breakdown())
        breakdown.append(self.linac2.get_cost_breakdown())
        breakdown.append(self.ip.get_cost_breakdown())
        return ('Construction', breakdown)
        
    def get_cost_breakdown_overheads(self):
        breakdown = []
        breakdown.append(('Design/development', self.overhead_cost_design_and_development()))
        breakdown.append(('Constrols/cabling', self.overhead_cost_controls_and_cabling()))
        breakdown.append(('Installation/commissioning', self.overhead_cost_installation_and_commissioning()))
        breakdown.append(('Management/inspection', self.overhead_cost_management_inspection()))
        return ('Overheads', breakdown)
    
    def get_cost_breakdown(self):
        breakdown = []
        breakdown.append(self.get_cost_breakdown_construction())
        breakdown.append(self.get_cost_breakdown_overheads())
        breakdown.append((f'Energy ({self.wallplug_power()/1e6:.0f} MW, {self.programme_duration(in_years=True):.1f} yrs)', self.energy_cost()))
        breakdown.append(('Maintenance', self.maintenance_cost()))
        breakdown.append(('Carbon tax', self.carbon_tax_cost()))
        return ('Collider', breakdown)
    
    # total cost of construction and running
    def full_programme_cost(self, include_carbon_tax=True):
        fpc = self.total_project_cost() + self.energy_cost() + self.maintenance_cost()
        if include_carbon_tax:
            fpc += self.carbon_tax_cost()
        return fpc

    
    def print_emissions(self):
        print('-- EMISSIONS ------------------------------------')
        print('>> Construction emissions ({:.1f} km): {:.0f} kton CO2e'.format(self.total_tunnel_length()/1e3, self.construction_emissions()/1e3))
        print('>> Operation emissions ({:.1f} TWh):   {:.0f} kton CO2e'.format(self.integrated_energy_usage()/(1e12*3600), self.energy_emissions()/1e3))
        print('-------------------------------------------------')
        print('>> Total emissions:                 {:.0f} kton CO2e'.format(self.total_emissions()/1e3))
        print('-------------------------------------------------')

    def print_power(self):
        print(f"-- POWER {'_'.ljust(38,'-')}")
        print('>> Linac 1:                     {:.0f} MW'.format(self.linac1.wallplug_power()/1e6))
        print('>> Linac2:                      {:.0f} MW'.format(self.linac2.wallplug_power()/1e6))
        print('>> Overhead:                    {:.0f} MW'.format(self.power_overhead()/1e6))
        print('-------------------------------------------------')
        print('>> Total power:                 {:.0f} MW'.format(self.wallplug_power()/1e6))
        print('-------------------------------------------------')


    # shot tracking function (to be repeated)
    def perform_shot(self, shot):
        
        self.shot = shot
        self.step = self.steps[shot]
        shot_in_step = np.mod(self.shot, self.num_shots_per_step)
        
        # apply scan function if it exists
        if self.scan_fcn is not None:
            vals_all = np.repeat(self.vals,self.num_shots_per_step)
            self.scan_fcn(self, vals_all[shot])
        
        # check if object exists
        if not self.overwrite and os.path.exists(self.object_path(shot)):
            print('>> SHOT ' + str(shot+1) + ' already exists and will not be overwritten.', flush=True)
            
        else:
    
            # instantiate the trackables
            self.assemble_trackables()
            
            # clear the shot folder
            self.clear_run_data(shot)

            # run tracking
            if self.num_shots > 1 and self.verbose:
                print('>> SHOT ' + str(shot+1) + '/' + str(self.num_shots), flush=True)
            
            # run first linac arm
            if self.verbose:
                print(">> LINAC #1")
            self.linac1.scan(self.run_name + "/linac1", fcn=lambda obj, val: obj, vals=self.vals, num_shots_per_step=self.num_shots_per_step, shot_filter=self.shot, savedepth=self.savedepth, verbose=self.verbose, overwrite=False)
            
            # run second linac arm
            if self.verbose:
                print(">> LINAC #2")
            self.linac2.scan(self.run_name + "/linac2", fcn=lambda obj, val: obj, vals=self.vals, num_shots_per_step=self.num_shots_per_step, shot_filter=self.shot, savedepth=self.savedepth, verbose=self.verbose, overwrite=False)
            
            # simulate collisions
            if self.verbose:
                print(">> INTERACTION POINT")
            event = self.ip.run(self.linac1, self.linac2, self.run_name + "/ip", all_by_all=True, overwrite=False, step_filter=self.step)
            
            # save object to file
            self.save()

    
    # plot survey    
    
    def plot_survey(self, save_fig=False):
        "Plot the layout of the collider"
        
        # setup figure
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(4)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        
        # initialize parameters
        labels = []
        for i, linac in enumerate([self.linac1, self.linac2]):
            
            x = 0
            y = 0
            angle = (i+1) * np.pi
            
            # get and iterate through objects
            objs = linac.survey_object()
            
            # extract secondary objects
            second_objs = None
            connect_to = None
            if len(objs)==2 and isinstance(objs[1], tuple):
                second_objs = objs[1][0]
                connect_to = objs[1][1]
                objs = objs[0]
    
            objs.reverse()
            for j, obj in enumerate(objs):
    
                xs0, ys0, final_angle, label, color = obj
                xs0 -= xs0[-1]
                ys0 -= ys0[-1]
                angle = angle + final_angle
                
                xs = x + xs0*np.cos(angle) + ys0*np.sin(angle)
                ys = y - xs0*np.sin(angle) + ys0*np.cos(angle)
                
                if label in labels:
                    label = None
                else:
                    labels.append(label)
                ax.plot(xs, ys, '-', color=color, label=label, linewidth=2)

                # add secondary objects
                if connect_to == len(objs)-j-1:
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
                    
                x = xs[0]
                y = ys[0]
            
        # add interaction point
        ax.plot([0], [0], 'k*', label='IP', markersize=10)
        
        ax.legend(loc='upper left', ncol=4, reverse=True)
        plt.show()

    
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
        
        
                                     
    