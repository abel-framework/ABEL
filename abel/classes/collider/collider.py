from abel.classes.runnable import Runnable
from abel.classes.cost_modeled import CostModeled
from abel.classes.event import Event
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.bds.bds import BeamDeliverySystem
from abel.classes.ip.ip import InteractionPoint
from abel.CONFIG import CONFIG
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

        self.target_integrated_luminosity = None
        self.target_integrated_luminosity_250GeV = 2e46 # [m^-2] or 1/ab # C3 assumes 2/ab @ 250 GeV and 4/ab at 550 GeV
        self.target_integrated_luminosity_550GeV = 4e46 # [m^-2] or 1/ab # C3 assumes 2/ab @ 250 GeV and 4/ab at 550 GeV

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
            s += f", {self.get_cost()/1e9:.1f} BILCU"
            s += f", {self.length_end_to_end()/1e3:.1f} km"
        except:
            pass

        return s
    
     # assemble the trackables
    def assemble_trackables(self):
        
        if self.target_integrated_luminosity is None:
            if self.com_energy == 250e9:
                self.target_integrated_luminosity = self.target_integrated_luminosity_250GeV
            elif self.com_energy == 250e9:
                self.target_integrated_luminosity = self.target_integrated_luminosity_550GeV
            else:
                self.target_integrated_luminosity = self.target_integrated_luminosity_250GeV + (self.target_integrated_luminosity_550GeV- self.target_integrated_luminosity_250GeV)/(550e9-250e9)*(self.com_energy-250e9)
        
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
        if self.bunch_separation is not None:
            self.linac1.bunch_separation = self.bunch_separation
            self.linac2.bunch_separation = self.bunch_separation
        if self.num_bunches_in_train is not None:
            self.linac1.num_bunches_in_train = self.num_bunches_in_train
            self.linac2.num_bunches_in_train = self.num_bunches_in_train
        if self.rep_rate_trains is not None:
            self.linac1.rep_rate_trains = self.rep_rate_trains
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
        
    
    def get_event(self, load_beams=False):
        return Event.load(self.ip.shot_path(shot1=self.linac1.shot, shot2=self.linac2.shot) + 'event.h5', load_beams=False)
        
    # full luminosity per crossing [m^-2]
    def full_luminosity_per_crossing(self):
        return self.get_event().full_luminosity()
    
    # peak luminosity per crossing [m^-2]
    def peak_luminosity_per_crossing(self):
        return self.get_event().peak_luminosity()

    # peak luminosity per crossing [m^-2]
    def geometric_luminosity_per_crossing(self):
        return self.get_event().geometric_luminosity()

    # enhancement factor (from pinching etc.)
    def enhancement_factor(self):
        event = self.get_event()
        return event.full_luminosity()/event.geometric_luminosity()

    # maximum upsilon parameter (fraction of Schwinger field)
    def maximum_upsilon(self):
        return self.get_event().maximum_upsilon()

    # number of coherent pairs produced
    def num_coherent_pairs(self):
        return self.get_event().num_coherent_pairs()

    # number of synchrotron photons produced
    def num_photons_beam1(self):
        return self.get_event().num_photons_beam1()
    def num_photons_beam2(self):
        return self.get_event().num_photons_beam2()

    # energy lost per particle through synchrotron radiation
    def energy_loss_per_particle_beam1(self):
        return self.get_event().energy_loss_per_particle_beam1()
    def energy_loss_per_particle_beam2(self):
        return self.get_event().energy_loss_per_particle_beam2()
        
    


    # full luminosity per power [m^-2/J]
    def full_luminosity_per_power(self):
        return self.full_luminosity_per_crossing() / self.energy_usage()
    
    # peak luminosity per power [m^-2/J]
    def peak_luminosity_per_power(self):
        return self.peak_luminosity_per_crossing() / self.energy_usage()

    def full_luminosity(self):
        return self.full_luminosity_per_crossing() * self.collision_rate()
        
    def peak_luminosity(self):
        return self.peak_luminosity_per_crossing() * self.collision_rate()
    
    # integrated energy usage (to reach target integrated luminosity)
    def integrated_energy_usage(self):
        return self.target_integrated_luminosity / self.peak_luminosity_per_power()

    def collision_rate(self):
        return self.linac1.rep_rate_average
        
    def length_end_to_end(self, return_arm_lengths=False):

        x_mins = [0,0]
        x_maxs = [0,0]
        
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

                        x_mins[i] = min(x_mins[i], np.min(x2))
                        x_maxs[i] = max(x_maxs[i], np.max(x2))
                        
                x = xs[0]
                y = ys[0]

                x_mins[i] = min(x_mins[i], np.min(xs))
                x_maxs[i] = max(x_maxs[i], np.max(xs))
        
        x_min = min(x_mins)
        x_max = max(x_maxs)

        if return_arm_lengths:
            return abs(x_max-x_min), -x_min, x_max
        else:
            return abs(x_max-x_min)
                

        
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
        return self.integrated_energy_usage() * CostModeled.cost_per_energy

    def construction_cost(self):
        return self.construction_cost_bare() + self.construction_cost_infrastructure_and_services() + self.construction_cost_controls_protection_and_safety()
        
    def construction_cost_bare(self):
        "Cost of construction, before services etc. [ILC units]"
        return self.linac1.get_cost() + self.linac2.get_cost() + self.ip.get_cost()
    
    def construction_cost_infrastructure_and_services(self):
        "Cost of infrastructure and services [ILC units]"
        return self.construction_cost_bare() * CostModeled.cost_factor_infrastructure_and_services

    def construction_cost_controls_protection_and_safety(self):
        "Cost of machine control, protection and safety systems [ILC units]"
        return self.construction_cost_bare() * CostModeled.cost_factor_controls_protection_and_safety
        
    
    
    # construction overhead costs
    def overhead_cost(self):
        #return self.overhead_cost_design_and_development() + self.overhead_cost_controls_and_cabling() + self.overhead_cost_installation_and_commissioning() + self.overhead_cost_management_inspection()
        return self.overhead_cost_design_and_development() + self.overhead_cost_management_inspection()
    def overhead_cost_design_and_development(self):
        return self.construction_cost() * self.overheads.design_and_development
    #def overhead_cost_controls_and_cabling(self):
    #    return self.construction_cost() * self.overheads.controls_and_cabling
    #def overhead_cost_installation_and_commissioning(self):
    #    return self.construction_cost() * self.overheads.installation_and_commissioning
    def overhead_cost_management_inspection(self):
        return self.construction_cost() * self.overheads.management_inspection

    def cost_snowmass_itf_accounting(self):
        "Total project cost (Snowmass ITF accounting) [ILC units]"
        return self.construction_cost() + self.overhead_cost()

    def cost_eu_accounting(self):
        "Project cost (EU accounting) [ILC units]"
        return self.construction_cost()
        
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
        breakdown.append(('Experimental area', CostModeled.cost_per_experimental_area))
        breakdown_post = []
        breakdown_post.append((f'{self.linac1.name} beamline', 300*np.sqrt(self.linac1.nom_energy/500e9)*CostModeled.cost_per_length_bds))
        breakdown_post.append((f'{self.linac2.name} beamline', 300*np.sqrt(self.linac2.nom_energy/500e9)*CostModeled.cost_per_length_bds))
        breakdown_post.append((f'{self.linac1.name} dump', self.linac1.get_nom_beam_power()*CostModeled.cost_per_power_beam_dump))
        breakdown_post.append((f'{self.linac2.name} dump', self.linac2.get_nom_beam_power()*CostModeled.cost_per_power_beam_dump))
        breakdown.append(('Post-collision beamlines', breakdown_post))
        breakdown.append(('Infrastructure & services', self.construction_cost_infrastructure_and_services()))
        breakdown.append(('Controls, protection & safety', self.construction_cost_controls_protection_and_safety()))
        return ('Construction', breakdown)
        
    def get_cost_breakdown_overheads(self):
        breakdown = []
        breakdown.append(('Design/development', self.overhead_cost_design_and_development()))
        #breakdown.append(('Constrols/cabling', self.overhead_cost_controls_and_cabling()))
        #breakdown.append(('Installation/commissioning', self.overhead_cost_installation_and_commissioning()))
        breakdown.append(('Management/inspection', self.overhead_cost_management_inspection()))
        return ('Overheads', breakdown)
    
    def get_cost_breakdown(self):
        breakdown = []
        breakdown.append(self.get_cost_breakdown_construction())
        breakdown.append(self.get_cost_breakdown_overheads())
        breakdown.append((f'Energy ({self.wallplug_power()/1e6:.0f} MW, {self.target_integrated_luminosity/1e46:.0f}/ab, {self.programme_duration(in_years=True):.1f} yrs)', self.energy_cost()))
        breakdown.append(('Maintenance', self.maintenance_cost()))
        breakdown.append((f'Carbon tax ({self.total_emissions()/1e3:.0f} kton CO2e)', self.carbon_tax_cost()))
        return ('Collider', breakdown)
    
    # total cost of construction and running
    def full_programme_cost(self, include_carbon_tax=True):
        fpc = self.construction_cost() + self.overhead_cost() + self.energy_cost() + self.maintenance_cost()
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
        print(f"-- POWER {'-'.ljust(40,'-')}")
        print(f">> {self.linac1.name}:                       {self.linac1.wallplug_power()/1e6:.0f} MW")
        print(f">> {self.linac2.name}:                 {self.linac2.wallplug_power()/1e6:.0f} MW")
        print('>> Overhead:                          {:.0f} MW'.format(self.power_overhead()/1e6))
        print('-------------------------------------------------')
        print('>> Total power:                 {:.0f} MW'.format(self.wallplug_power()/1e6))
        print('-------------------------------------------------')


    # shot tracking function (to be repeated)
    def perform_shot(self, shot):
        
        self.shot = shot
        self.step = self.steps[shot]
        shot_in_step = np.mod(self.shot, self.num_shots_per_step)

        # check if object exists
        if not self.overwrite and os.path.exists(self.object_path(shot)):
            print('>> SHOT ' + str(shot+1) + ' already exists and will not be overwritten.', flush=True)
        else:
            
            # apply scan function if it exists (before assembly)
            vals_all = np.repeat(self.vals,self.num_shots_per_step)
            applied_fnc_before_assembly = False
            try:
                if self.scan_fcn is not None:
                    self.scan_fcn(self, vals_all[shot])
                    applied_fnc_before_assembly = True
            except:
                pass
            
            # instantiate the trackables
            self.assemble_trackables()

            # apply scan function if it exists (if it wasn't done before assembly)
            if not applied_fnc_before_assembly and self.scan_fcn is not None:
                self.scan_fcn(self, vals_all[shot])
            
            # clear the shot folder
            self.clear_run_data(shot)

            # run tracking
            if self.num_shots > 1 and self.verbose:
                print('>> SHOT ' + str(shot+1) + '/' + str(self.num_shots), flush=True)
            
            # run first linac arm
            if self.verbose:
                print(">> LINAC #1")
            self.linac1.scan(self.run_name + "/linac1", fcn=lambda obj, val: obj, vals=self.vals, num_shots_per_step=self.num_shots_per_step, shot_filter=shot, savedepth=self.savedepth, verbose=self.verbose, overwrite=False)
            
            # run second linac arm
            if self.verbose:
                print(">> LINAC #2")
            self.linac2.scan(self.run_name + "/linac2", fcn=lambda obj, val: obj, vals=self.vals, num_shots_per_step=self.num_shots_per_step, shot_filter=shot, savedepth=self.savedepth, verbose=self.verbose, overwrite=False)
            
            # simulate collisions
            if self.verbose:
                print(">> INTERACTION POINT")
            event = self.ip.run(self.linac1, self.linac2, self.run_name + "/ip", all_by_all=False, overwrite=self.overwrite, step_filter=self.step, verbose=self.verbose)
            
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



    ## SCAN PLOTS

    # TODO: delete data option
    def plot_cost_variation(self, param, scan_name=None, num_shots_per_step=1, num_steps=11, lower=None, upper=None, scale=1, label=None, xscale='log', parallel=False, overwrite=True):
        
        # set default scan name
        if scan_name is None:
            scan_name = 'param_scan_' + param + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

        # set default upper and lower bounds
        if lower is None:
            lower = self.get_nested_attr(param)*0.7
        if upper is None:
            upper = self.get_nested_attr(param)*1.3

        # default label
        if label is None:
            label = param

        # set values based on type of scale
        if xscale == 'log':
            scan_vals = np.logspace(np.log10(lower), np.log10(upper), num_steps)
        elif xscale == 'linear':
            scan_vals = np.linspace(lower, upper, num_steps)

        full_programme_cost = np.zeros(num_steps)
        itf_cost = np.zeros(num_steps)
        eu_cost = np.zeros(num_steps)
        construction_cost = np.zeros(num_steps)
        construction_cost_infrastructure = np.zeros(num_steps)
        construction_cost_cps = np.zeros(num_steps)
        construction_cost_linac1 = np.zeros(num_steps)
        construction_cost_linac2 = np.zeros(num_steps)
        overhead_cost = np.zeros(num_steps)
        energy_cost = np.zeros(num_steps)
        maintenance_cost = np.zeros(num_steps)
        carbon_tax_cost = np.zeros(num_steps)
        for i in range(num_steps):
            scan_self = copy.deepcopy(self)
            scan_self.set_attr(param, scan_vals[i])
            scan_self.run(scan_name, num_shots=num_shots_per_step, parallel=parallel, overwrite=overwrite, verbose=False)

            # extract values (TODO: allow for more shots per step)
            full_programme_cost[i] = scan_self.full_programme_cost()
            itf_cost[i] = scan_self.cost_snowmass_itf_accounting()
            eu_cost[i] = scan_self.cost_eu_accounting()
            
            construction_cost[i] = scan_self.construction_cost()
            construction_cost_infrastructure[i] = scan_self.construction_cost_infrastructure_and_services()
            construction_cost_cps[i] = scan_self.construction_cost_controls_protection_and_safety()
            construction_cost_linac1[i] = scan_self.linac1.get_cost()
            construction_cost_linac2[i] = scan_self.linac2.get_cost()
            
            overhead_cost[i] = scan_self.overhead_cost()
            energy_cost[i] = scan_self.energy_cost()
            maintenance_cost[i] = scan_self.maintenance_cost()
            carbon_tax_cost[i] = scan_self.carbon_tax_cost()
            
        construction_cost_ip = construction_cost - construction_cost_linac1 - construction_cost_linac2 - construction_cost_infrastructure - construction_cost_cps
        
        # plot cost breakdown
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)

        cumul_cost = construction_cost_ip
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.zeros_like(scan_vals))), 'steelblue', label='Construction (IP)')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += construction_cost_linac1
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), 'skyblue', label=f'Construction ({scan_self.linac1.name})')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += construction_cost_linac2
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), 'lightblue', label=f'Construction ({scan_self.linac2.name})')

        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += construction_cost_infrastructure
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), 'lightskyblue', label='Infrastructure & services')

        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += construction_cost_cps
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), 'deepskyblue', label='Controls, protection & safety')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += overhead_cost
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), label='Overhead cost')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += energy_cost
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), label='Energy cost')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += maintenance_cost
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), label='Maintenance cost')
        
        cumul_cost_last = copy.deepcopy(cumul_cost)
        cumul_cost += carbon_tax_cost
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_cost/1e9, np.flip(cumul_cost_last/1e9))), label='Carbon tax')

        ax.plot(scan_vals/scale, eu_cost/1e9, ls='--', color='k', label='Cost (EU accounting)')
        ax.plot(scan_vals/scale, itf_cost/1e9, ls=':', color='k', label='Cost (Snowmass ITF accounting)')
        ax.plot(scan_vals/scale, full_programme_cost/1e9, ls='-', color='k', label='Full programme cost')
        
        
        ax.set_xlabel(label)
        ax.set_ylabel('Collider cost (BILCU)')
        ax.set_xscale(xscale)
        ax.set_yscale('linear')
        ax.set_xlim(lower/scale, upper/scale)
        ax.set_ylim(0, 1.25*max(full_programme_cost/1e9))
        ax.plot(np.array([1,1])*self.get_nested_attr(param)/scale, [0, 1.25*max(full_programme_cost/1e9)], ls=':', color='gray')
        plt.legend(reverse=True)

        return scan_self

    # TODO: delete data option
    def plot_length_variation(self, param, scan_name=None, num_shots_per_step=1, num_steps=11, lower=None, upper=None, scale=1, label=None, xscale='log', parallel=False, overwrite=True):
        
        # set default scan name
        if scan_name is None:
            scan_name = 'param_scan_' + param + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

        # set default upper and lower bounds
        if lower is None:
            lower = self.get_nested_attr(param)*0.7
        if upper is None:
            upper = self.get_nested_attr(param)*1.3

        # default label
        if label is None:
            label = param

        # set values based on type of scale
        if xscale == 'log':
            scan_vals = np.logspace(np.log10(lower), np.log10(upper), num_steps)
        elif xscale == 'linear':
            scan_vals = np.linspace(lower, upper, num_steps)

        length_end_to_end = np.zeros(num_steps)
        length_linac1 = np.zeros(num_steps)
        length_linac2 = np.zeros(num_steps)
        for i in range(num_steps):
            scan_self = copy.deepcopy(self)
            scan_self.set_attr(param, scan_vals[i])
            scan_self.run(scan_name, num_shots=num_shots_per_step, parallel=parallel, overwrite=overwrite, verbose=False)

            # extract values (TODO: allow for more shots per step)
            length_end_to_end[i], length_linac2[i], length_linac1[i] = scan_self.length_end_to_end(return_arm_lengths=True)
            
        # plot cost breakdown
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)

        cumul_length = length_linac2
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_length/1e3, np.zeros_like(scan_vals))), label=scan_self.linac2.name)
        
        cumul_length_last = copy.deepcopy(cumul_length)
        cumul_length += length_linac1
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_length/1e3, np.flip(cumul_length_last/1e3))), label=scan_self.linac1.name)
        
        ax.plot(scan_vals/scale, length_end_to_end/1e3, ls='-', color='k', label='End-to-end')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Collider length (km)')
        ax.set_xscale(xscale)
        ax.set_yscale('linear')
        ax.set_xlim(lower/scale, upper/scale)
        ax.set_ylim(0, 1.1*max(length_end_to_end/1e3))
        ax.plot(np.array([1,1])*self.get_nested_attr(param)/scale, [0, 1.1*max(length_end_to_end/1e3)], ls=':', color='gray')
        plt.legend(reverse=True)

        return scan_self

    # TODO: delete data option
    def plot_power_variation(self, param, scan_name=None, num_shots_per_step=1, num_steps=11, lower=None, upper=None, scale=1, label=None, xscale='log', parallel=False, overwrite=True):
        
        # set default scan name
        if scan_name is None:
            scan_name = 'param_scan_' + param + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

        # set default upper and lower bounds
        if lower is None:
            lower = self.get_nested_attr(param)*0.7
        if upper is None:
            upper = self.get_nested_attr(param)*1.3

        # default label
        if label is None:
            label = param

        # set values based on type of scale
        if xscale == 'log':
            scan_vals = np.logspace(np.log10(lower), np.log10(upper), num_steps)
        elif xscale == 'linear':
            scan_vals = np.linspace(lower, upper, num_steps)

        power_overhead = np.zeros(num_steps)
        power_linac1 = np.zeros(num_steps)
        power_linac2 = np.zeros(num_steps)
        for i in range(num_steps):
            scan_self = copy.deepcopy(self)
            scan_self.set_attr(param, scan_vals[i])
            scan_self.run(scan_name, num_shots=num_shots_per_step, parallel=parallel, overwrite=overwrite, verbose=False)

            # extract values (TODO: allow for more shots per step)
            power_linac1[i] = scan_self.linac1.wallplug_power()
            power_linac2[i] = scan_self.linac2.wallplug_power()
            power_overhead[i] = scan_self.power_overhead()

        power_total = power_overhead + power_linac1 + power_linac2
        
        # plot cost breakdown
        fig, ax = plt.subplots(1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.6)

        cumul_power = power_linac2
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_power/1e6, np.zeros_like(scan_vals))), label=scan_self.linac2.name)
        
        cumul_power_last = copy.deepcopy(cumul_power)
        cumul_power += power_linac1
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_power/1e6, np.flip(cumul_power_last/1e6))), label=scan_self.linac1.name)

        cumul_power_last = copy.deepcopy(cumul_power)
        cumul_power += power_overhead
        ax.fill(np.concatenate((scan_vals/scale, np.flip(scan_vals/scale))), np.concatenate((cumul_power/1e6, np.flip(cumul_power_last/1e6))), label='Overhead')
        
        ax.plot(scan_vals/scale, power_total/1e6, ls='-', color='k', label='Total')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Collider power (MW)')
        ax.set_xscale(xscale)
        ax.set_yscale('linear')
        ax.set_xlim(lower/scale, upper/scale)
        ax.set_ylim(0, 1.1*max(power_total/1e6))
        ax.plot(np.array([1,1])*self.get_nested_attr(param)/scale, [0, 1.1*max(power_total/1e6)], ls=':', color='gray')
        plt.legend(reverse=True)

        return scan_self
        
    
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
        
        
                                     
    