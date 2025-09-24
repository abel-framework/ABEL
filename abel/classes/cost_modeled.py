from abc import ABC, abstractmethod
import numpy as np

class CostModeled(ABC):

    # from CLIC-K 2018 
    cost_factor_electrical_distribution = 0.03933
    cost_factor_survey_alignment = 0.02379
    cost_factor_cooling_ventilation = 0.06636
    cost_factor_transport_installation = 0.00583
    
    cost_factor_infrastructure_and_services = cost_factor_electrical_distribution + cost_factor_survey_alignment + cost_factor_cooling_ventilation + cost_factor_transport_installation
    
    cost_factor_safety_systems = 0.01845
    cost_factor_machine_control_infrastructure = 0.02120
    cost_factor_machine_protection = 0.00129
    cost_factor_access_safety_control_system = 0.00372
    
    cost_factor_controls_protection_and_safety = cost_factor_safety_systems + cost_factor_machine_protection + cost_factor_machine_protection + cost_factor_access_safety_control_system
    
    # cost per length of tunnel
    # REF: ILC TDR 2013 (using 500 GeV example), same as for FCC
    #cost_per_length_tunnel = 0.06e6 # [ILCU/m]

    cost_per_volume_tunnel = 714.75/(1.2*1.04) # [ILCU/m^3] CLIC cost, converted from 2018 EUR to USD
    cost_per_volume_tunnel_widening = 561.65/(1.2*1.04) # [ILCU/m^3] FCC cost, converted from 2018 EUR to USD
    
    # 4.3 meter diameter tunnel (transport tunnels)
    cost_per_length_tunnel_small = 14835.64/(1.2*1.04) # [ILCU/m] (CLIC cost, converted from 2018 EUR to USD, deflated from 2018 to 2012, inflated by 75% today [31% in 2012])
    
    # 5.6 meter diameter tunnel (damping ring tunnels etc.)
    cost_per_length_tunnel_medium = 25199.694/(1.2*1.04) # [ILCU/m] (CLIC cost, converted from 2018 EUR to USD, deflated from 2018 to 2012, inflated by 75% today [31% in 2012])

    # 8 meter diameter tunnel (widened tunnel)
    cost_per_length_tunnel_large = 25199.694*2.0408/(1.2*1.04) # [ILCU/m] (CLIC cost, converted from 2018 EUR to USD, deflated from 2018 to 2012, inflated by 75% today [31% in 2012])

    cost_per_length_surfacebuilding = 41506/(1.2*1.04) # [ILCU/m] (CLIC cost, converted from 2018 EUR to USD, deflated from 2018 to 2012)
    #cost_per_length_cutandcover_small = 0.008768e6*1.1811*0.9139 # [ILCU/m] (CLIC cost, converted from 2018 EUR to USD, deflated from 2018 to 2012)
    cost_per_length_cutandcover = 12308/(1.2*1.04) # [ILCU/m] (CLIC cost, converted from 2018 EUR to ILCU)

    # cost per length of RF structure, not including klystrons (ILC is 0.24e6 with power)
    # REF: CLIC CDR update 2018 (using 380 GeV klystron-based example)
    # (1.0225/1.1) * 895 MILCU /(5650*2 structures * (0.46/0.718) m/structure)
    cost_per_length_rf_structure_normalconducting = 0.115e6 # [ILCU/m]

    # cost per length of RF structure/everything
    # REF: ILC TDR 2013 (using 500 GeV example)
    # 2.753 BILCU / (17804 structures * (1.038/0.711) m/structure)
    cost_per_length_rf_structure_superconducting = 0.106e6 # [ILCU/m]
    
    # TODO: add cost of RTML
    # TODO: add cost of power infrastructure
    # TODO: add cost of He cryo-plants

    # cost of klystrons
    # REF: ILC TDR 2013 (using 500 GeV example)
    # 10% added spares
    #cost_per_frequency_per_power_klystron = 1.1 * 1.475 *1.06e6/(125e3*1.3e9**2) # [ILCU/Hz^2/W]
    cost_per_frequency_per_power_klystron = 1.1 * 1.06e6/(125e3*1.3e9) # [ILCU/Hz/W]
    learning_curve_klystrons = 0.00 # fraction cheaper per doubled output

    # REF: R. Pitthan SLAC-PUB-8570 (August 2000)
    #cost_per_frequency_per_power_klystron = 1.33*110e3/(100e3*952e6**2) # [ILCU/Hz^2/W]
    #learning_curve_klystrons = 0.1 # fraction cheaper per doubled output

    # cost of transport line
    # REF: ILC TDR 2013 (using 500 GeV example, based on RTML)
    # 477 MILCU for 30 km of RTML
    #cost_per_length_transport_line = 0.0159e6 # [ILCU/m]
    
    # cost of source (thermionic; guesstimate)
    cost_per_source = 10e6 # [ILCU]

    # cost of polarized source
    # REF: ILC TDR 2013 (using 500 GeV example) same as ILC but subtracting 50 MILCU for the 5 GeV linac.
    cost_per_source_polarized_electrons = 82e6 # [ILCU]
    cost_per_source_polarized_positrons = 178e6 # [ILCU]

    # cost of turnarounds (as BDS)
    #cost_per_length_turnaround = 0.025e6 # [ILCU/m]
    cost_per_length_turnaround = 0.04044e6 # [ILCU/m]

    # cost of combiner rings (scaled to fit CLIC cost)
    cost_per_length_combiner_ring = 0.079e6 # [ILCU/m]
    cost_per_rfkicker_combiner_ring = 1e6 # [ILCU/kicker]

    # cost of interstages
    cost_per_length_interstage = 0.04044e6 # [ILCU/m] as BDS

    # cost of interstages
    cost_per_length_driver_delay_chicane = 0.04044e6 # [ILCU/m] as BDS due to many dipoles

    # cost of plasma stages
    #cost_per_length_plasma_stage = 0.095e6 # [ILCU/m] vessel, HV source/laser
    cost_per_length_plasma_stage = 3*0.0154e6 # [ILCU/m] 3 times instrumented beamline (diagnostics + cell material + guiding)

    # cost of interaction point (the halls etc.)
    # REF: ILC TDR 2013 (using 500 GeV example)
    cost_per_ip = (191.6e6/(1.2*1.04))/2 # [ILCU] Half of CLIC dual IP cost.

    cost_per_experimental_area = 20e6 # [ILCU] (from CLIC; 22 MCHF in 2018 deflated to 2012, not converted to dollar)

    # cost of damping ring
    # REF: ILC TDR 2013 (two rings)
    #cost_per_length_damping_ring = 0.0517e6 # [ILCU/m]
    cost_per_length_damping_ring = 0.260e6 # [ILCU/m] # REF: CLIC CDR 2018 (average of 3 rings)
    

    # cost of BDS
    cost_per_length_bds = 0.04044e6 # [ILCU/m]

    # cost of transfer lines
    cost_per_length_transfer_line = 0.0154e6 # [ILCU/m] REF: ILC (477 MILCU for 31 km)

    # cost per length of instrumented beamline (as transfer line)
    cost_per_length_instrumented_beamline = 0.0154e6 # [ILCU/m]

    # cost of beam dumps (scaled from ILC 1 TeV)
    cost_per_power_beam_dump = 67e6/(14e6*2) # [ILCU/W] based on cost for two 14 MW beams

    # cost of energy
    cost_per_energy = 0.05/(3600*1000)# ILCU/J (50 euros per MWh, based on CERN estimate)

    # cost of nitrogen reliquification plant per (cold) power
    # REF: https://accelconf.web.cern.ch/ipac2023/pdf/WEZG2_talk.pdf (page 25)
    cost_per_power_reliquification_plant_nitrogen = 18.0 * 0.75 # ILCU/W (18M$/MW in 2023)

    # cost of helium reliquification plant per (cold) power (TODO)
    cost_per_coldpower_reliquification_plant_helium = None
    
    
    @classmethod
    def cost_per_klystron(cls, num_klystrons, rf_frequency, average_power_klystron, peak_power_klystron):
        "Cost per klystron, including modulator, LLRF and waveguides [ILC units]"

        if rf_frequency == 1e9: # L-band
            
            # for L-band (multi-beam klystron), by Erk Jensen
            # reference: https://indico.cern.ch/event/275412/contributions/1617607/attachments/498755/688976/LbandKlystronDevelop.pdf
            model_now_peakpower = [1.000e6, 15.871e6, 31.594e6, 100.363e6]
            model_now_relcost = [0.31586, 0.07945, 0.07945, 0.80379]
            model_future_peakpower = [1.000e6, 25.146e6, 50.055e6, 100.727e6]
            model_future_relcost = [0.31586, 0.062978, 0.062978, 0.25507]
            rel_cost_per_peak_power = np.interp(peak_power_klystron, model_now_peakpower, model_now_relcost)
    
            # CLIC klystron cost in 2018 (15 MW, average over 470 with 0.92 learning curve)
            ref_peak_power = 15e6
            rel_cost_per_peak_power_ref = np.interp(ref_peak_power, model_now_peakpower, model_now_relcost)
            cost_klystron = (rel_cost_per_peak_power/rel_cost_per_peak_power_ref) * (peak_power_klystron/ref_peak_power) * 380e3 * 0.91 
    
            # cost of modulator (assumed to scale with average power)
            cost_modulator = average_power_klystron * 370e3/112.5e3 * 0.91 # CLIC modulator cost in 2018 (average over 470 with 0.92 learning curve)

        else: # rf_frequency == 3e9: # S-band
            
            # S-band C^3-like klystron
            cost_per_peak_power_klystron = 0.012 * 0.75 # [ILCU/peak W]
            cost_klystron = peak_power_klystron * cost_per_peak_power_klystron # CLIC modulator cost in 2023, rough
    
            # cost of modulator (assumed to scale with average power)
            cost_per_peak_power_modulator = 0.008 * 0.75 # [ILCU/peak W]
            cost_modulator = peak_power_klystron * cost_per_peak_power_modulator
        
        # cost of LLRF
        cost_llrf = 50e3 * 0.91

        # cost of waveguides
        cost_waveguides = 30e3 * 0.91

        # TODO: make breakdown of klystron costs
        
        return cost_klystron + cost_modulator + cost_llrf + cost_waveguides

    
    @abstractmethod
    def get_cost_breakdown(self):
        "Get the breakdown sub-costs for the element [ILC units]"
        pass
        
    
    def get_cost(self) -> float:
        "Get the cost of the element [ILC units]"
        return self.__get_cost_recursive()

    def __get_cost_recursive(self, elements=None) -> float:
        "Recursive helper function to get the cost of all sub-elements [ILC units]"
        
        if elements is None:
            name, elements = self.get_cost_breakdown()

        if isinstance(elements, list):
            cost = 0.0
            for element in elements:
                if isinstance(element[1], list):
                    cost += self.__get_cost_recursive(element[1])
                else:
                    cost += element[1]
        else:
            cost = elements
                
        return cost
    
    def print_cost(self):
        "Print the cost breakdown"

        print('-- COSTS '.ljust(50,'-'))
        self.__print_cost_recursive()
        print(''.ljust(50,'-'))
        print('-- Total: {:.2f} BILCU'.format(self.get_cost()/1e9))
        print(''.ljust(50,'-'))

    def __print_cost_recursive(self, elements=None, level=0) -> float:
        "Recursive helper function to get the cost of all sub-elements [ILC units]"
        
        if elements is None:
            name, elements = self.get_cost_breakdown()

        if isinstance(elements, list):
            for element in elements:
                if isinstance(element[1], list):
                    print(f"{'-- '.rjust(3*level+3)}{element[0]}: {round(self.__get_cost_recursive(element[1])/1e6)} MILCU")
                    self.__print_cost_recursive(element[1], level+1)
                else:
                    print(f"{'-- '.rjust(3*level+3)}{element[0]}: {round(element[1]/1e6)} MILCU")
        else:
            print(f"{'-- '.rjust(3*level+3)}{name}: {round(elements/1e6)} MILCU")

            
    ignore_cost_civil_construction = False
    
    def get_cost_civil_construction(self, tunnel_diameter=None, cut_and_cover=False, surface_building=False, tunnel_widening_factor=None):    
        "Get the civil engineering cost of the element [ILC units]"
        if self.ignore_cost_civil_construction:
            return 0
        else:
            total_cost = 0
            if cut_and_cover:
                total_cost = self.get_length() * self.cost_per_length_cutandcover
                if surface_building:
                    total_cost += self.get_length() * self.cost_per_length_surfacebuilding
                return total_cost
            else:
                if tunnel_diameter is None:
                    return self.get_length() * self.cost_per_length_tunnel_large
                else:
                    tunnel_area = np.pi*((tunnel_diameter+1.1)/2)**2
                    tunnel_cost = self.get_length() * tunnel_area * self.cost_per_volume_tunnel
                    if tunnel_widening_factor is not None:
                        widening_area = tunnel_area * (tunnel_widening_factor-1)
                        widening_cost = self.get_length() * widening_area * self.cost_per_volume_tunnel_widening
                        tunnel_cost += widening_cost
                    return tunnel_cost