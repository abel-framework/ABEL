from abc import ABC, abstractmethod
import numpy as np

class CostModeled(ABC):

    # cost per length of tunnel
    # REF: ILC TDR 2013 (using 500 GeV example), same as for FCC
    cost_per_length_tunnel = 0.06e6 # [ILCU/m]

    # cost per length of RF structure, not including klystrons (ILC is 0.24e6 with power)
    # REF: CLIC CDR update 2018 (using 380 GeV klystron-based example)
    # (1.0225/1.1) * 895 MILCU /(5650*2 structures * (0.46/0.718) m/structure)
    cost_per_length_rf_structure_normalconducting = 0.115e6 # [ILCU/m]

    # cost per length of RF structure/everything
    # REF: ILC TDR 2013 (using 500 GeV example)
    # 2.753 BILCU / (17804 structures * (1.038/0.711) m/structure)
    cost_per_length_rf_structure_superconducting = 0.106e6 # [ILCU/m]

    # TODO: add cost of dumps
    # TODO: add cost of RTML
    # TODO: add cost of power infrastructure
    # TODO: add cost of cryo-plants

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
    cost_per_length_transport_line = 0.0159e6 # [ILCU/m]
    
    # cost of source
    cost_per_source = 50e6 # [ILCU]

    # cost of polarized source
    # REF: ILC TDR 2013 (using 500 GeV example)
    cost_per_source_polarized_electrons = 96e6 # [ILCU]
    cost_per_source_polarized_positrons = 192e6 # [ILCU]

    # cost of turnarounds
    cost_per_length_turnaround = 0.025e6 # [ILCU/m]

    # cost of interstages
    cost_per_length_interstage = 0.095e6#0.04044e6 # [ILCU/m] as BDS

    # cost of plasma stages
    cost_per_length_plasma_stage = 0.095e6 # [ILCU/m] vessel, HV source/laser

    # cost of interaction point (the halls etc.)
    # REF: ILC TDR 2013 (using 500 GeV example)
    cost_per_ip = 184e6 # [ILCU]

    # cost of damping ring
    # REF: ILC TDR 2013 (two rings)
    cost_per_length_damping_ring = 0.0517e6 # [ILCU/m]

    # cost of BDS
    cost_per_length_bds = 0.04044e6 # [ILCU/m]

    # cost of beam dumps (TODO: add more detail and per power costs)
    cost_per_driver_dump = 1e6 # [ILCU]
    
    # REF: R. Pitthan SLAC-PUB-8570 (August 2000) says it scales as frequency squared, but we find that for ILC and CLIC, it scales as frequency linearly.
    @classmethod
    def cost_per_klystron(cls, num_klystrons, rf_frequency, avarage_power_klystron):
        "Cost per klystron, including modulator [ILC units]"
        modulator_multiplier = 2
        #return cls.cost_per_frequency_per_power_klystron * modulator_multiplier * avarage_power_klystron * rf_frequency**2 * num_klystrons**np.log2(1-cls.learning_curve_klystrons)
        return cls.cost_per_frequency_per_power_klystron * modulator_multiplier * avarage_power_klystron * rf_frequency * num_klystrons**np.log2(1-cls.learning_curve_klystrons)

    
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
            
            
            