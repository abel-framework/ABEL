from abc import ABC, abstractmethod
import numpy as np

class CostModeled(ABC):

    # cost per length of tunnel (FCC estimate)
    cost_per_length_tunnel = 0.06e6 # [ILCU/m]

    # cost per length of RF structure, not including klystrons (ILC is 0.24e6 with power)
    cost_per_length_rf_structure = 0.22e6 # [ILCU/m]

    # cost of klystrons
    cost_per_frequency_per_power_klystron = 1.33*110e3/(100e3*952e6**2) # [ILCU/Hz^2/W]
    learning_curve_klystrons = 0.1 # fraction cheaper per doubled output

    # cost of source
    cost_per_source = 9.4e6 # [ILCU]

    # cost of turnarounds
    cost_per_length_turnaround = 0.2e6 # [ILCU/m]

    # cost of interstages
    cost_per_length_interstage = 0.1e6 # [ILCU/m]

    # cost of plasma stages
    cost_per_length_plasma_stage = 0.2e6 # [ILCU/m]

    # cost of interaction point (the halls etc.)
    cost_per_ip = 50e6 # [ILCU]

    # cost of damping ring
    cost_per_length_damping_ring = 0.1e6 # [ILCU/m]

    # cost of BDS
    cost_per_length_bds = 0.05e6 # [ILCU/m]
    
    
    @classmethod
    def cost_per_klystron(cls, num_klystrons, rf_frequency, avarage_power_klystron):
        "Cost per klystron, including modulator [ILC units]"
        modulator_multiplier = 2
        return cls.cost_per_frequency_per_power_klystron * modulator_multiplier * avarage_power_klystron * rf_frequency**2 * num_klystrons**np.log2(1-cls.learning_curve_klystrons)

    
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
            
            
            