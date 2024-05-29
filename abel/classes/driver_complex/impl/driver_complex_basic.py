from abc import abstractmethod
from matplotlib import patches
from abel import DriverComplex

class DriverComplexBasic(DriverComplex):
    
    def __init__(self, source=None, rf_accelerator=None, nom_energy=None, num_drivers=None):
        super().__init__(source=source, rf_accelerator=rf_accelerator, nom_energy=nom_energy, num_drivers=num_drivers)

    def track(self, _=None, savedepth=0, runnable=None, verbose=False, stage_number=None):
        
        # make the driver
        driver = self.source.track()

        # accelerate the driver
        driver = self.rf_accelerator.track(driver)
        
        return driver
    
    def get_length(self):
        return self.source.get_length() + self.rf_accelerator.get_length()
    
    def get_cost(self):
        return self.source.get_cost() + self.rf_accelerator.get_cost()
        