from abel import Stage

class StageBasic(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1):
        
        super().__init__(length, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # apply plasma-density up ramp (demagnify beta function)
        driver0 = self.driver_source.track()
        beam.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # betatron oscillations
        beam.apply_betatron_motion(self.length, self.plasma_density, self.nom_energy_gain, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())
        
        # accelerate beam
        beam.set_Es(beam.Es() + self.nom_energy_gain)
        
        # apply plasma-density down ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        return super().track(beam, savedepth, runnable, verbose)
        
    