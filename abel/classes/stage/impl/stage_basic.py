from abel.classes.stage.stage import Stage
import scipy.constants as SI

SI.r_e = SI.physical_constants['classical electron radius'][0]

class StageBasic(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1, transformer_ratio=1):
        
        super().__init__(nom_accel_gradient, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)

        self.transformer_ratio = transformer_ratio
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # get the driver
        driver0 = self.driver_source.track()
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density()
        
        # apply plasma-density up ramp (demagnify beta function)
        beam.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # betatron oscillations
        beam.apply_betatron_motion(self.get_length(), self.plasma_density, self.get_nom_energy_gain(), x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())
        
        # accelerate beam
        beam.set_Es(beam.Es() + self.get_nom_energy_gain())
        
        # apply plasma-density down ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def optimize_plasma_density(self, source):
        
        # approximate extraction efficiency
        extraction_efficiency = (self.transformer_ratio/0.75)*abs(source.get_charge()/self.driver_source.get_charge())

        energy_density_z_extracted = abs(source.get_charge()*self.nom_accel_gradient)
        energy_density_z_wake = energy_density_z_extracted/extraction_efficiency
        norm_blowout_radius = ((32*SI.r_e/(SI.m_e*SI.c**2))*energy_density_z_wake)**(1/4)
        
        # optimal wakefield loading (finding the plasma density)
        norm_accel_gradient = 1/3 * (norm_blowout_radius)**1.15
        wavebreaking_field = self.nom_accel_gradient / norm_accel_gradient
        plasma_wavenumber = wavebreaking_field/(SI.m_e*SI.c**2/SI.e)
        self.plasma_density = plasma_wavenumber**2*SI.m_e*SI.c**2*SI.epsilon_0/SI.e**2
        
        
        
    