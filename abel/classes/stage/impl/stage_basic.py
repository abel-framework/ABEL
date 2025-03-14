from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
import numpy as np
import scipy.constants as SI
import copy
import warnings

SI.r_e = SI.physical_constants['classical electron radius'][0]

class StageBasic(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, transformer_ratio=1, depletion_efficiency=0.75):
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.transformer_ratio = transformer_ratio
        self.depletion_efficiency = depletion_efficiency
        
    
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):
        
        # get the driver
        driver_incoming = self.driver_source.track()
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density()

        # Set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done
        self._prepare_ramps()

        # plasma-density ramps (de-magnify beta function)
        if self.upramp is not None:
            beam0, driver0 = self.track_upramp(beam_incoming, driver_incoming)
        else:
            beam0 = copy.deepcopy(beam_incoming)
            driver0 = copy.deepcopy(driver_incoming)
            if self.ramp_beta_mag is not None:
                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
                driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)   
                
        # apply plasma-density up ramp (demagnify beta function)
        beam = copy.deepcopy(beam0)
        
        # non-evolving driver
        driver = copy.deepcopy(driver0)

        # ========== Rotate the coordinate system of the beams ==========
        if isinstance(self.driver_source, Source) and (self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0):
            drive_beam_ramped = copy.deepcopy(driver0)
            #drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)

            driver_x_angle = drive_beam_ramped.x_angle()
            driver_y_angle = drive_beam_ramped.y_angle()
            
            beam0_x_angle = beam.x_angle()
            beam0_y_angle = beam.y_angle()


            # Calculate the angles that will be used to rotate the beams' frame
            rotation_angle_x, rotation_angle_y = drive_beam_ramped.beam_alignment_angles()
            rotation_angle_y = -rotation_angle_y  # Minus due to right hand rule.

            # The model currently does not support beam tilt not aligned with beam propagation, so that active transformation is used to rotate the beam around x- and y-axis and align it to its own prapagation direction. 
            drive_beam_ramped.add_pointing_tilts(rotation_angle_x, rotation_angle_y)

            # Use passive transformation to rotate the frame of the beams
            drive_beam_ramped.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y)  # Align the z-axis to the drive beam propagation.
            beam.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y)
            
            if np.abs( drive_beam_ramped.x_angle() ) > 5e-10:
                driver_error_string = 'Drive beam may not have been accurately rotated in the zx-plane.\n' + 'drive_beam_ramped x_angle before coordinate transformation: ' + str(driver_x_angle) + '\ndrive_beam_ramped x_angle after coordinate transformation: ' + str(drive_beam_ramped.x_angle())
                warnings.warn(driver_error_string)

            if np.abs( drive_beam_ramped.y_angle() ) > 5e-10:
                driver_error_string = 'Drive beam may not have been accurately rotated in the zy-plane.\n' + 'drive_beam_ramped y_angle before coordinate transformation: ' + str(driver_y_angle) + '\ndrive_beam_ramped y_angle after coordinate transformation: ' + str(drive_beam_ramped.y_angle())
                warnings.warn(driver_error_string)

    
            if np.abs( -(beam.x_angle() - beam0_x_angle) / rotation_angle_x - 1) > 1e-3:
                warnings.warn('Main beam may not have been accurately rotated in the zx-plane.')
                
            if np.abs( (beam.y_angle() - beam0_y_angle) / rotation_angle_y - 1) > 1e-3:
                warnings.warn('Main beam may not have been accurately rotated in the zy-plane.')
                

        # ========== Betatron oscillations ==========
        beam.apply_betatron_motion(self.length_flattop, self.plasma_density, self.nom_energy_gain_flattop, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())


        # ========== Accelerate beam with homogeneous energy gain ==========
        beam.set_Es(beam.Es() + self.nom_energy_gain_flattop)

        # ========== Rotate the coordinate system of the beams back to original ==========
        if isinstance(self.driver_source, Source) and (self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0):

            # Angles of beam before rotating back to original coordinate system
            beam_x_angle = beam.x_angle()
            beam_y_angle = beam.y_angle()
            
            beam.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y, invert=True)

            # Add drifts to the beam
            x_drift = self.length * np.tan(driver_x_angle)
            y_drift = self.length * np.tan(driver_y_angle)
            
            xs = beam.xs()
            ys = beam.ys()
            
            beam.set_xs(xs + x_drift)
            beam.set_ys(ys + y_drift)

            
            if driver0.x_angle() != 0 and np.abs( (beam.x_angle() - beam_x_angle) / rotation_angle_x - 1) > 1e-3:
                warnings.warn('Main beam may not have been accurately rotated in the xz-plane.')
                
            if driver0.y_angle() != 0 and np.abs( -(beam.y_angle() - beam_y_angle) / rotation_angle_y - 1) > 1e-3:
                warnings.warn('Main beam may not have been accurately rotated in the yz-plane.')


        # apply plasma-density down ramp (magnify beta function)
        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)
        else:
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:
                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)

        # ========== Decelerate the driver with homogeneous energy loss ==========
        driver_outgoing.set_Es(driver_outgoing.Es()*(1-self.depletion_efficiency))

        # calculate efficiency
        self.calculate_efficiency(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)

        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)

    
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
     
        
    