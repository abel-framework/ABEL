from abel import Stage
import numpy as np
import copy
import warnings

class StageBasic(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1):
        
        super().__init__(length, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # ========== Apply plasma-density up ramp (demagnify beta function) ==========
        driver0 = self.driver_source.track()
        beam.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        

         # ========== Rotate the coordinate system of the beams ==========
        if self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0:
            drive_beam_ramped = copy.deepcopy(driver0)
            drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)

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
        beam.apply_betatron_motion(self.length, self.plasma_density, self.nom_energy_gain, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())

        
        # ========== Accelerate beam with homogeneous energy gain ==========
        beam.set_Es(beam.Es() + self.nom_energy_gain)
        
        # ========== Rotate the coordinate system of the beams back to original ==========
        if self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0:

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
                
        
        # ========== Apply plasma-density down ramp (magnify beta function) ==========
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        return super().track(beam, savedepth, runnable, verbose)
        
    