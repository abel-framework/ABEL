from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
import numpy as np
import scipy.constants as SI
import copy
import warnings

SI.r_e = SI.physical_constants['classical electron radius'][0]

class StageBasic(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, transformer_ratio=1, depletion_efficiency=0.75, calc_evolution=False, test_beam_between_ramps=False):
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.transformer_ratio = transformer_ratio
        self.depletion_efficiency = depletion_efficiency
        self.calc_evolution = calc_evolution
        self.test_beam_between_ramps = test_beam_between_ramps
        
    
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        print('test_beam_between_ramps:', self.test_beam_between_ramps)
        
        # get the driver
        driver_incoming = self.driver_source.track()
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density()


        # ========== Rotate the coordinate system of the beams ==========
        # Perform beam rotations before calling on upramp tracking.
        if self.parent is None:  # Ensures that this is the main stage and not a ramp.
            drive_beam_rotated = copy.deepcopy(driver_incoming)  # Make a deep copy to not affect the original drive beam.
            beam_rotated = copy.deepcopy(beam_incoming)

            # Check if the driver source of stage has angular jitter
            has_angular_offset = self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0

            # Perform rotation if there is angular jitter
            if has_angular_offset:

                driver_x_angle = driver_incoming.x_angle()
                driver_y_angle = driver_incoming.y_angle()
                
                beam0_x_angle = beam_incoming.x_angle()
                beam0_y_angle = beam_incoming.y_angle()

                # Calculate the angles that will be used to rotate the beams' frame
                rotation_angle_x, rotation_angle_y = drive_beam_rotated.beam_alignment_angles()
                rotation_angle_y = -rotation_angle_y  # Minus due to right hand rule.

                # The model currently does not support drive beam tilt not aligned with beam propagation, so need to first ensure that the drive beam is aligned to its own propagation direction. This is done using active transformation to rotate the beam around x- and y-axis
                drive_beam_rotated.add_pointing_tilts(rotation_angle_x, rotation_angle_y)

                # Use passive transformation to rotate the frame of the beams
                drive_beam_rotated.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y)  # Align the z-axis to the drive beam propagation.
                beam_rotated.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y)

                if np.abs( drive_beam_rotated.x_angle() ) > 5e-10:
                    driver_error_string = 'Drive beam may not have been accurately rotated in the zx-plane.\n' + 'driver_incoming x_angle before coordinate transformation: ' + str(driver_x_angle) + '\ndrive_beam_rotated x_angle after coordinate transformation: ' + str(drive_beam_rotated.x_angle())
                    warnings.warn(driver_error_string)

                if np.abs( drive_beam_rotated.y_angle() ) > 5e-10:
                    driver_error_string = 'Drive beam may not have been accurately rotated in the zy-plane.\n' + 'driver_incoming y_angle before coordinate transformation: ' + str(driver_y_angle) + '\ndrive_beam_rotated y_angle after coordinate transformation: ' + str(drive_beam_rotated.y_angle())
                    warnings.warn(driver_error_string)
        
                if np.abs( -(beam_rotated.x_angle() - beam0_x_angle) / rotation_angle_x - 1) > 1e-3:
                    warnings.warn('Main beam may not have been accurately rotated in the zx-plane.')
                    
                if np.abs( (beam_rotated.y_angle() - beam0_y_angle) / rotation_angle_y - 1) > 1e-3:
                    warnings.warn('Main beam may not have been accurately rotated in the zy-plane.')


        # ========== Prepare ramps ==========
        # If ramps exist, set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done.
        self._prepare_ramps()


        # ========== Apply plasma density up ramp (demagnify beta function) ==========
        if self.upramp is not None:  # if self has an upramp

            # Pass the drive beam and main beam to track_upramp() and get the ramped beams in return
            beam0, drive_beam_ramped = self.track_upramp(beam_rotated, drive_beam_rotated)
        
        else:  # Do the following if there are no upramp (can be either a lone stage or the first upramp)
            if self.parent is None:  # Just a lone stage
                beam0 = copy.deepcopy(beam_rotated)
                drive_beam_ramped = copy.deepcopy(drive_beam_rotated)
            else:
                beam0 = copy.deepcopy(beam_incoming)
                drive_beam_ramped = copy.deepcopy(driver_incoming)

            if self.ramp_beta_mag is not None:

                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)
                drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)

                ## NOTE: beam0 and drive_beam_ramped should not be changed after this line to check for continuity between ramps and stage.

        """         
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

        """  

        driver0 = drive_beam_ramped
        beam = copy.deepcopy(beam0)
        driver = copy.deepcopy(driver0)

                

        # ========== Betatron oscillations ==========
        deltaEs = np.full(len(beam.Es()), self.nom_energy_gain_flattop)  # Homogeneous energy gain for all macroparticles.
        if self.calc_evolution:
            _, evol = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset(), calc_evolution=self.calc_evolution)
            self.evolution.beam = evol
        else:
            beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())


        # ========== Accelerate beam with homogeneous energy gain ==========
        beam.set_Es(beam.Es() + self.nom_energy_gain_flattop)
        

        # ==========  Apply plasma density down ramp (magnify beta function) ==========
        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(copy.deepcopy(beam), copy.deepcopy(driver))

        else:  # Do the following if there are no downramp. 
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:  # Do the following before rotating back to original frame.

                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)


        # ========== Rotate the coordinate system of the beams back to original ==========
        # Perform un-rotation after track_downramp()
        if self.parent is None:  # Ensures that the un-rotation is only performed by the main stage and not its ramps.

            if self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0:

                # Angles of beam before rotating back to original coordinate system
                beam_x_angle = beam_outgoing.x_angle()
                beam_y_angle = beam_outgoing.y_angle()

                driver_outgoing.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y, invert=True) # TODO: check if this roates back correctly.
                beam_outgoing.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y, invert=True)
                
                # Add drifts to the beam
                x_drift = self.length_flattop * np.tan(driver_x_angle)
                y_drift = self.length_flattop * np.tan(driver_y_angle)
                xs = beam_outgoing.xs()
                ys = beam_outgoing.ys()
                beam_outgoing.set_xs(xs + x_drift)
                beam_outgoing.set_ys(ys + y_drift)
                xs_driver = driver_outgoing.xs()
                ys_driver = driver_outgoing.ys()
                driver_outgoing.set_xs(xs_driver + x_drift)
                driver_outgoing.set_ys(ys_driver + y_drift)
                
                #drive_beam_ramped.yx_rotate_coord_sys(-rotation_angle_x, -rotation_angle_y)
            
                if driver_incoming.x_angle() != 0 and np.abs( (beam_outgoing.x_angle() - beam_x_angle) / rotation_angle_x - 1) > 1e-3:
                    warnings.warn('Main beam may not have been accurately rotated in the xz-plane.')
                    
                if driver_incoming.y_angle() != 0 and np.abs( -(beam_outgoing.y_angle() - beam_y_angle) / rotation_angle_y - 1) > 1e-3:
                    warnings.warn('Main beam may not have been accurately rotated in the yz-plane.')


        """
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
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver) """
        


        # ========== Decelerate the driver with homogeneous energy loss ==========
        driver_outgoing.set_Es(driver_outgoing.Es()*(1-self.depletion_efficiency))


        # ========== Bookkeeping ==========
        # Store beams for comparison between ramps and its parent
        
        if self.test_beam_between_ramps:
            print('test_beam_between_ramps:', self.test_beam_between_ramps)
            if self.parent is None:

                # The original drive beam before roation and ramps
                self.driver_incoming = driver_incoming

                # The outgoing beams for the main stage need to be recorded before potential rotation for correct comparison with its ramps.
                self.beam_out = beam
                self.driver_out = driver

                # # Store beams for the main stage
                # self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # Drive beam after the upramp
                #                                beam_before_tracking=beam0,  # Main beam after the upramp
                #                                driver_outgoing=driver,  # Drive beam after tracking, before the downramp
                #                                beam_outgoing=beam,  # Main beam after tracking, before the downramp
                #                                driver_incoming=driver_incoming)  # The original drive beam before rotation and ramps
            else:
                # Ramps record the final beams as their output beams, as they should not perform any rotation between instability tracking and this line.
                self.beam_out = beam_outgoing
                self.driver_out = driver_outgoing

                # # Store beams for the ramps
                # self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # A deepcopy of the incoming drive beam before tracking.
                #                                beam_before_tracking=beam0,  # A deepcopy of the incoming main beam before tracking.
                #                                driver_outgoing=driver_outgoing,  # Drive beam after tracking through the ramp (has not been un-rotated)
                #                                beam_outgoing=beam_outgoing,  # Main beam after tracking through the ramp (has not been un-rotated)
                #                                driver_incoming=None)  # Only the main stage needs to store the original drive beam

            self.driver_in = drive_beam_ramped  # Drive beam before instability tracking
            self.beam_in = beam0  # Main beam before instability tracking


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
     
        
    