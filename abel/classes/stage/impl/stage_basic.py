from abel.classes.stage.stage import Stage, StageError
from abel.classes.source.impl.source_capsule import SourceCapsule
import numpy as np
import scipy.constants as SI
import copy, warnings


SI.r_e = SI.physical_constants['classical electron radius'][0]

class StageBasic(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, transformer_ratio=1, depletion_efficiency=0.75, calc_evolution=False, test_beam_between_ramps=False):
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.transformer_ratio = transformer_ratio
        self.depletion_efficiency = depletion_efficiency
        self.calc_evolution = calc_evolution
        self.test_beam_between_ramps = test_beam_between_ramps
        
    
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):
        
        # get the driver
        driver_incoming = self.driver_source.track()

        if self.test_beam_between_ramps:
            original_driver = copy.deepcopy(driver_incoming)
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density()


        # ========== Rotate the coordinate system of the beams ==========
        # Perform beam rotations before calling on upramp tracking.
        if self.parent is None:  # Ensures that this is the main stage and not a ramp.

            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            drive_beam_rotated, beam_rotated = self.rotate_beam_coordinate_systems(driver_incoming, beam_incoming)


        # ========== Prepare ramps ==========
        # If ramps exist, set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done.
        self._prepare_ramps()

        
        # ========== Apply plasma density up ramp (demagnify beta function) ==========
        if self.upramp is not None:  # if self has an upramp

            if self.upramp.nom_energy_gain is None or self.upramp.nom_energy_gain == 0:
                ramp_energy_gain = self.upramp.length * self.nom_accel_gradient_flattop * 0.05*np.sqrt(self.upramp.plasma_density/self.plasma_density)
                warnings.warn(f"Upramp nominal energy gain for StageBasic must be non-zero. Setting this to {ramp_energy_gain/1e9  :.3f} GeV.")
                self.upramp.nom_energy_gain = ramp_energy_gain

            # Pass the drive beam and main beam to track_upramp() and get the ramped beams in return
            beam_ramped, drive_beam_ramped = self.track_upramp(beam_rotated, drive_beam_rotated)
        
        else:  # Do the following if there are no upramp (a lone stage)
            beam_ramped = copy.deepcopy(beam_rotated)
            drive_beam_ramped = copy.deepcopy(drive_beam_rotated)
            
            if self.ramp_beta_mag is not None:
                beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)
                drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)

        # Save beams to check for consistency between ramps and stage
        if self.test_beam_between_ramps:
            stage_beam_in = copy.deepcopy(beam_ramped)
            stage_driver_in = copy.deepcopy(drive_beam_ramped)

        
        # ========== Main tracking sequence ==========
        beam, driver = self.main_tracking_procedure(beam_ramped, drive_beam_ramped)


        # ==========  Apply plasma density down ramp (magnify beta function) ==========
        if self.downramp is not None:

            if self.downramp.nom_energy_gain is None or self.downramp.nom_energy_gain == 0:
                ramp_energy_gain = self.downramp.length * self.nom_accel_gradient_flattop * 0.05*np.sqrt(self.downramp.plasma_density/self.plasma_density)
                warnings.warn(f"Downramp nominal energy gain for StageBasic must be non-zero. Setting this to {ramp_energy_gain/1e9  :.3f} GeV.")

                self.downramp.nom_energy_gain = ramp_energy_gain

            # TODO: Temporary "drive beam evolution": Magnify the driver
            # Needs to be performed before self.track_downramp().
            if self.downramp.ramp_beta_mag is not None:
                driver.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)

            # Save beams to probe for consistency between ramps and stage
            if self.test_beam_between_ramps:
                stage_beam_out = copy.deepcopy(beam)
                stage_driver_out = copy.deepcopy(driver)

            beam_outgoing, driver_outgoing = self.track_downramp(copy.deepcopy(beam), copy.deepcopy(driver))

        else:  # Do the following if there are no downramp. 
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:  # Do the following before rotating back to original frame.

                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)

                # Save beams to probe for consistency between ramps and stage
                if self.test_beam_between_ramps:
                    stage_beam_out = copy.deepcopy(beam_outgoing)
                    stage_driver_out = copy.deepcopy(driver_outgoing)


        # ========== Rotate the coordinate system of the beams back to original ==========
        # Perform un-rotation after track_downramp(). Also adds drift to the drive beam.
        if self.parent is None:  # Ensures that the un-rotation is only performed by the main stage and not by its ramps.
            
            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            driver_outgoing, beam_outgoing = self.undo_beam_coordinate_systems_rotation(driver_incoming, driver_outgoing, beam_outgoing)


        # ========== Bookkeeping ==========
        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        
        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        if self.test_beam_between_ramps:
            # Store beams for the main stage
            self.store_beams_between_ramps(driver_before_tracking=stage_driver_in,  # Drive beam after the upramp, before tracking
                                            beam_before_tracking=stage_beam_in,  # Main beam after the upramp, before tracking
                                            driver_outgoing=stage_driver_out,  # Drive beam after tracking, before the downramp
                                            beam_outgoing=stage_beam_out,  # Main beam after tracking, before the downramp
                                            driver_incoming=original_driver)  # The original drive beam before rotation and ramps

        # calculate efficiency
        self.calculate_efficiency(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)

        # Copy meta data from input beam_outgoing (will be iterated by super)
        beam_outgoing.trackable_number = beam_incoming.trackable_number
        beam_outgoing.stage_number = beam_incoming.stage_number
        beam_outgoing.location = beam_incoming.location

        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
        

    # ==================================================
    def main_tracking_procedure(self, beam_ramped, drive_beam_ramped):
        """
        Prepares and performs the main reduced model beam tracking.
        

        Parameters
        ----------
        beam_ramped : ABEL ``Beam`` object
            Main beam.

        drive_beam_ramped : ABEL ``Beam`` object
            Drive beam.

            
        Returns
        ----------
        beam : ABEL ``Beam`` object
            Main beam after tracking.

        drive_beam : ABEL ``Beam`` object
            Drive beam after tracking.
        """

        beam = copy.deepcopy(beam_ramped)
        drive_beam = copy.deepcopy(drive_beam_ramped)

        # Betatron oscillations
        deltaEs = np.full(len(beam.Es()), self.nom_energy_gain_flattop)  # Homogeneous energy gain for all macroparticles.
        if self.calc_evolution:
            _, evol = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset(), calc_evolution=self.calc_evolution)
            self.evolution.beam = evol
        else:
            beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset())

        # Accelerate beam with homogeneous energy gain
        beam.set_Es(beam.Es() + self.nom_energy_gain_flattop)

        # Decelerate the driver with homogeneous energy loss
        drive_beam.set_Es(drive_beam_ramped.Es()*(1-self.depletion_efficiency))

        # Update the beam locations
        drive_beam.location = drive_beam_ramped.location + self.length_flattop
        beam.location = beam_ramped.location + self.length_flattop

        return beam, drive_beam
    

    # ==================================================
    def track_upramp(self, beam0, driver0):
        """
        Called by a stage to perform upramp tracking.
    
        
        Parameters
        ----------
        driver0 : ABEL ``Beam`` object
            Drive beam.

        beam0 : ABEL ``Beam`` object
            Main beam.
    
            
        Returns
        ----------
        beam : ABEL ``Beam`` object
            Main beam after tracking.

        driver : ABEL ``Beam`` object
            Drive beam after tracking.
        """

        from abel.classes.stage.impl.plasma_ramp import PlasmaRamp

        # Save beams to check for consistency between ramps and stage
        if self.test_beam_between_ramps:
            ramp_beam_in = copy.deepcopy(beam0)
            ramp_driver_in = copy.deepcopy(driver0)

        # Convert PlasmaRamp to a StagePrtclWakeInstability
        if type(self.upramp) is PlasmaRamp:

            upramp = self.convert_PlasmaRamp(self.upramp)
            if type(upramp) is not StageBasic:
                raise TypeError('upramp is not a StageBasic.')

        elif type(self.upramp) is Stage:
            upramp = self.upramp  # Allow for other types of ramps
        else:
            raise StageError('Ramp is not an instance of Stage class.')
        
        if upramp.plasma_density is None:
            raise ValueError('Upramp plasma density is invalid.')
        if upramp.nom_energy is None:
            raise ValueError('Upramp nominal enegy is invalid.')
        if upramp.nom_energy_flattop is None:
            raise ValueError('Upramp flattop nominal energy is invalid.')
        if upramp.length is None:
            raise ValueError('Upramp length is invalid.')
        if upramp.length_flattop is None:
            raise ValueError('Upramp flattop length is invalid.')
        if upramp.nom_energy_gain is None:
            raise ValueError('Upramp nominal enegy gain is invalid.')
        if upramp.nom_energy_gain_flattop is None:
            raise ValueError('Upramp flattop nominal energy gain is invalid.')
        if upramp.nom_accel_gradient is None:
            raise ValueError('Upramp nominal acceleration gradient is invalid.')
        if upramp.nom_accel_gradient_flattop is None:
            raise ValueError('Upramp flattop nominal acceleration gradient is invalid.')

        # Set driver
        upramp.driver_source = SourceCapsule(beam=driver0)


        # ========== Main tracking sequence ==========
        beam, driver = upramp.main_tracking_procedure(beam0, driver0)


        # ========== Bookkeeping ==========
        # calculate efficiency
        self.upramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.upramp.calculate_beam_current(beam0, driver0, beam, driver)

        # TODO: Temporary "drive beam evolution": Demagnify the driver
        # Needs to be performed before self.upramp.store_beams_between_ramps().
        if self.ramp_beta_mag is not None:  
            driver.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver)

        # Save beams to check for consistency between ramps and stage
        if self.test_beam_between_ramps:
            ramp_beam_out = copy.deepcopy(beam)
            ramp_driver_out = copy.deepcopy(driver)

        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        if self.test_beam_between_ramps:
            self.upramp.store_beams_between_ramps(driver_before_tracking=ramp_driver_in,  # A deepcopy of the incoming drive beam before tracking.
                                            beam_before_tracking=ramp_beam_in,  # A deepcopy of the incoming main beam before tracking.
                                            driver_outgoing=ramp_driver_out,  # Drive beam after tracking through the ramp (has not been un-rotated)
                                            beam_outgoing=ramp_beam_out,  # Main beam after tracking through the ramp (has not been un-rotated)
                                            driver_incoming=None)  # Only the main stage needs to store the original drive beam
            
        return beam, driver
    

    # ==================================================
    def track_downramp(self, beam0, driver0):
        """
        Called by a stage to perform downramp tracking.
    
        
        Parameters
        ----------
        driver0 : ABEL ``Beam`` object
            Drive beam.

        beam0 : ABEL ``Beam`` object
            Main beam.
    
            
        Returns
        ----------
        beam : ABEL ``Beam`` object
            Main beam after tracking.

        driver : ABEL ``Beam`` object
            Drive beam after tracking.
        """

        from abel.classes.stage.impl.plasma_ramp import PlasmaRamp

        # Save beams to check for consistency between ramps and stage
        if self.test_beam_between_ramps:
            ramp_beam_in = copy.deepcopy(beam0)
            ramp_driver_in = copy.deepcopy(driver0)

        # Convert PlasmaRamp to a StagePrtclWakeInstability
        if type(self.downramp) is PlasmaRamp:

            downramp = self.convert_PlasmaRamp(self.downramp)
            if type(downramp) is not StageBasic:
                raise TypeError('downramp is not a StageBasic.')

        elif type(self.downramp) is Stage:
            downramp = self.downramp  # Allow for other types of ramps
        else:
            raise StageError('Ramp is not an instance of Stage class.')
        
        if downramp.plasma_density is None:
            raise ValueError('Downramp plasma density is invalid.')
        if downramp.nom_energy is None:
            raise ValueError('Downramp nominal enegy is invalid.')
        if downramp.nom_energy_flattop is None:
            raise ValueError('Downramp flattop nominal energy is invalid.')
        if downramp.length is None:
            raise ValueError('Downramp length is invalid.')
        if downramp.length_flattop is None:
            raise ValueError('Downramp flattop length is invalid.')
        if downramp.nom_energy_gain is None:
            raise ValueError('Downramp nominal enegy gain is invalid.')
        if downramp.nom_energy_gain_flattop is None:
            raise ValueError('Downramp flattop nominal energy gain is invalid.')
        if downramp.nom_accel_gradient is None:
            raise ValueError('Downramp nominal acceleration gradient is invalid.')
        if downramp.nom_accel_gradient_flattop is None:
            raise ValueError('Downramp flattop nominal acceleration gradient is invalid.')

        # Set driver
        downramp.driver_source = SourceCapsule(beam=driver0)


        # ========== Main tracking sequence ==========
        beam, driver = downramp.main_tracking_procedure(beam0, driver0)


        # ========== Bookkeeping ==========
        # calculate efficiency
        self.downramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.downramp.calculate_beam_current(beam0, driver0, beam, driver)

        # Save beams to check for consistency between ramps and stage
        if self.test_beam_between_ramps:
            ramp_beam_out = copy.deepcopy(beam)
            ramp_driver_out = copy.deepcopy(driver)

        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        if self.test_beam_between_ramps:
            self.downramp.store_beams_between_ramps(driver_before_tracking=ramp_driver_in,  # A deepcopy of the incoming drive beam before tracking.
                                            beam_before_tracking=ramp_beam_in,  # A deepcopy of the incoming main beam before tracking.
                                            driver_outgoing=ramp_driver_out,  # Drive beam after tracking through the ramp (has not been un-rotated)
                                            beam_outgoing=ramp_beam_out,  # Main beam after tracking through the ramp (has not been un-rotated)
                                            driver_incoming=None)  # Only the main stage needs to store the original drive beam
            
        return beam, driver


    # ==================================================
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
     
        
    # ==================================================
    def copy_config2blank_stage(self, transformer_ratio=None, depletion_efficiency=None, calc_evolution=None):
        """
        Makes a deepcopy of the stage to copy the configurations and settings,
        but most of the parameters in the deepcopy are set to ``None``.
    
        Parameters
        ----------
        ...

        calc_evolution : bool, optional
            Flag for recording the beam parameter evolution. Default set to the
            same value as ``self``.
            
        Returns
        ----------
        stage_copy : ``Stage`` object
            A modified deep copy of the original stage. 
            ``stage_copy.plasma_density``, ``stage_copy.length``, 
            ``stage_copy.length_flattop``, ``stage_copy.nom_energy_gain``, 
            ``stage_copy.nom_energy_gain_flattop``, 
            ``stage_copy.nom_accel_gradient``, 
            ``stage_copy.nom_accel_gradient_flattop``, 
            ``stage_copy.driver_source`` and its ramps are all set to ``None``.
        """

        stage_copy = super().copy_config2blank_stage()

        # Additional configurations
        if transformer_ratio is not None:
            stage_copy.transformer_ratio = transformer_ratio
        if depletion_efficiency is not None:
            stage_copy.depletion_efficiency = depletion_efficiency
        if calc_evolution is not None:
            stage_copy.calc_evolution = calc_evolution

        return stage_copy

    
    
###################################################
# Custom formatting that omits the line of source code
def custom_formatwarning(msg, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {msg}\n"

# Tell Python to use custom_formatwarning() instead of the default warnings.formatwarning(), so any subsequent warnings will follow this formatting
warnings.formatwarning = custom_formatwarning
    