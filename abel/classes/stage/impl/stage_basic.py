from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
import numpy as np
import scipy.constants as SI
import copy


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

        print('driver_incoming.ux_offset()', driver_incoming.ux_offset())       ########## TODO: delete
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density()


        # ========== Rotate the coordinate system of the beams ==========
        # Perform beam rotations before calling on upramp tracking.
        if self.parent is None:  # Ensures that this is the main stage and not a ramp.

            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            drive_beam_rotated, beam_rotated = self.rotate_beam_coordinate_systems(driver_incoming, beam_incoming)

            print('drive_beam_rotated.ux_offset()', drive_beam_rotated.ux_offset())       ########## TODO: delete


        # ========== Prepare ramps ==========
        # If ramps exist, set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done.
        self._prepare_ramps()

        
        # ========== Apply plasma density up ramp (demagnify beta function) ==========
        if self.upramp is not None:  # if self has an upramp

            # Pass the drive beam and main beam to track_upramp() and get the ramped beams in return
            beam_ramped, drive_beam_ramped = self.track_upramp(beam_rotated, drive_beam_rotated)
        
        else:  # Do the following if there are no upramp (can be either a lone stage or the first upramp)
            if self.parent is None:  # Just a lone stage
                beam_ramped = copy.deepcopy(beam_rotated)
                drive_beam_ramped = copy.deepcopy(drive_beam_rotated)
            else:
                beam_ramped = copy.deepcopy(beam_incoming)
                drive_beam_ramped = copy.deepcopy(driver_incoming)

            if self.ramp_beta_mag is not None:

                beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)
                drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)

                ## NOTE: beam_ramped and drive_beam_ramped should not be changed after this line to check for continuity between ramps and stage. 

        beam = copy.deepcopy(beam_ramped)
        driver = copy.deepcopy(drive_beam_ramped)
                

        # ========== Betatron oscillations ==========
        deltaEs = np.full(len(beam.Es()), self.nom_energy_gain_flattop)  # Homogeneous energy gain for all macroparticles.
        if self.calc_evolution:
            _, evol = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset(), calc_evolution=self.calc_evolution)
            self.evolution.beam = evol
        else:
            beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset())


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
        # Perform un-rotation after track_downramp(). Also adds drift to the drive beam.
        if self.parent is None:  # Ensures that the un-rotation is only performed by the main stage and not by its ramps.

            # ========== Decelerate the driver with homogeneous energy loss ==========

            print('driver_outgoing.ux_offset() before driver_outgoing.set_Es:', driver_outgoing.ux_offset())       ########## TODO: delete

            driver_outgoing.set_Es(driver_outgoing.Es()*(1-self.depletion_efficiency))

            print('driver_outgoing.ux_offset() after driver_outgoing.set_Es:', driver_outgoing.ux_offset())       ########## TODO: delete
            
            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            driver_outgoing, beam_outgoing = self.undo_beam_coordinate_systems_rotation(driver_incoming, driver_outgoing, beam_outgoing)

            print('driver_outgoing.ux_offset() after undo_beam_coordinate_systems_rotation:', driver_outgoing.ux_offset())       ########## TODO: delete


        # ========== Bookkeeping ==========
        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        
        if self.test_beam_between_ramps:

            if self.parent is None:
                # Store beams for the main stage
                self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # Drive beam after the upramp
                                               beam_before_tracking=beam_ramped,  # Main beam after the upramp
                                               driver_outgoing=driver,  # Drive beam after tracking, before the downramp
                                               beam_outgoing=beam,  # Main beam after tracking, before the downramp
                                               driver_incoming=driver_incoming)  # The original drive beam before rotation and ramps
            else:
                # Store beams for the ramps
                self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # A deepcopy of the incoming drive beam before tracking.
                                               beam_before_tracking=beam_ramped,  # A deepcopy of the incoming main beam before tracking.
                                               driver_outgoing=driver_outgoing,  # Drive beam after tracking through the ramp (has not been un-rotated)
                                               beam_outgoing=beam_outgoing,  # Main beam after tracking through the ramp (has not been un-rotated)
                                               driver_incoming=None)  # Only the main stage needs to store the original drive beam

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
    def stage2ramp(self, ramp_plasma_density=None, ramp_length=None, transformer_ratio=1, depletion_efficiency=0.0, calc_evolution=False):
        """
        Used for copying a predefined stage's settings and configurations to set
        up flat ramps. Overloads the parent class' method.
    
        Parameters
        ----------
        ramp_plasma_density : [m^-3] float, optional
            Plasma density for the ramp.

        ramp_length : [m] float, optional
            Length of the ramp.

        ...
    
            
        Returns
        ----------
        stage_copy : ``Stage`` object
            A modified deep copy of the original stage.
        """

        stage_copy = super().stage2ramp(ramp_plasma_density, ramp_length)

        # Additional configurations 
        stage_copy.transformer_ratio = transformer_ratio
        stage_copy.depletion_efficiency = depletion_efficiency
        stage_copy.calc_evolution = calc_evolution

        return stage_copy

    
    