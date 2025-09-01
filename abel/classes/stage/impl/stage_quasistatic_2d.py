from abel.CONFIG import CONFIG
from abel.classes.stage.stage import Stage
import numpy as np
import scipy.constants as SI
from abel.utilities.plasma_physics import *
import os, shutil, uuid, copy, sys
from abel.physics_models.particles_transverse_wake_instability import transverse_wake_instability_particles

class StageQuasistatic2d(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, enable_radiation_reaction=False, probe_evolution=False, store_beams_for_tests=False):
        
        super().__init__(nom_accel_gradient, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
        # physics flags
        self.enable_radiation_reaction = enable_radiation_reaction

        # simulation flags
        self.probe_evolution = probe_evolution
        self.store_beams_for_tests = store_beams_for_tests
    
        
    # ==================================================
    # track the particles through
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        import warnings
        
        # suppress numba warnings from Ocelot
        warnings.simplefilter('ignore', category=RuntimeWarning)
        
        # make driver
        driver_incoming = self.driver_source.track()
        original_driver = copy.deepcopy(driver_incoming)


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
        if self.upramp is not None:
            beam0, driver0 = self.track_upramp(beam_rotated, drive_beam_rotated)
        else:
            beam0 = beam_rotated
            driver0 = drive_beam_rotated
            # if self.ramp_beta_mag is not None:
            #     beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
            #     driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)

        
        # ========== Perform tracking in the flattop stage ==========
        beam, driver = self.main_tracking_procedure(beam0, driver0)

        
        # ==========  Apply plasma density down ramp (magnify beta function) ==========
        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)
        else:
            beam_outgoing = beam
            driver_outgoing = driver
            # if self.ramp_beta_mag is not None:
            #     beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
            #     driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
        

        # ========== Rotate the coordinate system of the beams back to original ==========
        # Perform un-rotation after track_downramp(). Also adds drift to the drive beam.
        if self.parent is None:  # Ensures that the un-rotation is only performed by the main stage and not by its ramps.
            
            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            driver_outgoing, beam_outgoing = self.undo_beam_coordinate_systems_rotation(original_driver, driver_outgoing, beam_outgoing)


        # ========== Bookkeeping ==========
        # Store beams for tests
        if self.store_beams_for_tests:
            # The original drive beam before rotation and ramps
            self.driver_incoming = original_driver
            
        # copy meta data from input beam (will be iterated by super)
        beam_outgoing.trackable_number = beam_incoming.trackable_number
        beam_outgoing.stage_number = beam_incoming.stage_number
        beam_outgoing.location = beam_incoming.location
        
        # clean nan particles and extreme outliers
        beam.remove_nans()
        beam.remove_halo_particles()
        
        # calculate efficiency
        self.calculate_efficiency(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
        

    # ==================================================
    def main_tracking_procedure(self, beam0, driver0):
        """
        Prepares and performs the beam tracking using the physics models of the 
        stage.
        

        Parameters
        ----------
        beam0 : ABEL ``Beam`` object
            Main beam.

        driver0 : ABEL ``Beam`` object
            Drive beam.

            
        Returns
        ----------
        beam : ABEL ``Beam`` object
            Main beam after tracking.

        driver : ABEL ``Beam`` object
            Drive beam after tracking.
        """

        import wake_t
        from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam

        # make copy of the beams to update later
        beam = copy.deepcopy(beam0)
        driver = copy.deepcopy(driver0)
        driver0_location = driver0.location
        beam0_location = beam0.location

        # ========== Wake-T simulation and extraction ==========
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')
        
        # create plasma stage
        box_min_z = beam0.z_offset(clean=True) - 5 * beam0.bunch_length(clean=True) - 0.25/k_p(self.plasma_density)
        box_max_z = driver0.z_offset(clean=True) + 4 * driver0.bunch_length(clean=True)
        box_size_r = 3 * blowout_radius(self.plasma_density, driver0.peak_current())
        k_beta_driver = k_p(self.plasma_density)/np.sqrt(2*driver0.gamma())
        k_beta_beam = k_p(self.plasma_density)/np.sqrt(2*beam0.gamma())
        lambda_betatron_min = 2*np.pi/max(k_beta_beam, k_beta_driver)
        lambda_betatron_max = 2*np.pi/min(k_beta_beam, k_beta_driver)
        dz = lambda_betatron_min/10
        
        # need to make sufficiently many steps
        n_out = max(1, round(lambda_betatron_max/lambda_betatron_min/2))
        plasma = wake_t.PlasmaStage(length=dz, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=512, n_xi=512, dz_fields=dz, ppc=4)
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
                    
        # perform tracking
        with suppress_stdout():
            bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder)
        
        # convert back to ABEL beams
        beam_waket = wake_t_bunch2beam(bunches[1][-1])
        driver_waket = wake_t_bunch2beam(bunches[0][-1])
        
        # save evolution of the beam and driver
        if self.probe_evolution:
            self._driver_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[0])
            self._beam_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[1])

        # Store wakefield and beam quantities
        self.__save_initial_step(tmpfolder)

        # remove temporary directory
        shutil.rmtree(tmpfolder)
        

        # ========== Main tracking sequence ==========
        # calculate energy gain
        delta_Es = self.length_flattop*(beam_waket.Es() - beam.Es())/dz

        # find driver offset (to shift the beam relative) and apply betatron motion
        output = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, delta_Es, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset(), radiation_reaction=self.enable_radiation_reaction, probe_evolution=self.probe_evolution)
        if self.probe_evolution:
            Es_final, self.evolution.beam = output
        else:
            Es_final = output
        
        # accelerate beam (and remove nans)
        beam.set_Es(Es_final)
            
        # decelerate driver (and remove nans)
        delta_Es_driver = self.length_flattop*(driver0.Es()-driver_waket.Es())/dz
        driver.apply_betatron_damping(delta_Es_driver)
        driver.set_Es(driver0.Es() + delta_Es_driver)

        # Update the beam locations
        driver.location = driver0_location + self.length_flattop
        beam.location = beam0_location + self.length_flattop
        
        return beam, driver
    

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

        from abel.classes.stage.stage import Stage, PlasmaRamp
        from abel.classes.source.impl.source_capsule import SourceCapsule

        # # Save beams to check for consistency between ramps and stage
        # if self.test_beam_between_ramps:
        #     ramp_beam_in = copy.deepcopy(beam0)
        #     ramp_driver_in = copy.deepcopy(driver0)

        # Convert PlasmaRamp to a StageQuasistatic2d
        if type(self.upramp) is PlasmaRamp:

            upramp = self.convert_PlasmaRamp(self.upramp)
            if type(upramp) is not StageQuasistatic2d:
                raise TypeError('upramp is not a StageQuasistatic2d.')

        elif isinstance(self.upramp, Stage):
            upramp = self.upramp  # Allow for other types of ramps
        
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
        # clean nan particles and extreme outliers
        beam.remove_nans()
        beam.remove_halo_particles()

        # calculate efficiency
        self.upramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.upramp.calculate_beam_current(beam0, driver0, beam, driver)

        # # TODO: Temporary "drive beam evolution": Demagnify the driver
        # # Needs to be performed before self.upramp.store_beams_between_ramps().
        # driver.magnify_beta_function(1/self.upramp.ramp_beta_mag, axis_defining_beam=driver)

        # # Save beams to check for consistency between ramps and stage
        # if self.test_beam_between_ramps:
        #     ramp_beam_out = copy.deepcopy(beam)
        #     ramp_driver_out = copy.deepcopy(driver)

        # # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        # if self.test_beam_between_ramps:
        #     self.upramp.store_beams_between_ramps(driver_before_tracking=ramp_driver_in,  # A deepcopy of the incoming drive beam before tracking.
        #                                     beam_before_tracking=ramp_beam_in,  # A deepcopy of the incoming main beam before tracking.
        #                                     driver_outgoing=ramp_driver_out,  # Drive beam after tracking through the ramp (has not been un-rotated)
        #                                     beam_outgoing=ramp_beam_out,  # Main beam after tracking through the ramp (has not been un-rotated)
        #                                     driver_incoming=None)  # Only the main stage needs to store the original drive beam
            
        # Save parameter evolution to the ramp
        if self.probe_evolution:
            self.upramp.evolution = upramp.evolution  # TODO: save to self instead, but need to change stage diagnostics and how this is saved in self.main_tracking_procedure() first.
            
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

        from abel.classes.stage.stage import Stage, PlasmaRamp
        from abel.classes.source.impl.source_capsule import SourceCapsule

        # # Save beams to check for consistency between ramps and stage
        # if self.test_beam_between_ramps:
        #     ramp_beam_in = copy.deepcopy(beam0)
        #     ramp_driver_in = copy.deepcopy(driver0)

        # Convert PlasmaRamp to a StageQuasistatic2d
        if type(self.downramp) is PlasmaRamp:

            downramp = self.convert_PlasmaRamp(self.downramp)
            if type(downramp) is not StageQuasistatic2d:
                raise TypeError('downramp is not a StageQuasistatic2d.')

        elif isinstance(self.downramp, Stage):
            downramp = self.downramp  # Allow for other types of ramps
        
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
        # clean nan particles and extreme outliers
        beam.remove_nans()
        beam.remove_halo_particles()

        # calculate efficiency
        self.downramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.downramp.calculate_beam_current(beam0, driver0, beam, driver)

        # # Save beams to check for consistency between ramps and stage
        # if self.test_beam_between_ramps:
        #     ramp_beam_out = copy.deepcopy(beam)
        #     ramp_driver_out = copy.deepcopy(driver)

        # # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        # if self.test_beam_between_ramps:
        #     self.downramp.store_beams_between_ramps(driver_before_tracking=ramp_driver_in,  # A deepcopy of the incoming drive beam before tracking.
        #                                     beam_before_tracking=ramp_beam_in,  # A deepcopy of the incoming main beam before tracking.
        #                                     driver_outgoing=ramp_driver_out,  # Drive beam after tracking through the ramp (has not been un-rotated)
        #                                     beam_outgoing=ramp_beam_out,  # Main beam after tracking through the ramp (has not been un-rotated)
        #                                     driver_incoming=None)  # Only the main stage needs to store the original drive beam
            
        # Save parameter evolution to the ramp
        if self.probe_evolution:
            self.downramp.evolution = downramp.evolution  # TODO: save to self instead, but need to change stage diagnostics and how this is saved in self.main_tracking_procedure() first.
            
        return beam, driver
    

    # ==================================================
    def __save_initial_step(self, tmpfolder):
    #def __save_initial_step(self, Ez0_axial, zs_Ez0, rho0, metadata_rho0, driver0, beam0):
        """
        Saves initial electric field, plasma and beam quantities to the stage.
        """

        from openpmd_viewer import OpenPMDTimeSeries

        # extract wakefield info
        ts = OpenPMDTimeSeries(tmpfolder+'hdf5/')
        Ez, metadata = ts.get_field(field='E', coord='z', iteration=min(ts.iterations))
        self.initial.plasma.wakefield.onaxis.zs = metadata.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez[round(len(metadata.r)/2),:].flatten()
        
        # extract initial plasma density
        rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
        self.initial.plasma.density.extent = metadata0_plasma.imshow_extent
        self.initial.plasma.density.rho = -(rho0_plasma/SI.e)
        
        # extract initial beam density
        data0_beam = ts.get_particle(species='beam', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        data0_driver = ts.get_particle(species='driver', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        extent0 = metadata0_plasma.imshow_extent
        Nbins0 = self.initial.plasma.density.rho.shape
        dr0 = (extent0[3]-extent0[2])/Nbins0[0]
        dz0 = (extent0[1]-extent0[0])/Nbins0[1]
        mask0_beam = np.logical_and(data0_beam[1] < dr0/2, data0_beam[1] > -dr0/2)
        jz0_beam, _, _ = np.histogram2d(data0_beam[0][mask0_beam], data0_beam[2][mask0_beam], weights=data0_beam[3][mask0_beam], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        mask0_driver = np.logical_and(data0_driver[1] < dr0/2, data0_driver[1] > -dr0/2)
        jz0_driver, _, _ = np.histogram2d(data0_driver[0][mask0_driver], data0_driver[2][mask0_driver], weights=data0_driver[3][mask0_driver], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        self.initial.beam.density.extent = metadata0_plasma.imshow_extent
        self.initial.beam.density.rho = (jz0_beam+jz0_driver)/(dr0*dr0*dz0)



###################################################
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    """
    Function to quiet the tracking output.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout