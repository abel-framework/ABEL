# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.stage.stage import Stage, PlasmaRamp
from abel.classes.source.source import Source
from abel.classes.source.impl.source_capsule import SourceCapsule
from abel.classes.beamline.impl.driver_complex import DriverComplex
import numpy as np
import scipy.constants as SI
import copy, warnings


SI.r_e = SI.physical_constants['classical electron radius'][0]

class StageBasic(Stage):
    """
    Basic implementation of a plasma stage. Solves Hill's equation, increases 
    the energy of all main beam macro particles with a homogeneous energy gain, 
    and decreases the energy of all drive beam macro particles with the same 
    energy gain.

    Inherits all attributes from :class:`Stage <abel.classes.stage.stage.Stage>`.
    

    Attributes
    ----------
    driver_source : ``Source`` or ``DriverComplex``
        The source of the drive beam. The beam axis is always aligned to its 
        propagation direction. Defaults to ``None``.

    transformer_ratio : float
        Transformer ratio. Default set to 1.0.

    depletion_efficiency : float, optional
        Energy depletion efficiency for the drive beam. Defaults to 0.75.

    probe_evolution : bool, optional
        Flag for storing the beam parameter evolution data. Defaults to 
        ``False``.

    store_beams_for_tests : bool, optional
        Flag for storing intermediate beam states for testing. Defaults to 
        ``False``.

    stage_number : int
        Keeps track of which stage it is in the beamline.
    """
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, transformer_ratio=1.0, depletion_efficiency=0.75, probe_evolution=False, store_beams_for_tests=False):
        """
        Parameters
        ----------
        nom_accel_gradient : [V/m], float
            Nominal acceleration gradient of the acceleration stage.
        
        nom_energy_gain : [eV] float
            Nominal/target energy gain of the acceleration stage.
        
        plasma_density : [m^-3] float
            Plasma density.

        driver_source : ``Source`` or ``DriverComplex``
            The source of the drive beam. The beam axis is always aligned to its 
            propagation direction. Defaults to ``None``.

        ramp_beta_mag : float, optional
            Used for demagnifying and magnifying beams passing through entrance 
            and exit plasma ramps. Defaults to ``None``

        transformer_ratio : float, optional
            Transformer ratio. Defaults to 1.0

        depletion_efficiency : float, optional
            Energy depletion efficiency for the drive beam. Defaults to 0.75.

        probe_evolution : bool, optional
            Flag for storing the beam parameter evolution data. Defaults to 
            ``False``.

        store_beams_for_tests : bool, optional
            Flag for storing intermediate beam states for testing. Defaults to 
            ``False``.
        """
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.transformer_ratio = transformer_ratio
        self.depletion_efficiency = depletion_efficiency
        self.probe_evolution = probe_evolution
        self.store_beams_for_tests = store_beams_for_tests
        

    # ==================================================
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        self.stage_number = beam_incoming.stage_number
        
        # get the driver
        driver_incoming = self.driver_source.track()

        original_driver = copy.deepcopy(driver_incoming)
        original_beam = copy.deepcopy(beam_incoming)
        
        # set ideal plasma density if not defined
        if self.plasma_density is None:
            self.optimize_plasma_density() # TODO: this lacks an input parameter. Fix in separate PR.


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

            if type(self.upramp) is PlasmaRamp and self.upramp.ramp_shape != 'uniform':
                raise TypeError('Only uniform ramps have been implemented.')

            # Pass the drive beam and main beam to track_upramp() and get the ramped beams in return
            beam_ramped, drive_beam_ramped = self.track_upramp(beam_rotated, drive_beam_rotated)
        
        else:  # Do the following if there are no upramp (a lone stage)
            beam_ramped = beam_rotated
            drive_beam_ramped = drive_beam_rotated

        
        # ========== Perform tracking in the flattop stage ==========
        beam, driver = self.main_tracking_procedure(beam_ramped, drive_beam_ramped)


        # ==========  Apply plasma density down ramp (magnify beta function) ==========
        if self.downramp is not None:

            if type(self.downramp) is PlasmaRamp and self.downramp.ramp_shape != 'uniform':
                raise TypeError('Only uniform ramps have been implemented.')

            # TODO: Temporary "drive beam evolution": Magnify the driver
            # Needs to be performed before self.track_downramp().
            driver.magnify_beta_function(self.downramp.ramp_beta_mag, axis_defining_beam=driver)

            # Track the beams through the downramp
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)

        else:  # Do the following if there are no downramp.
            beam_outgoing = beam
            driver_outgoing = driver


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

        # calculate efficiency
        self.calculate_efficiency(original_beam, original_driver, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(original_beam, original_driver, beam_outgoing, driver_outgoing)

        # Copy meta data from input beam_outgoing (will be iterated by super)
        beam_outgoing.trackable_number = original_beam.trackable_number
        beam_outgoing.stage_number = original_beam.stage_number
        beam_outgoing.location = original_beam.location

        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
        

    # ==================================================
    def main_tracking_procedure(self, beam_ramped, drive_beam_ramped):
        """
        Prepares and performs the beam tracking using the physics models of the 
        stage.
        

        Parameters
        ----------
        beam_ramped : ``Beam``
            Main beam.

        drive_beam_ramped : ``Beam``
            Drive beam.

            
        Returns
        -------
        beam : ``Beam`` 
            Main beam after tracking.

        drive_beam : ``Beam``
            Drive beam after tracking.
        """

        beam = beam_ramped
        drive_beam = drive_beam_ramped
        drive_beam_ramped_location = drive_beam_ramped.location
        beam_ramped_location = beam_ramped.location

        # Betatron oscillations
        deltaEs = np.full(len(beam.Es()), self.nom_energy_gain_flattop)  # Homogeneous energy gain for all macroparticles.
        if self.probe_evolution:
            _, evol = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset(), probe_evolution=self.probe_evolution)
            self.evolution.beam = evol
        else:
            beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=drive_beam_ramped.x_offset(), y0_driver=drive_beam_ramped.y_offset())

        # Accelerate beam with homogeneous energy gain
        beam.set_Es(beam.Es() + self.nom_energy_gain_flattop)

        # Decelerate the driver with homogeneous energy loss
        drive_beam.set_Es(drive_beam_ramped.Es()*(1-self.depletion_efficiency))

        # Update the beam locations
        drive_beam.location = drive_beam_ramped_location + self.length_flattop
        beam.location = beam_ramped_location + self.length_flattop

        return beam, drive_beam
    

    # ==================================================
    def track_upramp(self, beam0, driver0):
        """
        Called by a stage to perform upramp tracking.
    
        
        Parameters
        ----------
        driver0 : ``Beam``
            Drive beam.

        beam0 : ``Beam``
            Main beam.
    
            
        Returns
        -------
        beam : ``Beam``
            Main beam after tracking.

        driver : ``Beam``
            Drive beam after tracking.
        """

        # Convert PlasmaRamp to a StageBasic
        if type(self.upramp) is PlasmaRamp:

            upramp = self.convert_PlasmaRamp(self.upramp)
            if type(upramp) is not StageBasic:
                raise TypeError('upramp is not a StageBasic.')

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
        # calculate efficiency
        self.upramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.upramp.calculate_beam_current(beam0, driver0, beam, driver)
            
        # Save parameter evolution to the ramp
        if self.probe_evolution:
            self.upramp.evolution = upramp.evolution  # TODO: save to self instead, but need to change stage diagnostics and how this is saved in self.main_tracking_procedure() first.


        # ========== Modify the driver before the stage ==========
        driver.magnify_beta_function(1/self.upramp.ramp_beta_mag, axis_defining_beam=driver)
            
        return beam, driver
    

    # ==================================================
    def track_downramp(self, beam0, driver0):
        """
        Called by a stage to perform downramp tracking.
    
        
        Parameters
        ----------
        driver0 : ``Beam``
            Drive beam.

        beam0 : ``Beam``
            Main beam.
    
            
        Returns
        -------
        beam : ``Beam``
            Main beam after tracking.

        driver : ``Beam``
            Drive beam after tracking.
        """

        # Convert PlasmaRamp to a StageBasic
        if type(self.downramp) is PlasmaRamp:

            downramp = self.convert_PlasmaRamp(self.downramp)
            if type(downramp) is not StageBasic:
                raise TypeError('downramp is not a StageBasic.')

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
        # calculate efficiency
        self.downramp.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.downramp.calculate_beam_current(beam0, driver0, beam, driver)
            
        # Save parameter evolution to the ramp
        if self.probe_evolution:
            self.downramp.evolution = downramp.evolution  # TODO: save to self instead, but need to change stage diagnostics and how this is saved in self.main_tracking_procedure() first.
            
        return beam, driver


    # ==================================================
    def optimize_plasma_density(self, source):
        """
        Optimize the stage plasma density (float) based on parameters for a 
        given ``source``. The optimised value is stored in 
        ``self.plasma_density``.

        Parameters
        ----------
        source : ``Source``
            ``Source`` object.

        Returns
        -------
        None
        """
        
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
    def copy_config2blank_stage(self, transformer_ratio=None, depletion_efficiency=None, probe_evolution=None):
        """
        Makes a deepcopy of the stage to copy the configurations and settings,
        but most of the parameters in the deepcopy are set to ``None``.
    
        Parameters
        ----------
        transformer_ratio : float
            Transformer ratio. Default set to ``None``.

        depletion_efficiency : float
            Energy depletion efficiency for the drive beam. Default set to 
            ``None``.

        probe_evolution : bool, optional
            Flag for recording the beam parameter evolution. Default set to the
            same value as ``self.probe_evolution``.
            
        Returns
        -------
        stage_copy : ``StageBasic``
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
        if probe_evolution is None:
            stage_copy.probe_evolution = self.probe_evolution

        return stage_copy
    

    # ==================================================
    @Stage.driver_source.setter
    def driver_source(self, source : Source | DriverComplex | None):
        """
        The driver source or the driver complex of the stage. The generated 
        drive beam's beam axis is always aligned to its propagation direction.
        """
        
        # Delegate to parent setter
        Stage.driver_source.fset(self, source)
        
        # Set the driver source to always align drive beam axis to its propagation direction
        if source is not None:

            if isinstance(source, DriverComplex):
                self.driver_source.source.align_beam_axis = True
            elif isinstance(source, Source):
                self.driver_source.align_beam_axis = True

    
###################################################
# Custom formatting that omits the line of source code
def custom_formatwarning(msg, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {msg}\n"

# Tell Python to use custom_formatwarning() instead of the default warnings.formatwarning(), so any subsequent warnings will follow this formatting
warnings.formatwarning = custom_formatwarning
    