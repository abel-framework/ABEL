from abel.classes.beamline.impl.experiment.pwfa import ExperimentPWFA
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.rf_accelerator.impl.rf_accelerator_basic import RFAcceleratorBasic
from abel.classes.beamline.impl.linac.impl.conventional_linac import ConventionalLinac
from abel.classes.source.source import Source
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.stage.impl.stage_quasistatic_2d import StageQuasistatic2d
from abel.classes.stage.impl.stage_hipace import StageHipace
from abel.classes.spectrometer.quad_imaging.preset.facet2 import SpectrometerFACET2
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.bds.impl.bds_basic import BeamDeliverySystemBasic

class FACET2(ExperimentPWFA):
    
    def __init__(self, energy=10e9, charge=1.2e-9, plasma_length=0.40, plasma_density=4e22, beta_x=0.5, beta_y=0.05, rel_energy_spread=0.01, ion_species='Li', num_particles=100000, stage_class=StageQuasistatic2d):

        super().__init__()
        
        self.energy = energy
        self.charge = charge
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.rel_energy_spread = rel_energy_spread
        self.ion_species = ion_species
        self.plasma_length = plasma_length
        self.plasma_density = plasma_density

        self.num_particles = num_particles
        
        self.stage_class = stage_class
        

    
    # assemble the trackables
    def assemble_trackables(self):
        

        # set up source
        source = SourceBasic()
        source.energy = 20e6
        source.charge = -1*abs(self.charge)
        source.rel_energy_spread = self.rel_energy_spread
        source.bunch_length = 30e-6
        source.z_offset = 0
        source.beta_x = 1.0
        source.beta_y = 1.0
        source.emit_nx = 5e-6 # [m rad]
        source.emit_ny = 5e-6 # [m rad]
        source.num_particles = self.num_particles
        
        # set up RF accelerator
        rf_accelerator = RFAcceleratorBasic()
        rf_accelerator.length = 900.0
        rf_accelerator.nom_energy_gain = self.energy - source.energy

        # set up BDS (TODO: make into W-chicane with collimators)
        bds = BeamDeliverySystemBasic()
        bds.beta_x = self.beta_x
        bds.beta_y = self.beta_y
        bds.length = 100.0
        
        # set up full RF linac
        linac = ConventionalLinac()
        linac.source = source
        linac.rf_accelerator = rf_accelerator
        linac.bds = bds
        
        # set up PWFA stage
        stage = self.stage_class()
        stage.num_nodes = 16
        stage.num_cell_xy = 511
        stage.ion_motion = True
        stage.beam_ionization = True
        stage.ion_species = self.ion_species
        stage.length_flattop = self.plasma_length # [eV]
        stage.plasma_density = self.plasma_density
        stage.nom_accel_gradient = 10e9 # [GV/m]
        stage.probe_evolution = True

        # set up spectrometer
        spectrometer = SpectrometerFACET2()

        # assigning the objects
        self.linac = linac
        self.stage = stage
        self.spectrometer = spectrometer
        
        # run PWFA experiment constructor
        super().assemble_trackables()
    
    