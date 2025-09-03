from abel.classes.beamline.impl.experiment.pwfa import ExperimentPWFA
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.source.source import Source
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.stage.impl.stage_hipace import StageHipace
from abel.classes.spectrometer.impl.spectrometer_flashforward_impactx import SpectrometerFLASHForwardImpactX
from abel.classes.beamline.impl.linac.linac import Linac

class FACET2(ExperimentPWFA):
    
    def __init__(self, energy=10e9, charge=1.2e-9, plasma_length=0.40/10, plasma_density=4e22, beta_x=0.5, beta_y=0.05, rel_energy_spread=0.01, ion_species='Li'):

        self.energy = energy
        self.charge = charge
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.rel_energy_spread = rel_energy_spread
        self.ion_species = ion_species
        self.plasma_length = plasma_length
        self.plasma_density = plasma_density
        
        # set up empty elements
        source = SourceBasic()
        source.energy = self.energy
        source.charge = -1*abs(self.charge)
        source.rel_energy_spread = self.rel_energy_spread
        source.bunch_length = 30e-6
        source.z_offset = 0
        source.beta_x = self.beta_x
        source.beta_y = self.beta_y
        source.emit_nx = 20e-6 # [m rad]
        source.emit_ny = 20e-6 # [m rad]
        source.num_particles = 200000
        source.length = 1000.0
        
        stage = StageHipace()
        #stage = StageBasic()
        stage.num_nodes = 16
        stage.num_cell_xy = 511
        stage.ion_motion = True
        stage.beam_ionization = True
        stage.ion_species = self.ion_species
        stage.length_flattop = self.plasma_length # [eV]
        stage.plasma_density = self.plasma_density
        stage.nom_accel_gradient = 10e9 # [GV/m]
        
        spectrometer = SpectrometerFLASHForwardImpactX()
        spectrometer.imaging_energy_x = self.energy
        spectrometer.imaging_energy_y = self.energy
        
        super().__init__(linac=source, stage=stage, spectrometer=spectrometer)
        