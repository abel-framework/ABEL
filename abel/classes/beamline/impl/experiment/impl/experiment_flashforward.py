from abel.classes.beamline.impl.experiment.experiment import Experiment
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.spectrometer.impl.spectrometer_flashforward_impactx import SpectrometerFLASHForwardImpactX

class ExperimentFLASHForward(Experiment):
    
    def __init__(self, energy=1.05e9, plasma_length=0.2, rel_energy_spread=0.01):

        self.energy = energy
        self.rel_energy_spread = rel_energy_spread
        
        self.plasma_length = plasma_length
        
        # set up empty elements
        source = SourceBasic()
        stage = StageBasic()
        spectrometer = SpectrometerFLASHForwardImpactX()
        
        super().__init__(linac=source, component=stage, spectrometer=spectrometer)
        
    # assemble the trackables
    def assemble_trackables(self):

        # define source
        source = self.linac
        source.energy = self.energy
        source.charge = 1e-9
        source.rel_energy_spread = self.rel_energy_spread
        source.bunch_length = 150e-6
        source.z_offset = 0
        source.emit_nx = 3e-6 # [m rad]
        source.emit_ny = 1e-6 # [m rad]
        source.beta_x = 0.02 # [m]
        source.beta_y = 0.02 # [m]
        source.num_particles = 100000

        # define stage
        stage = self.component
        stage.length = self.plasma_length # [eV]
        stage.nom_accel_gradient = 2e9 # [V/m]
        stage.plasma_density = 1e22 # [m^-3]

        # define spectrometer
        spectrometer = self.spectrometer
        spectrometer.imaging_energy = source.energy
        
        # set the bunch train pattern etc.
        super().assemble_trackables()