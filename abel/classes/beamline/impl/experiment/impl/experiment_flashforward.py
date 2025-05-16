from abel.classes.beamline.impl.experiment.experiment import Experiment
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.source.source import Source
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.stage.impl.stage_hipace import StageHipace
from abel.classes.spectrometer.impl.spectrometer_flashforward_impactx import SpectrometerFLASHForwardImpactX
from abel.classes.beamline.impl.linac.linac import Linac

class ExperimentFLASHForward(Experiment):
    
    def __init__(self, energy=1.05e9, plasma_length=0.033, plasma_density=7e21, rel_energy_spread=0.001, ion_species='H'):

        self.energy = energy
        self.rel_energy_spread = rel_energy_spread
        self.ion_species = ion_species
        self.plasma_length = plasma_length
        self.plasma_density = plasma_density
        
        # set up empty elements
        source = SourceBasic()
        source.energy = self.energy
        source.charge = -0.6e-9
        source.rel_energy_spread = self.rel_energy_spread
        source.bunch_length = 83e-6
        source.z_offset = 0
        source.emit_nx = 3e-6 # [m rad]
        source.emit_ny = 1e-6 # [m rad]
        source.num_particles = 500000
        
        stage = StageHipace()
        stage.num_nodes = 16
        stage.num_cell_xy = 1023
        stage.ion_motion = True
        stage.beam_ionization = True
        stage.ion_species = self.ion_species
        stage.length_flattop = self.plasma_length # [eV]
        stage.nom_accel_gradient = 1e9 # [GV/m]
        
        stage.plasma_density = self.plasma_density # [m^-3]
        stage.ramp_beta_mag = 6
        stage.mesh_refinement = True
        
        stage.upramp = stage.__class__()
        stage.upramp.num_nodes = 5
        stage.upramp.nom_energy = source.energy
        stage.upramp.ion_motion = stage.ion_motion
        stage.upramp.ion_species = stage.ion_species
        stage.upramp.mesh_refinement = stage.mesh_refinement
        stage.upramp.num_cell_xy = stage.num_cell_xy
        
        stage.downramp = stage.__class__()
        stage.downramp.num_nodes = stage.upramp.num_nodes
        stage.downramp.nom_energy = source.energy
        stage.downramp.ion_motion = stage.ion_motion
        stage.downramp.ion_species = stage.ion_species
        stage.downramp.mesh_refinement = stage.mesh_refinement
        stage.downramp.num_cell_xy = stage.num_cell_xy

        source.beta_x = stage.matched_beta_function(source.energy)
        source.beta_y = source.beta_x
        
        spectrometer = SpectrometerFLASHForwardImpactX()
        spectrometer.imaging_energy_x = self.energy
        spectrometer.imaging_energy_y = self.energy
        
        super().__init__(linac=source, component=stage, spectrometer=spectrometer)
        
    # assemble the trackables
    def assemble_trackables(self):
        # define spectrometer
        if isinstance(self.linac, Source):
            self.spectrometer.imaging_energy = self.linac.energy
        elif isinstance(self.linac, Linac):
            self.spectrometer.imaging_energy = self.linac.source.energy
        else:
            raise ValueError("You must define a linac that is either a type Source or a type Linac")

        # set the bunch train pattern etc.
        super().assemble_trackables()