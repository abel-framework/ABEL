from abel.classes.collider.collider import Collider
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.rf_accelerator.impl.rf_accelerator_clicopti import RFAcceleratorCLICopti
from abel.classes.rf_accelerator.impl.rf_accelerator_basic import RFAcceleratorBasic
from abel.classes.damping_ring.impl.damping_ring_basic import DampingRingBasic
from abel.classes.beamline.impl.driver_complex import DriverComplex
from abel.classes.turnaround.impl.turnaround_basic import TurnaroundBasic
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.combiner_ring.impl.combiner_ring_basic import CombinerRingBasic
from abel.classes.interstage.plasma_lens.basic import InterstagePlasmaLensBasic
from abel.classes.bds.impl.bds_basic import BeamDeliverySystemBasic
from abel.classes.beamline.impl.linac.impl.plasma_linac import PlasmaLinac
from abel.classes.ip.impl.ip_basic import InteractionPointBasic
import scipy.constants as SI
import numpy as np

class HALHFgg(Collider):

    def __init__(self):

        super().__init__()

        # OPTIMIZATION VARIABLES
        self.com_energy = 160e9 # [eV]
        
        self.num_bunches_in_train = 160
        self.rep_rate_trains = 100.0 # [Hz]
        
        self.driver_separation_num_buckets = 4
        self.driver_linac_rf_frequency = 1e9 # [Hz]
        self.driver_linac_gradient = 4e6 # [V/m]
        self.driver_linac_structure_num_rf_cells = 23
        self.driver_linac_num_structures_per_klystron = 1.0

        self.combiner_ring_compression_factor = 12
        self.num_combiner_rings = 2
        
        self.pwfa_num_stages = 32
        self.pwfa_transformer_ratio = 2
        self.pwfa_gradient = 1e9

        self.target_integrated_luminosity = 2e46
        
        

    # pre-assembly of the collider subsystems
    def assemble_trackables(self):

        driver_separation = self.driver_separation_num_buckets/self.driver_linac_rf_frequency
        colliding_bunch_separation = self.pwfa_num_stages*driver_separation
        driver_energy = (self.com_energy/2)/(self.pwfa_transformer_ratio*self.pwfa_num_stages)

        self.bunch_separation = colliding_bunch_separation
        
        # define driver
        driver_source = SourceBasic()
        driver_source.charge = -8e-9 # [C]
        driver_source.energy = 0.15e9 # [eV]
        driver_source.rel_energy_spread = 0.01
        driver_source.bunch_length = 700e-6 # [m]
        driver_source.z_offset = 10e-6 # [m]
        driver_source.emit_nx, driver_source.emit_ny = 10e-6, 10e-6 # [m rad]
        driver_source.beta_x, driver_source.beta_y = 30e-3, 30e-3 # [m]
        driver_source.num_particles = 1000
        driver_source.wallplug_efficiency = 0.5
        driver_source.accel_gradient = 10e6 # [V/m]
        
        # define driver accelerator
        driver_accel = RFAcceleratorCLICopti()
        driver_accel.nom_energy_gain = driver_energy-driver_source.energy # [eV]
        driver_accel.rf_frequency = self.driver_linac_rf_frequency
        driver_accel.fill_factor = 0.75 # [Hz]
        driver_accel.nom_accel_gradient = self.driver_linac_gradient * driver_accel.fill_factor # [V/m]  # OPTIMIZATION VARIABLE
        driver_accel.num_rf_cells = self.driver_linac_structure_num_rf_cells
        driver_accel.num_structures_per_klystron = self.driver_linac_num_structures_per_klystron

        
        # define driver complex
        driver_complex = DriverComplex()
        driver_complex.source = driver_source
        driver_complex.rf_accelerator = driver_accel
        driver_complex.bunch_separation = driver_separation
        
        #driver_complex.turnaround = TurnaroundBasic()
        driver_complex.combiner_ring = CombinerRingBasic()
        driver_complex.combiner_ring.num_rings = self.num_combiner_rings
        driver_complex.combiner_ring.compression_factor = self.combiner_ring_compression_factor
        driver_complex.combiner_ring.exit_angle = np.pi
        
        # define beam
        esource = SourceBasic()
        esource.charge = -1e10 * SI.e # [C]
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.01
        esource.emit_nx, esource.emit_ny = 1e-6, 1e-6 # [m rad]
        esource.num_particles = 1000
        esource.wallplug_efficiency = 0.3
        esource.accel_gradient = 23.5e6 # [V/m]
        esource.is_polarized = True
        esource.bunch_separation = colliding_bunch_separation
        
        # define stage
        stage = StageBasic()
        stage.driver_source = driver_complex
        stage.nom_accel_gradient = self.pwfa_gradient
        stage.plasma_density = 6e20 # [m^-3]
        stage.ramp_beta_mag = 10
        stage.transformer_ratio = self.pwfa_transformer_ratio

        # define rest of beam
        esource.bunch_length = 18e-6 # [m]
        esource.z_offset = -34e-6 # [m]
        esource.beta_x = stage.matched_beta_function(esource.energy)
        esource.beta_y = esource.beta_x
        
        # define interstage
        interstage = InterstagePlasmaLensBasic()
        interstage.beta0 = lambda E: stage.matched_beta_function(E)
        interstage.dipole_length = lambda E: 1.2 * np.sqrt(E/10e9) # [m(eV)]
        interstage.dipole_field = 0.5 # [T]
        
        # define electron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x, ebds.beta_y = 3.3e-3, 0.1e-3 # [m]
        ebds.bunch_length = 0.75 * ebds.beta_y
        
        # electron injector
        einjector = RFAcceleratorBasic()
        einjector.fill_factor = 0.75
        einjector.nom_accel_gradient = 20e6
        einjector.nom_energy_gain = 1e9 - esource.energy # [eV]
        einjector.rf_frequency = 3e9
        einjector.num_rf_cells = 100
        einjector.num_structures_per_klystron = 1

        # define electron linac
        elinac = PlasmaLinac()
        elinac.driver_complex = driver_complex
        elinac.source = esource
        elinac.rf_injector = einjector
        elinac.stage = stage
        elinac.num_stages = self.pwfa_num_stages
        elinac.interstage = interstage
        elinac.bds = ebds
        
        # define interaction point
        ip = InteractionPointBasic()
        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = copy.deepcopy(elinac)
        self.ip = ip

        # assemble everything
        super().assemble_trackables()
        