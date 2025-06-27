from abel.classes.collider.collider import Collider
from abel.classes.source.impl.source_trapezoid import SourceTrapezoid
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.rf_accelerator.impl.rf_accelerator_clicopti import RFAcceleratorCLICopti
from abel.classes.rf_accelerator.impl.rf_accelerator_basic import RFAcceleratorBasic
from abel.classes.damping_ring.impl.damping_ring_basic import DampingRingBasic
from abel.classes.beamline.impl.driver_complex import DriverComplex
from abel.classes.turnaround.impl.turnaround_basic import TurnaroundBasic
from abel.classes.combiner_ring.impl.combiner_ring_basic import CombinerRingBasic
from abel.classes.transfer_line.impl.transfer_line_basic import TransferLineBasic
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.interstage.plasma_lens.basic import InterstagePlasmaLensBasic
from abel.classes.bds.impl.bds_basic import BeamDeliverySystemBasic
from abel.classes.beamline.impl.linac.impl.plasma_linac import PlasmaLinac
from abel.classes.beamline.impl.linac.impl.conventional_linac import ConventionalLinac
from abel.classes.ip.impl.ip_basic import InteractionPointBasic
import scipy.constants as SI
import numpy as np

class HALHFv2(Collider):

    def __init__(self, com_energy=250e9, use_cool_copper_positron_linac=True):

        super().__init__()


        # NEW SOLUTION
        
        self.com_energy = com_energy
        self.energy_asymmetry = 3
        
        self.num_bunches_in_train = 160
        self.rep_rate_trains = 100.0 # [Hz]
        
        self.driver_separation_num_buckets = 4
        self.driver_linac_rf_frequency = 1e9 # [Hz]
        self.driver_linac_gradient = 4e6 # [V/m]
        self.driver_linac_structure_num_rf_cells = 23
        self.driver_linac_num_structures_per_klystron = 1.0

        self.positron_charge = 3e10 * SI.e
        self.driver_charge = 8e-9 # [C]
        self.energy_transfer_efficiency = 0.5
        self.driver_depletion_efficiency = 0.8

        self.combiner_ring_compression_factor = 12
        self.num_combiner_rings = 2
        
        self.pwfa_num_stages = 48
        self.pwfa_transformer_ratio = 2
        self.pwfa_gradient = 1e9
        
        self.use_cool_copper_positron_linac = use_cool_copper_positron_linac

        self.positron_linac_rf_frequency = 3e9
        
        self.positron_linac_gradient_cool = 40e6 # [V/m]
        self.positron_linac_num_structures_per_klystron_cool = 140.0
        self.positron_linac_num_rf_cells_cool = 1
        self.positron_linac_temperature_cool = 77 # [K]
    
        self.positron_linac_gradient_warm = 25e6 # [V/m]
        self.positron_linac_num_structures_per_klystron_warm = 1.0
        self.positron_linac_num_rf_cells_warm = 75

        self.num_particles = 10000
        self.electron_ip_bunch_length = 150e-6#230e-6
        self.positron_ip_bunch_length = 150e-6#100e-6
        self.enable_waist_shift = True
        self.waist_shift_frac = 0.5

        self.num_bds = 2
        
    

    # pre-assembly of the collider subsystems
    def assemble_trackables(self):

        self.pwfa_num_stages = int(self.pwfa_num_stages)
        
        if self.use_cool_copper_positron_linac:
            self.positron_linac_gradient = self.positron_linac_gradient_cool # [V/m]
            self.positron_linac_num_structures_per_klystron = float(self.positron_linac_num_structures_per_klystron_cool)
            self.positron_linac_num_rf_cells = self.positron_linac_num_rf_cells_cool
        else:
            self.positron_linac_gradient = self.positron_linac_gradient_warm
            self.positron_linac_num_structures_per_klystron = float(self.positron_linac_num_structures_per_klystron_warm)
            self.positron_linac_num_rf_cells = self.positron_linac_num_rf_cells_warm
         
        driver_separation = self.driver_separation_num_buckets/self.driver_linac_rf_frequency
        colliding_bunch_separation = self.pwfa_num_stages*driver_separation/self.combiner_ring_compression_factor
        driver_energy = (self.com_energy/2)*self.energy_asymmetry/(self.pwfa_transformer_ratio*self.pwfa_num_stages)
        
        self.bunch_separation = colliding_bunch_separation

        electron_charge = -abs(self.energy_transfer_efficiency * self.driver_depletion_efficiency*self.driver_charge/self.pwfa_transformer_ratio)
        
        # define driver
        driver_source = SourceTrapezoid()
        driver_source.charge = self.driver_charge
        driver_source.energy = 0.15e9 # [eV]
        driver_source.rel_energy_spread = 0.01
        driver_source.bunch_length = 1050e-6 # [m]
        driver_source.gaussian_blur = 50e-6 # [m]
        driver_source.current_head = 0.1e3
        driver_source.z_offset = 1630e-6 # [m]
        driver_source.emit_nx, driver_source.emit_ny = 50e-6, 100e-6 # [m rad]
        driver_source.beta_x, driver_source.beta_y = 0.5, 0.5 # [m]
        driver_source.symmetrize = True
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
        driver_accel.use_tunnel = False
        driver_accel.use_cutandcover = True
        driver_accel.use_surfacebuilding = True
        
        # define driver complex
        driver_complex = DriverComplex()
        driver_complex.source = driver_source
        driver_complex.rf_accelerator = driver_accel
        driver_complex.bunch_separation = driver_separation
        
        #driver_complex.turnaround = TurnaroundBasic()
        driver_complex.combiner_ring = CombinerRingBasic()
        driver_complex.combiner_ring.num_rings = self.num_combiner_rings
        driver_complex.combiner_ring.compression_factor = self.combiner_ring_compression_factor

        driver_complex.transfer_line = TransferLineBasic()
        driver_complex.transfer_line.length = 2500 # [m] 125 m, maximum 5% angle
        driver_complex.transfer_line.nom_energy = driver_source.energy

        driver_complex.turnaround = TurnaroundBasic()
        driver_complex.turnaround.bend_radius = 61 # [m] 
        driver_complex.turnaround.nom_energy = driver_source.energy
        driver_complex.turnaround.use_semi_circle = True
        
        # define stage
        stage = StageBasic()
        stage.driver_source = driver_complex
        stage.nom_accel_gradient = self.pwfa_gradient
        stage.plasma_density = 6e20 # [m^-3]
        stage.ramp_beta_mag = 10
        stage.transformer_ratio = self.pwfa_transformer_ratio
        stage.depletion_efficiency = 0.8
        
        # define beam
        esource = SourceBasic()
        #esource.charge = -abs(self.electron_charge) # [C]
        esource.charge = -abs(electron_charge) # [C]
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.01
        esource.emit_nx, esource.emit_ny = self.energy_asymmetry**2*10e-6, self.energy_asymmetry**2*0.035e-6 # [m rad]
        esource.bunch_length = 42e-6 # [m]
        esource.num_particles = self.num_particles
        esource.wallplug_efficiency = 0.3
        esource.accel_gradient = 23.5e6 # [V/m]
        esource.is_polarized = True
        esource.beta_x = stage.matched_beta_function(esource.energy)
        esource.beta_y = esource.beta_x
        esource.bunch_separation = colliding_bunch_separation
        
        # define interstage
        interstage = InterstageBasic()
        interstage.beta0 = lambda E: stage.matched_beta_function(E)
        interstage.dipole_length = lambda E: 0.8 * np.sqrt(E/10e9) # [m(eV)]
        interstage.dipole_field = 0.5 # [T]
        
        # define electron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x = 3.3e-3
        ebds.beta_y = 0.1e-3
        ebds.bunch_length = self.electron_ip_bunch_length
        ebds.num_bds = self.num_bds
        
        # define positron source
        psource = SourceBasic()
        psource.charge = abs(self.positron_charge)
        psource.energy = 60e6 # [eV]
        psource.rel_energy_spread = 0.0015
        psource.bunch_length = 75e-6 # [m]
        psource.emit_nx, psource.emit_ny = 1e-2, 1e-2 # [m rad]
        psource.beta_x = 10 # [m]
        psource.beta_y = 10 # [m]
        psource.num_particles = esource.num_particles
        psource.wallplug_efficiency = esource.wallplug_efficiency
        #psource.accel_gradient = esource.accel_gradient
        psource.length = 176 # [m] based on ILC 500 GeV study.
        psource.bunch_separation = colliding_bunch_separation
        psource.is_polarized = True
        
        # define RF accelerator
        if self.use_cool_copper_positron_linac:
            paccel = RFAcceleratorBasic()
            paccel.operating_temperature = self.positron_linac_temperature_cool
            paccel.fill_factor = 0.9 # [Hz]
            paccel.nom_accel_gradient = self.positron_linac_gradient * paccel.fill_factor
            paccel.num_rf_cells = self.positron_linac_num_rf_cells
            paccel.num_structures_per_klystron = self.positron_linac_num_structures_per_klystron
        else:
            paccel = RFAcceleratorCLICopti()
            paccel.operating_temperature = 330
            paccel.fill_factor = 0.75 # [Hz]
            paccel.nom_accel_gradient = self.positron_linac_gradient * paccel.fill_factor # [V/m] # OPTIMIZATION VARIABLE
            paccel.num_rf_cells = self.positron_linac_num_rf_cells # OPTIMIZATION VARIABLE
            paccel.num_structures_per_klystron = self.positron_linac_num_structures_per_klystron

        paccel.rf_frequency = self.positron_linac_rf_frequency
        
        # injector
        pinjector = paccel.__class__()
        pinjector.fill_factor = paccel.fill_factor
        pinjector.nom_energy_gain = 3e9 # [V/m]
        pinjector.nom_accel_gradient = paccel.nom_accel_gradient
        pinjector.rf_frequency = paccel.rf_frequency
        pinjector.num_rf_cells = paccel.num_rf_cells
        pinjector.num_structures_per_klystron = paccel.num_structures_per_klystron

        ptransfer_line = TransferLineBasic()
        ptransfer_line.ignore_cost_civil_construction = True # shared tunnel
        
        # damping ring
        pdamping_ring = DampingRingBasic()
        pdamping_ring.num_rings = 2
        pdamping_ring.emit_nx_target = 10e-6 # [m rad]
        pdamping_ring.emit_ny_target = 0.035e-6 # [m rad]
        
        # define positron BDS
        pbds = BeamDeliverySystemBasic()
        pbds.beta_x = ebds.beta_x # [m]
        pbds.beta_y = ebds.beta_y # [m]
        pbds.bunch_length = self.positron_ip_bunch_length
        pbds.num_bds = self.num_bds
        
        # define positron linac
        plinac = ConventionalLinac()
        plinac.rf_injector = pinjector
        plinac.transfer_line = ptransfer_line
        plinac.damping_ring = pdamping_ring
        plinac.source = psource
        plinac.rf_accelerator = paccel
        plinac.bds = pbds
        
        # electron injector
        einjector = paccel.__class__()
        einjector.fill_factor = paccel.fill_factor # [Hz]
        einjector.nom_accel_gradient = paccel.nom_accel_gradient
        einjector.nom_energy_gain = 3e9 - esource.energy # [eV]
        einjector.rf_frequency = paccel.rf_frequency
        einjector.num_rf_cells = paccel.num_rf_cells
        einjector.num_structures_per_klystron = paccel.num_structures_per_klystron
        einjector.beta_x = stage.matched_beta_function(esource.energy + einjector.nom_energy_gain)
        einjector.beta_y = einjector.beta_x

        # define electron linac
        elinac = PlasmaLinac()
        elinac.driver_complex = driver_complex
        elinac.source = esource
        elinac.rf_injector = einjector
        elinac.stage = stage
        elinac.interstage = interstage
        elinac.bds = ebds
        elinac.num_stages = self.pwfa_num_stages
        
        
        # define interaction point
        ip = InteractionPointBasic()
        #ip = InteractionPointGuineaPig()
        ip.num_ips = self.num_bds
        ip.enable_waist_shift = self.enable_waist_shift
        ip.waist_shift_frac = self.waist_shift_frac
        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = plinac
        self.ip = ip
        
        # assemble everything
        super().assemble_trackables()

        # set the transfer line length
        ptransfer_line.length = ebds.get_length() + pbds.get_length() + paccel.get_length() - pinjector.get_length()
        