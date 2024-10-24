from abel import *
import scipy.constants as SI
import numpy as np

class HALHFv2(Collider):

    def __init__(self):

        super().__init__()

        # OPTIMIZATION VARIABLES
        self.com_energy = 250e9 # [eV]
        self.energy_asymmetry = 2.7
        
        self.num_bunches_in_train = 100
        self.rep_rate_trains = 100.0 # [Hz]
        
        self.driver_separation_num_buckets = 3
        self.driver_linac_rf_frequency = 1e9 # [Hz]
        self.driver_linac_gradient = 4e6 # [V/m]
        self.driver_linac_structure_num_rf_cells = 20
        self.driver_linac_num_structures_per_klystron = 1.0

        self.use_cool_copper_positron_linac = True
        if self.use_cool_copper_positron_linac:
            self.positron_linac_gradient = 30e6 # [V/m]
            self.positron_linac_num_structures_per_klystron = 300.0
            self.positron_linac_num_rf_cells = 1
        else:
            self.positron_linac_gradient = 20e6 # [V/m]
            self.positron_linac_num_structures_per_klystron = 1.0
            self.positron_linac_num_rf_cells = 80
        
        self.combiner_ring_compression_factor = 5
        
        self.pwfa_num_stages = 32
        self.pwfa_transformer_ratio = 2
        self.pwfa_gradient = 1e9
        

    # pre-assembly of the collider subsystems
    def assemble_trackables(self):

        driver_separation = self.driver_separation_num_buckets/self.driver_linac_rf_frequency
        colliding_bunch_separation = self.pwfa_num_stages*driver_separation/self.combiner_ring_compression_factor
        driver_energy = (self.com_energy/2)*self.energy_asymmetry/(self.pwfa_transformer_ratio*self.pwfa_num_stages)

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
        #driver_source.bunch_separation = driver_separation
        
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
        driver_complex.combiner_ring.compression_factor = self.combiner_ring_compression_factor
        
        # define beam
        esource = SourceBasic()
        esource.charge = -1e10 * SI.e # [C]
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.01
        esource.emit_nx, esource.emit_ny = self.energy_asymmetry**2*10e-6, self.energy_asymmetry**2*0.035e-6 # [m rad]
        esource.num_particles = 1000
        esource.wallplug_efficiency = 0.3
        esource.accel_gradient = 23.5e6 # [V/m]
        esource.is_polarized = True
        esource.bunch_separation = colliding_bunch_separation
        
        # define stage
        stage = StageBasic()
        stage.driver_source = driver_complex
        stage.nom_accel_gradient = self.pwfa_gradient
        #stage.plasma_density = stage.optimize_plasma_density(source=esource)
        stage.plasma_density = 6e20 # [m^-3]
        stage.ramp_beta_mag = 10
        stage.transformer_ratio = self.pwfa_transformer_ratio

        # define rest of beam
        esource.bunch_length = 18e-6 # [m]
        esource.z_offset = -34e-6 # [m]
        esource.beta_x = stage.matched_beta_function(esource.energy)
        esource.beta_y = esource.beta_x
        
        # define interstage
        interstage = InterstageBasic()
        interstage.beta0 = lambda E: stage.matched_beta_function(E)
        interstage.dipole_length = lambda E: 1.2 * np.sqrt(E/10e9) # [m(eV)]
        interstage.dipole_field = 0.5 # [T]
        
        # define electron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x, ebds.beta_y = 3.3e-3, 0.1e-3 # [m]
        ebds.bunch_length = 0.75 * ebds.beta_y
        
        # define positron source
        psource = SourceBasic()
        psource.charge = 3e10 * SI.e # [C]  # OPTIMIZATION VARIABLE
        psource.energy = 60e6 # [eV]
        psource.rel_energy_spread = 0.0015
        psource.bunch_length = 75e-6 # [m]
        psource.emit_nx, psource.emit_ny = 1e-2, 1e-2 # [m rad]
        psource.beta_x = 10 # [m]
        psource.beta_y = 10 # [m]
        psource.num_particles = esource.num_particles
        psource.wallplug_efficiency = esource.wallplug_efficiency
        psource.accel_gradient = esource.accel_gradient
        psource.bunch_separation = colliding_bunch_separation
        psource.is_polarized = True
        
        # define RF accelerator
        if self.use_cool_copper_positron_linac:
            paccel = RFAcceleratorBasic()
            paccel.operating_temperature = 77
            paccel.fill_factor = 0.9 # [Hz]
            paccel.nom_accel_gradient = self.positron_linac_gradient * paccel.fill_factor
            paccel.num_rf_cells = self.positron_linac_num_rf_cells
            paccel.num_structures_per_klystron = self.positron_linac_num_structures_per_klystron
        else:
            paccel = RFAcceleratorCLICopti()
            #paccel = RFAcceleratorBasic()
            paccel.operating_temperature = 330
            paccel.fill_factor = 0.75 # [Hz]
            paccel.nom_accel_gradient = self.positron_linac_gradient * paccel.fill_factor # [V/m] # OPTIMIZATION VARIABLE
            paccel.num_rf_cells = self.positron_linac_num_rf_cells # OPTIMIZATION VARIABLE
            paccel.num_structures_per_klystron = self.positron_linac_num_structures_per_klystron
        paccel.rf_frequency = 3e9
        
        # injector
        pinjector = paccel.__class__()
        pinjector.fill_factor = paccel.fill_factor
        pinjector.nom_energy_gain = 3e9 # [V/m]
        pinjector.nom_accel_gradient = paccel.nom_accel_gradient
        pinjector.rf_frequency = paccel.rf_frequency
        pinjector.num_rf_cells = paccel.num_rf_cells
        pinjector.num_structures_per_klystron = paccel.num_structures_per_klystron
        
        # damping ring
        pdamping_ring = DampingRingBasic()
        pdamping_ring.emit_nx_target = 10e-6 # [m rad]
        pdamping_ring.emit_ny_target = 0.035e-6 # [m rad]
        
        # define positron BDS
        pbds = BeamDeliverySystemBasic()
        pbds.beta_x = 3.3e-3 # [m]
        pbds.beta_y = 0.1e-3 # [m]
        
        # define positron linac
        plinac = ConventionalLinac()
        plinac.rf_injector = pinjector
        plinac.damping_ring = pdamping_ring
        plinac.source = psource
        plinac.rf_accelerator = paccel
        plinac.bds = pbds

        
        # electron injector
        einjector = paccel.__class__()
        einjector.fill_factor = paccel.fill_factor # [Hz]
        einjector.nom_accel_gradient = paccel.nom_accel_gradient
        einjector.nom_energy_gain = 5e9 - esource.energy # [eV]
        einjector.rf_frequency = paccel.rf_frequency
        einjector.num_rf_cells = paccel.num_rf_cells
        einjector.num_structures_per_klystron = paccel.num_structures_per_klystron

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
        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = plinac
        self.ip = ip

        # assemble everything
        super().assemble_trackables()
    
        