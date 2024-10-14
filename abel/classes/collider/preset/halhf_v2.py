from abel import Collider, SourceBasic, RFAcceleratorBasic, RFAcceleratorCLICopti, DriverComplex, StageBasic, InterstageBasic, PlasmaLinac, DampingRingBasic, BeamDeliverySystemBasic, ConventionalLinac, InteractionPointBasic, TurnaroundBasic
import scipy.constants as SI
import numpy as np

class HALHFv2(Collider):

    def __init__(self):

        super().__init__()
        
        self.com_energy = 250e9
        self.energy_asymmetry = 3
        driver_separation = 3e-9
        Nstages = 25
        transformer_ratio = 2
        driver_energy = (self.com_energy/2)*self.energy_asymmetry/(transformer_ratio*Nstages)
        self.bunch_separation = Nstages*driver_separation
        self.num_bunches_in_train = 100
        self.rep_rate_trains = 100.0 # [Hz]
        
        # define driver
        driver_source = SourceBasic()
        driver_source.charge = -8e-9 # [C]
        driver_source.energy = 0.15e9 # [eV]
        driver_source.rel_energy_spread = 0.01
        driver_source.bunch_length = 700e-6 # [m]
        driver_source.z_offset = 10e-6 # [m]
        driver_source.emit_nx, driver_source.emit_ny = 10e-6, 10e-6 # [m rad]
        driver_source.beta_x, driver_source.beta_y = 30e-3, 30e-3 # [m]
        driver_source.num_particles = 5000
        driver_source.wallplug_efficiency = 0.5
        driver_source.accel_gradient = 10e6 # [V/m]
        
        # define driver accelerator
        driver_accel = RFAcceleratorCLICopti()
        
        driver_accel.nom_energy_gain = driver_energy-driver_source.energy # [eV]
        driver_accel.rf_frequency = 1e9 # [Hz]
        driver_accel.fill_factor = 0.75 # [Hz]
        driver_accel.nom_accel_gradient = 5e6*driver_accel.fill_factor # [V/m]
        driver_accel.num_rf_cells = 20 # [V/m]
        driver_accel.num_structures_per_klystron = 1
        
        # define driver complex
        driver_complex = DriverComplex()
        driver_complex.source = driver_source
        driver_complex.rf_accelerator = driver_accel
        driver_complex.turnaround = TurnaroundBasic()
        
        # define beam
        esource = SourceBasic()
        esource.charge = -1e10 * SI.e # [C]
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.01
        esource.emit_nx, esource.emit_ny = self.energy_asymmetry**2*10e-6, self.energy_asymmetry**2*0.035e-6 # [m rad]
        esource.num_particles = 15000
        esource.wallplug_efficiency = 0.3
        esource.accel_gradient = 23.5e6 # [V/m]
        esource.is_polarized = True
        
        # define stage
        stage = StageBasic()
        stage.driver_source = driver_complex
        stage.nom_accel_gradient = 1e9 # [m]
        stage.plasma_density = 6e20 # [m^-3]
        stage.ramp_beta_mag = 5
        stage.transformer_ratio = transformer_ratio
        #stage.optimize_plasma_density(source=esource)

        # define rest of beam
        esource.bunch_length = 18e-6 # [m]
        esource.z_offset = -34e-6 # [m]
        esource.beta_x = stage.matched_beta_function(esource.energy)
        esource.beta_y = esource.beta_x
        
        # electron injector
        einjector = RFAcceleratorCLICopti()
        einjector.fill_factor = 0.75 # [Hz]
        einjector.nom_accel_gradient = 15e6*einjector.fill_factor # [V/m]
        einjector.nom_energy_gain = 5e9 - esource.energy # [eV]
        einjector.rf_frequency = 3e9
        einjector.num_rf_cells = 70
        einjector.num_structures_per_klystron = 1
        
        # define interstage
        interstage = InterstageBasic()
        interstage.beta0 = lambda E: stage.matched_beta_function(E)
        interstage.dipole_length = lambda E: 1.2 * np.sqrt(E/10e9) # [m(eV)]
        interstage.dipole_field = 0.5 # [T]
        
        # define electron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x, ebds.beta_y = 3.3e-3, 0.1e-3 # [m]
        ebds.bunch_length = 0.75 * ebds.beta_y
        
        # define electron linac
        elinac = PlasmaLinac()
        elinac.driver_complex = driver_complex
        elinac.source = esource
        elinac.rf_injector = einjector
        elinac.stage = stage
        elinac.interstage = interstage
        elinac.bds = ebds
        elinac.num_stages = Nstages


        # define positron source
        psource = SourceBasic()
        psource.charge = 3e10 * SI.e # [C]
        psource.energy = 60e6 # [eV]
        psource.rel_energy_spread = 0.0015
        psource.bunch_length = 75e-6 # [m]
        psource.emit_nx, psource.emit_ny = 1e-2, 1e-2 # [m rad]
        psource.beta_x = 10 # [m]
        psource.beta_y = 10 # [m]
        psource.num_particles = esource.num_particles
        psource.wallplug_efficiency = esource.wallplug_efficiency
        psource.accel_gradient = esource.accel_gradient
        psource.is_polarized = True
        
        # define RF accelerator
        paccel = RFAcceleratorCLICopti()
        paccel.fill_factor = 0.75 # [Hz]
        paccel.nom_accel_gradient = 15e6*paccel.fill_factor # [V/m]
        paccel.rf_frequency = 3e9
        paccel.num_rf_cells = 70
        paccel.num_structures_per_klystron = 1
        
        # injector
        pinjector = RFAcceleratorCLICopti()
        pinjector.fill_factor = paccel.fill_factor
        pinjector.nom_energy_gain = 2.80e9 # [V/m]
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

        # define interaction point
        ip = InteractionPointBasic()
        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = plinac
        self.ip = ip

        # assemble everything
        self.assemble_trackables()
        
        