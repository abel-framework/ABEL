from abel import *

class HALHFv1(Collider):

    def __init__(self):

        super().__init__()
        
        # define driver
        driver_source = SourceBasic()
        driver_source.charge = -2.7e10 * SI.e # [C]
        driver_source.energy = 0.15e9 # [eV]
        driver_source.rel_energy_spread = 0.01
        driver_source.bunch_length = 42e-6 # [m]
        driver_source.z_offset = 300e-6 # [m]
        driver_source.emit_nx, driver_source.emit_ny = 10e-6, 10e-6 # [m rad]
        driver_source.beta_x, driver_source.beta_y = 30e-3, 30e-3 # [m]
        driver_source.num_particles = 5000
        driver_source.wallplug_efficiency = 0.5
        driver_source.accel_gradient = 10e6 # [V/m]
        
        # define driver accelerator
        driver_accel = RFAcceleratorBasic()
        driver_accel.nom_energy_gain = 31.1e9 # [eV]
        driver_accel.nom_accel_gradient = 25e6 # [V/m]
        
        # define driver complex
        driver_complex = DriverComplex()
        driver_complex.source = driver_source
        driver_complex.rf_accelerator = driver_accel
        driver_complex.turnaround = TurnaroundBasic()

        # define beam
        esource = SourceBasic()
        esource.charge = -1e10 * SI.e # [C]
        esource.energy = 76e6 # [eV] same as ILC source
        esource.rel_energy_spread = 0.01
        esource.emit_nx, esource.emit_ny = 160e-6, 0.56e-6 # [m rad]
        esource.num_particles = 5000
        esource.wallplug_efficiency = 0.1
        esource.accel_gradient = 25e6 # [V/m]
        esource.is_polarized = True
        
        # define stage
        stage = StageBasic()
        stage.driver_source = driver_complex
        stage.nom_accel_gradient = 6.4e9 # [m]
        stage.ramp_beta_mag = 5
        charge = -1e10 * SI.e # [C]
        stage.optimize_plasma_density(source=esource)
        
        # define rest of beam
        esource.bunch_length = 18e-6 # [m]
        esource.z_offset = -34e-6 # [m]
        esource.beta_x = stage.matched_beta_function(esource.energy)
        esource.beta_y = esource.beta_x

        # electron injector
        einjector = RFAcceleratorBasic()
        einjector.name = 'Electron RF injector'
        einjector.nom_accel_gradient = 31.5e6*0.711 # [V/m]
        einjector.nom_energy_gain = 5e9 - esource.energy # [eV]
        einjector.rf_frequency = 3e9
        einjector.structure_length = 5
        einjector.peak_power_klystron = 50e6
        
        # define interstage
        interstage = InterstageBasic()
        interstage.beta0 = lambda E: stage.matched_beta_function(E)
        interstage.dipole_length = lambda E: 1 * np.sqrt(E/10e9) # [m(eV)]
        interstage.dipole_field = 0.5 # [T]
        
        # define electron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x, ebds.beta_y = 3.3e-3, 0.1e-3 # [m]
        ebds.bunch_length = 0.75 * ebds.beta_y
        
        # define electron linac
        elinac = PlasmaLinac()
        elinac.name = 'Electron arm (plasma)'
        elinac.driver_complex = driver_complex
        elinac.source = esource
        elinac.rf_injector = einjector
        elinac.stage = stage
        elinac.interstage = interstage
        elinac.bds = ebds
        elinac.num_stages = 16


        # define positron source
        psource = SourceBasic()
        psource.charge = 4e10 * SI.e # [C]
        psource.energy = 60e6 # [eV]
        psource.rel_energy_spread = 0.0015
        psource.bunch_length = 75e-6 # [m]
        psource.emit_nx, psource.emit_ny = 1e-2, 1e-2 # [m rad]
        psource.beta_x = 10 # [m]
        psource.beta_y = 10 # [m]
        psource.num_particles = esource.num_particles
        psource.wallplug_efficiency = 0.5
        psource.accel_gradient = 25e6 # [V/m]
        psource.is_polarized = False
        
        # injector
        pinjector = RFAcceleratorBasic()
        pinjector.nom_accel_gradient = 25e6 # [V/m]
        pinjector.nom_energy_gain = 2.80e9 # [V/m]
        pinjector.rf_frequency = 3e9
        pinjector.structure_length = 5
        pinjector.peak_power_klystron = 50e6
        
        # damping ring
        damping_ring = DampingRingBasic()
        damping_ring.name = 'Positron damping ring'
        damping_ring.emit_nx_target = 10e-6 # [m rad]
        damping_ring.emit_ny_target = 0.035e-6 # [m rad]
        
        # define RF accelerator
        paccel = RFAcceleratorBasic()
        paccel.nom_accel_gradient = 25e6 # [V/m]
        paccel.rf_frequency = 3e9
        paccel.structure_length = 5
        paccel.peak_power_klystron = 50e6

        # define positron turnaround
        pturnaround = TurnaroundBasic()
        
        # define positron BDS
        pbds = BeamDeliverySystemBasic()
        pbds.beta_x = 3.3e-3 # [m]
        pbds.beta_y = 0.1e-3 # [m]
        
        # define positron linac
        plinac = ConventionalLinac()
        plinac.name = 'Positron arm (RF)'
        plinac.source = psource
        plinac.rf_injector = pinjector
        plinac.damping_ring = damping_ring
        plinac.rf_accelerator = paccel
        plinac.turnaround = pturnaround
        plinac.bds = pbds

        # define interaction point
        ip = InteractionPointBasic()
        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = plinac
        self.ip = ip
        self.com_energy = 250e9
        self.energy_asymmetry = 4
        self.bunch_separation = 80e-9 # [s]
        self.num_bunches_in_train = 100
        self.rep_rate_trains = 100 # [Hz]

        # assemble everything
        self.assemble_trackables()
        
        