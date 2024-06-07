from abel import *
import copy

class ILC(Collider):

    def __init__(self):

        super().__init__()

        
        # define positron source
        esource = SourceBasic()
        esource.charge = 2e10 * SI.e # [C]
        esource.energy = 60e6 # [eV]
        esource.rel_energy_spread = 0.0015
        esource.bunch_length = 300e-6 # [m]
        esource.emit_nx, esource.emit_ny = 1e-4, 1e-4 # [m rad]
        esource.beta_x = 10 # [m]
        esource.beta_y = 10 # [m]
        esource.num_particles = 5000
        
        # injector
        einjector = RFAcceleratorBasic()
        einjector.nom_accel_gradient = 30e6 # [V/m]
        einjector.nom_energy_gain = 2.80e9 # [V/m]
        
        # damping ring
        edamping_ring = DampingRingBasic()
        edamping_ring.emit_nx_target = 10e-6 # [m rad]
        edamping_ring.emit_ny_target = 0.035e-6 # [m rad]
        
        # define RF accelerator
        eaccel = RFAcceleratorBasic()
        eaccel.nom_accel_gradient = 30e6 # [V/m]
        eaccel.operating_temperature = 4 # [K]
        
        # define positron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.beta_x = 3.3e-3 # [m]
        ebds.beta_y = 0.1e-3 # [m]
        
        # define positron linac
        elinac = ConventionalLinac()
        elinac.rf_injector = einjector
        elinac.damping_ring = edamping_ring
        elinac.source = esource
        elinac.rf_accelerator = eaccel
        elinac.bds = ebds
        
        
        # define positron source
        psource = copy.deepcopy(esource)
        psource.charge = -esource.charge
        
        # injector
        pinjector = copy.deepcopy(einjector)
        
        # damping ring
        pdamping_ring = copy.deepcopy(edamping_ring)
        
        # define RF accelerator
        paccel = copy.deepcopy(eaccel)
        
        # define positron BDS
        pbds = copy.deepcopy(ebds)
        
        # define positron linac
        plinac = ConventionalLinac()
        plinac.rf_injector = pinjector
        plinac.damping_ring = pdamping_ring
        plinac.source = psource
        plinac.rf_accelerator = paccel
        plinac.bds = pbds

        
        # define interaction point
        ip = InteractionPointGuineaPig()

        
        # define collider (with two different linacs)
        self.linac1 = elinac
        self.linac2 = plinac
        self.ip = ip
        self.com_energy = 250e9
        self.bunch_separation = 312e-9 # [s]
        self.num_bunches_in_train = 1312
        self.rep_rate_trains = 5 # [Hz]

        
        # assemble everything
        self.assemble_trackables()
        
        