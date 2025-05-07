from abel.classes.collider.collider import Collider
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.rf_accelerator.impl.scrf_accelerator_basic import SCRFAcceleratorBasic
from abel.classes.damping_ring.impl.damping_ring_basic import DampingRingBasic
from abel.classes.bds.impl.bds_basic import BeamDeliverySystemBasic
from abel.classes.beamline.impl.linac.impl.conventional_linac import ConventionalLinac
from abel.classes.ip.impl.ip_basic import InteractionPointBasic
import copy

class ILC(Collider):

    def __init__(self):

        super().__init__()
        
        
        # define positron source
        esource = SourceBasic()
        esource.charge = -2e10 * SI.e # [C]
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.0015
        esource.bunch_length = 300e-6 # [m]
        esource.emit_nx, esource.emit_ny = 1e-4, 1e-4 # [m rad]
        esource.beta_x = 10 # [m]
        esource.beta_y = 10 # [m]
        esource.num_particles = 5000
        esource.is_polarized = True
        
        # injector
        einjector = SCRFAcceleratorBasic()
        einjector.nom_accel_gradient = 31.5e6*0.711 # [V/m]
        einjector.nom_energy_gain = 5e9 - esource.energy # [eV]
        
        # damping ring
        edamping_ring = DampingRingBasic()
        edamping_ring.name = 'Electron damping ring'
        edamping_ring.emit_nx_target = 10e-6 # [m rad]
        edamping_ring.emit_ny_target = 0.035e-6 # [m rad]
        
        # define RF accelerator
        eaccel = SCRFAcceleratorBasic()
        eaccel.nom_accel_gradient = 31.5e6*0.711 # [V/m]
        
        # define positron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.length = 2.25e3 # [m]
        ebds.beta_x = 3.3e-3 # [m]
        ebds.beta_y = 0.1e-3 # [m]
        
        # define positron linac
        elinac = ConventionalLinac()
        elinac.name = 'Electron arm'
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
        pdamping_ring.name = 'Positron damping ring'
        
        # define RF accelerator
        paccel = copy.deepcopy(eaccel)
        
        # define positron BDS
        pbds = copy.deepcopy(ebds)
        
        # define positron linac
        plinac = ConventionalLinac()
        plinac.name = 'Positron arm'
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
        self.com_energy = 500e9
        self.bunch_separation = 554e-9 # [s]
        self.num_bunches_in_train = 1312
        self.rep_rate_trains = 5 # [Hz]

        
        # assemble everything
        self.assemble_trackables()
        
        