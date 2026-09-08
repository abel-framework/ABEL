# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.collider.collider import Collider
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.rf_accelerator.impl.rf_accelerator_basic import RFAcceleratorBasic
from abel.classes.damping_ring.impl.damping_ring_basic import DampingRingBasic
from abel.classes.bds.impl.bds_basic import BeamDeliverySystemBasic
from abel.classes.beamline.impl.linac.impl.conventional_linac import ConventionalLinac
from abel.classes.ip.impl.ip_basic import InteractionPointBasic
import copy
import scipy.constants as SI

class CLIC(Collider):

    def __init__(self):

        super().__init__()


        self.com_energy = 380e9
        self.energy_asymmetry = 1.0

        # define positron source
        esource = SourceBasic()
        esource.charge = -3.87e9 * SI.e # [C]
        esource.energy = 76e6 # [eV] from ILC
        esource.rel_energy_spread = 0.0015
        esource.bunch_length = 60e-6 # [m]
        esource.emit_nx, esource.emit_ny = 1e-4, 1e-4 # [m rad]
        esource.beta_x = 10 # [m]
        esource.beta_y = 10 # [m]
        esource.num_particles = 5000
        esource.is_polarized = True
        
        # injector
        einjector = RFAcceleratorBasic()
        einjector.nom_accel_gradient = 72e6*0.718 # [V/m]
        einjector.nom_energy_gain = 2.86e9 - esource.energy # [eV]
        
        
        # damping ring
        edamping_ring = DampingRingBasic()
        edamping_ring.name = 'Electron damping ring'
        edamping_ring.emit_nx_target = 0.63e-6 # [m rad]
        edamping_ring.emit_ny_target = 0.02e-6 # [m rad]
        
        # define RF accelerator
        eaccel = RFAcceleratorBasic()
        eaccel.nom_accel_gradient = 72e6*0.718 # [V/m]
        eaccel.fill_factor = 0.718 
        eaccel.rf_frequency = 12e9 # [Hz]
        eaccel.structure_length = 0.46 # [m]
        eaccel.peak_power_klystron = 170e6
        
        # define positron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.length = 2.2e3 # [m]
        ebds.beta_x = 0.836e-3 # [m]
        ebds.beta_y = 0.1564e-3 # [m]
        
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
        self.bunch_separation = 0.5e-9 # [s]
        self.num_bunches_in_train = 485
        self.rep_rate_trains = 50.0 # [Hz]

        
        # assemble everything
        self.assemble_trackables()
        
        