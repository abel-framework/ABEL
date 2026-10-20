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
import numpy as np

class C3(Collider):

    def __init__(self, com_energy=250e9):

        super().__init__()
        
        # OPTIMIZATION VARIABLES
        self.com_energy = com_energy
        
        self.num_bunches_in_train = 133
        self.rep_rate_trains = 120.0 # [Hz]
        self.bunch_separation_ns = 5.26 # [s]

        self.bunch_charge = 1e-9 # [C]

        self.rf_frequency = 5.712e9 # [V/m]
        self.linac_gradient = 70e6 # [V/m]
        self.num_structures_per_klystron = 130

        # from optimization @ 250 GeV: {'bunch_separation_ns': 8.192714440875887, 'linac_gradient': 51417581.36622494, 'num_structures_per_klystron': 270}
        # from optimization @ 550 GeV: {'bunch_separation_ns': 8.534381894337127, 'linac_gradient': 52715332.71703247, 'num_structures_per_klystron': 270}
        

    # pre-assembly of the collider subsystems
    def assemble_trackables(self):
        
        self.bunch_separation = self.bunch_separation_ns*1e-9
        
        # define positron source
        esource = SourceBasic()
        esource.charge = -self.bunch_charge
        esource.energy = 76e6 # [eV]
        esource.rel_energy_spread = 0.0015
        esource.bunch_length = 100e-6 # [m]
        esource.emit_nx, esource.emit_ny = 1e-4, 1e-4 # [m rad]
        esource.beta_x = 10 # [m]
        esource.beta_y = 10 # [m]
        esource.num_particles = 5000
        esource.is_polarized = True
        
        # injector
        einjector = RFAcceleratorBasic()
        einjector.nom_accel_gradient = 30e6 # [V/m]
        einjector.num_rf_cells = 1
        einjector.num_structures_per_klystron = self.num_structures_per_klystron
        einjector.fill_factor = 0.9 # [V/m]
        einjector.nom_energy_gain = 3e9 - esource.energy # [eV]
        einjector.rf_frequency = 2.856e9 # [Hz]
        einjector.operating_temperature = 77 # [K]
        
        # damping ring
        edamping_ring = DampingRingBasic()
        edamping_ring.name = 'Electron damping ring'
        edamping_ring.emit_nx_target = 0.9e-6 # [m rad]
        edamping_ring.emit_ny_target = 0.02e-6 # [m rad]
        
        # define RF accelerator
        eaccel = RFAcceleratorBasic()
        eaccel.nom_energy_gain = self.com_energy/2 - (einjector.nom_energy_gain + esource.energy)
        eaccel.num_rf_cells = 1
        eaccel.num_structures_per_klystron = self.num_structures_per_klystron
        eaccel.fill_factor = 0.9
        eaccel.nom_accel_gradient = self.linac_gradient * eaccel.fill_factor # [V/m]
        eaccel.rf_frequency = self.rf_frequency # [Hz]
        eaccel.operating_temperature = 77 # [K]
        
        # define positron BDS
        ebds = BeamDeliverySystemBasic()
        ebds.length = 1200*np.sqrt(self.com_energy/250e9)
        ebds.beta_x = 12e-3 # [m]
        ebds.beta_y = 0.12e-3 # [m]
        
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
        psource.charge = self.bunch_charge
        psource.is_polarized = False
        
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
        
        # assemble everything
        super().assemble_trackables()
    