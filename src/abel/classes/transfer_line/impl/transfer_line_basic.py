# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import scipy.constants as SI
import numpy as np
from abel.classes.transfer_line.transfer_line import TransferLine

class TransferLineBasic(TransferLine):
    
    def __init__(self, nom_energy=None, length=None):
        super().__init__(nom_energy=nom_energy, length=length)
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
