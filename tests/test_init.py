# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : Core Tests
"""

import os
import pytest

@pytest.mark.core
def testCore_init():
    """Test package initialisation.
    """
    import abel

# END Test testCore_init
