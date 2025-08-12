# -*- coding: utf-8 -*-
"""
ABEL : Stage Geometry calculation tests
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, J.B.B.Chen, O.G.Finnerud,
D.Kallvik, E.Hørlyk, K.N.Sjobak, E.Adli, University of Oslo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pytest
from abel.classes.beamline.impl.linac.linac import Linac


@pytest.mark.linac_unit_test
def test_init():
    "Tests for ``Linac.__init__()``."

    linac = Linac()
    assert linac.source is None
    assert linac.nom_energy is None
    assert linac.num_bunches_in_train is None
    assert linac.bunch_separation is None
    assert linac.rep_rate_trains is None