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
import numpy as np
import scipy.constants as SI
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.stage.impl.stage_basic import StageBasic


@pytest.mark.linac_unit_test
def test_init():
    "Tests for ``Linac.__init__()``."

    linac = Linac()
    assert linac.source is None
    assert linac.nom_energy is None
    assert linac.num_bunches_in_train is None
    assert linac.bunch_separation is None
    assert linac.rep_rate_trains is None

    linac = Linac(
        source = SourceBasic(),
        nom_energy=5e9,                                                             # [eV]
        num_bunches_in_train=2, 
        bunch_separation=2e-6,                                                      # [s]
        rep_rate_trains=1e3                                                         # [Hz]
    )

    assert type(linac.source) is SourceBasic
    assert np.isclose(linac.nom_energy, 5e9, rtol=1e-15, atol=0.0)
    assert linac.num_bunches_in_train == 2
    assert np.isclose(linac.bunch_separation, 2e-6,rtol=1e-15, atol=0.0)
    assert np.isclose(linac.rep_rate_trains, 1e3,rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_assemble_trackables():
    """
    Tests for ``Linac.assemble_trackables()`` asserting parameters from 
    ``Linac.source`` are correctly copied. to ``Linac``.
    """

    linac = Linac()
    source = SourceBasic()
    stage = StageBasic()
    linac.source = source
    source.bunch_separation = 2e-6
    source.num_bunches_in_train = 3
    source.rep_rate_trains = 1e3

    linac.trackables = []
    linac.trackables.append(source)
    linac.trackables.append(stage)

    linac.assemble_trackables()
    assert type(linac.source) is SourceBasic
    assert linac.num_bunches_in_train == 3
    assert np.isclose(linac.bunch_separation, 2e-6,rtol=1e-15, atol=0.0)
    assert np.isclose(linac.rep_rate_trains, 1e3,rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_nom_energy():
    """
    Tests for ``Linac.nom_energy()`` getter and setter.
    """

    linac = Linac()
    linac.nom_energy = 5e9
    assert np.isclose(linac.nom_energy, 5e9, rtol=1e-15, atol=0.0)
    assert np.isclose(linac.get_nom_energy(), 5e9, rtol=1e-15, atol=0.0)

    # Setting a negative nominal energy should raise an exception
    with pytest.raises(ValueError):
        linac.nom_energy = -1


@pytest.mark.linac_unit_test
def test_get_nom_beam_power():
    """
    Tests for ``Linac.get_nom_beam_power()``.
    """

    nom_energy = 5e9
    charge = -1e9 * SI.e
    num_bunches_in_train = 3
    rep_rate_trains = 1e3

    linac = Linac()
    linac.nom_energy = nom_energy
    source = SourceBasic()
    linac.source = source
    source.charge = charge
    source.num_bunches_in_train = num_bunches_in_train
    source.rep_rate_trains = rep_rate_trains

    get_rep_rate_average = num_bunches_in_train * rep_rate_trains
    nom_beam_power = nom_energy * charge * get_rep_rate_average

    assert np.isclose(linac.get_nom_beam_power(), nom_beam_power, rtol=1e-15, atol=0.0)


    



