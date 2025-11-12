# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later


"""
ABEL : unit tests for the Linac and Beamline class
"""

import pytest
import numpy as np
import scipy.constants as SI

from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.stage.impl.stage_basic import StageBasic
from abel.classes.interstage.plasma_lens.basic import InterstagePlasmaLensBasic


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
    assert np.isclose(linac.bunch_separation, 2e-6, rtol=1e-15, atol=0.0)
    assert np.isclose(linac.rep_rate_trains, 1e3, rtol=1e-15, atol=0.0)

    # Setting a negative nominal energy should raise an exception
    with pytest.raises(ValueError):
        linac = Linac(nom_energy=-5e9)

    # Setting a negative num_bunches_in_train should raise an exception
    with pytest.raises(ValueError):
        linac = Linac(num_bunches_in_train=-1)

    # Setting a float num_bunches_in_train should raise an exception
    with pytest.raises(TypeError):
        linac = Linac(num_bunches_in_train=1.0)

    # Setting a negative bunch_separation should raise an exception
    with pytest.raises(ValueError):
        linac = Linac(bunch_separation=-0.2)

    # Setting a negative rep_rate_trains should raise an exception
    with pytest.raises(ValueError):
        linac = Linac(rep_rate_trains=-0.2)


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
    linac.num_bunches_in_train = num_bunches_in_train
    linac.rep_rate_trains = rep_rate_trains

    rep_rate_average = num_bunches_in_train * rep_rate_trains
    nom_beam_power = nom_energy * np.abs(charge) * rep_rate_average

    assert np.isclose(linac.get_nom_beam_power(), nom_beam_power, rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_get_effective_gradient():
    """
    Tests for ``Linac.get_effective_gradient()``.
    """

    linac = Linac()
    linac.nom_energy = 5e9
    stage = StageBasic()
    stage.length = 1.0
    linac.trackables = []
    linac.trackables.append(stage)

    assert np.isclose(linac.get_effective_gradient(), 5e9, rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_energy_usage():
    """
    Tests for ``Linac.energy_usage()``.
    """

    linac = Linac()
    source = SourceBasic()
    source.energy = 1e9
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    stage = StageBasic()
    stage.driver_source = driver_source
    linac.source = source
    linac.trackables = []
    linac.trackables.append(source)
    linac.trackables.append(stage)

    assert np.isclose(linac.energy_usage(), source.energy_usage() + stage.energy_usage(), rtol=1e-15, atol=0.0)
    

@pytest.mark.linac_unit_test
def test_get_cost_breakdown():
    """
    Tests for ``Linac.get_cost_breakdown()``.
    """

    linac = Linac()
    source = SourceBasic()
    source.energy = 1e9
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    stage = StageBasic()
    stage.length = 1.0
    stage.driver_source = driver_source
    interstage = InterstagePlasmaLensBasic()
    interstage.length_dipole = 0.35
    interstage.field_dipole = 1
    interstage.nom_energy = source.energy
    linac.source = source
    linac.trackables = []
    linac.trackables.append(source)
    linac.trackables.append(stage)
    linac.trackables.append(interstage)

    cost_breakdown = linac.get_cost_breakdown()

    assert cost_breakdown[0] == 'Linac'
    assert cost_breakdown[1][0][0] == 'Source'
    assert np.isclose(cost_breakdown[1][0][1], 10000000.0, rtol=1e-15, atol=0.0)
    assert cost_breakdown[1][1][0] == 'Plasma stage'
    assert cost_breakdown[1][1][1][0][0] == 'Plasma cell'
    assert np.isclose(cost_breakdown[1][1][1][0][1], 46200.0, rtol=1e-15, atol=0.0)
    assert cost_breakdown[1][2][0] == 'Interstage'
    assert np.isclose(cost_breakdown[1][2][1], 84924.0, rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_wallplug_power():
    """
    Tests for ``Beamline.wallplug_power()``.
    """

    linac = Linac()
    assert linac.wallplug_power() is None
    
    linac.num_bunches_in_train = 2
    linac.rep_rate_trains = 1e3
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    driver_source.rep_rate_trains = 1e3
    stage = StageBasic()
    stage.driver_source = driver_source
    linac.trackables = []
    linac.trackables.append(stage)
    assert np.isclose(linac.wallplug_power(), linac.energy_usage() * linac.get_rep_rate_average(), rtol=1e-15, atol=0.0)

    
@pytest.mark.linac_unit_test
def test_get_length():
    """
    Tests for ``Beamline.get_length()``.
    """

    linac = Linac()
    source = SourceBasic()
    source.energy = 1e9
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    stage = StageBasic()
    stage.length = 1.0
    stage.driver_source = driver_source
    interstage = InterstagePlasmaLensBasic()
    interstage.length_dipole = 0.35
    interstage.field_dipole = 1
    interstage.nom_energy = source.energy
    linac.source = source

    linac.trackables = []
    assert np.isclose(linac.get_length(), 0.0, rtol=1e-15, atol=0.0)

    linac.trackables.append(source)
    linac.trackables.append(stage)
    linac.trackables.append(interstage)
    linac.trackables.append(stage)

    assert np.isclose(linac.get_length(), 1.0*2 + 2.1, rtol=1e-15, atol=0.0)
    assert np.isclose(linac.get_length(), source.get_length() + stage.get_length()*2 + interstage.get_length(), rtol=1e-15, atol=0.0)


@pytest.mark.linac_unit_test
def test_get_cost_breakdown_civil_construction():
    """
    Tests for ``Beamline.get_cost_breakdown_civil_construction()``.
    """

    linac = Linac()
    source = SourceBasic()
    source.energy = 1e9
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    stage = StageBasic()
    stage.length = 1.0
    stage.driver_source = driver_source
    interstage = InterstagePlasmaLensBasic()
    interstage.length_dipole = 0.35
    interstage.field_dipole = 1
    interstage.nom_energy = source.energy
    linac.source = source
    linac.trackables = []
    linac.trackables.append(source)
    linac.trackables.append(stage)
    linac.trackables.append(interstage)

    cost_breakdown = linac.get_cost_breakdown_civil_construction()

    assert cost_breakdown[0] == 'Civil construction'
    assert cost_breakdown[1][0][0] == 'SourceBasic'
    assert np.isclose(cost_breakdown[1][0][1], 0.0, rtol=1e-5, atol=0.0)
    assert cost_breakdown[1][1][0] == 'Plasma stage'
    assert np.isclose(cost_breakdown[1][1][1], 41207.961149999996, rtol=1e-5, atol=0.0)
    assert cost_breakdown[1][2][0] == 'InterstagePlasmaLensBasic'
    assert np.isclose(cost_breakdown[1][2][1], 86536.718415, rtol=1e-5, atol=0.0)
    
    
@pytest.mark.linac_unit_test
def test_surveys():
    """
    Tests for ``Beamline.survey_object()`` and ``Beamline.plot_survey()``.
    """

    from matplotlib import pyplot as plt

    linac = Linac()
    source = SourceBasic()
    source.energy = 1e9
    driver_source = SourceBasic()
    driver_source.energy = 1e9
    stage = StageBasic()
    stage.length = 1.0
    stage.driver_source = driver_source
    interstage = InterstagePlasmaLensBasic()
    interstage.length_dipole = 0.35
    interstage.field_dipole = 1
    interstage.beta0 = 0.01
    interstage.nom_energy = source.energy
    linac.source = source
    linac.trackables = []
    
    objs = linac.survey_object()
    assert type(objs) is list
    assert not objs  # Check for an empty list

    linac.trackables.append(source)
    linac.trackables.append(stage)
    linac.trackables.append(interstage)
    linac.trackables.append(stage)
    linac.trackables.append(interstage)
    linac.trackables.append(stage)

    objs = linac.survey_object()

    assert type(objs) is list

    source_survey = linac.trackables[0].survey_object()
    assert np.allclose(objs[0][0], source_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[0][1], source_survey[1], rtol=1e-15, atol=0.0)
    assert objs[0][2] == source_survey[2]
    assert objs[0][3] == source_survey[3]
    assert objs[0][4] == source_survey[4]

    stage1_survey = linac.trackables[1].survey_object()
    assert np.allclose(objs[1][0], stage1_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[1][1], stage1_survey[1], rtol=1e-15, atol=0.0)
    assert objs[1][2] == stage1_survey[2]
    assert objs[1][3] == stage1_survey[3]
    assert objs[1][4] == stage1_survey[4]

    interstage1_survey = linac.trackables[2].survey_object()
    assert np.allclose(objs[2][0], interstage1_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[2][1], interstage1_survey[1], rtol=1e-15, atol=0.0)
    assert objs[2][2] == interstage1_survey[2]
    assert objs[2][3] == interstage1_survey[3]
    assert objs[2][4] == interstage1_survey[4]

    stage2_survey = linac.trackables[3].survey_object()
    assert np.allclose(objs[3][0], stage2_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[3][1], stage2_survey[1], rtol=1e-15, atol=0.0)
    assert objs[3][2] == stage2_survey[2]
    assert objs[3][3] == stage2_survey[3]
    assert objs[3][4] == stage2_survey[4]

    interstage2_survey = linac.trackables[4].survey_object()
    assert np.allclose(objs[4][0], interstage2_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[4][1], interstage2_survey[1], rtol=1e-15, atol=0.0)
    assert objs[4][2] == interstage2_survey[2]
    assert objs[4][3] == interstage2_survey[3]
    assert objs[4][4] == interstage2_survey[4]

    stage3_survey = linac.trackables[5].survey_object()
    assert np.allclose(objs[5][0], stage3_survey[0], rtol=1e-15, atol=0.0)
    assert np.allclose(objs[5][1], stage3_survey[1], rtol=1e-15, atol=0.0)
    assert objs[5][2] == stage3_survey[2]
    assert objs[5][3] == stage3_survey[3]
    assert objs[5][4] == stage3_survey[4]
    
    plt.ion()
    linac.plot_survey()
