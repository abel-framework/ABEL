# -*- coding: utf-8 -*-
"""
ABEL : Stage Geometry calculation tests
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, B.Chen, O.G. Finnerud,
D. Kallvik, E. Hørlyk, K.N. Sjobak, E.Adli, University of Oslo

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

import os
import pytest

from abel import *
from abel.classes.stage.stage import VariablesOverspecifiedError


#Helpers
def printStuff(stage):
    print("length / flattop                  =",stage.length, stage.length_flattop, '[m]')
    if stage.upramp is not None:
        print("  length_upramp                   =", stage.upramp.length, '[m]')
    if stage.downramp is not None:
        print("  length_downramp                 =", stage.downramp.length, '[m]')

    print("nom_energy_gain / flattop         =",stage.nom_energy_gain, stage.nom_energy_gain_flattop, '[eV]')
    if stage.upramp is not None:
        print("  nom_energy_gain_upramp          =", stage.upramp.nom_energy_gain, '[m]')
    if stage.downramp is not None:
        print("  nom_energy_gain_downramp        =", stage.downramp.nom_energy_gain, '[m]')

    print("nom_accel_gradient / flattop      =",stage.nom_accel_gradient, stage.nom_accel_gradient_flattop, '[eV/m]')
    if stage.upramp is not None:
        print("  nom_accel_gradient_upramp       =", stage.upramp.nom_accel_gradient, '[m]')
    if stage.downramp is not None:
        print("  nom_accel_gradient_downramp     =", stage.downramp.nom_accel_gradient, '[m]')

    print()

def printStuff_internal(stage):
    stage._printLengthEnergyGradient_internal()
    print()

@pytest.mark.stageGeometry
def testStageGeom_basic():
    """
    """
    stageTest = StageBasic()
    stageTest.doVerbosePrint_debug = True

    #Mess around with length and length_flattop
    print("Set length:")
    stageTest.length = 10
    printStuff(stageTest)
    assert stageTest.length == 10
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 10
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None
    
    print("Change length:")
    stageTest.length = 15
    printStuff(stageTest)
    assert stageTest.length == 15
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 15
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None

    print("Also set gradient - this implicitly specifies nom_energy_gain:")
    stageTest.nom_accel_gradient = 10
    printStuff_internal(stageTest)
    printStuff(stageTest)
    assert stageTest.length == 15
    assert stageTest.nom_energy_gain == 10*15
    assert stageTest.nom_accel_gradient == 10
    assert stageTest.length_flattop == 15
    assert stageTest.nom_energy_gain_flattop == 10*15
    assert abs(stageTest.nom_accel_gradient_flattop - 10.0) < 1e-10

    #Try to explicitly set or delete the calculated and thus known values
    #This should crash the calculation
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain = 42
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain = None

    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain_flattop = 42
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain_flattop = None

    with pytest.raises(VariablesOverspecifiedError):
        stageTest.length_flattop = 2
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.length_flattop = None

    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_accel_gradient_flattop = 2
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_accel_gradient_flattop = None

    print()
    print("** setting things in different order **")
    print()
    
    print("Delete length and gradient:")
    stageTest.length = None
    stageTest.nom_accel_gradient = None
    printStuff(stageTest)
    assert stageTest.length == None
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == None
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None
    
    print("Set length_flattop (no upramp/downramp):")
    stageTest.length_flattop = 5
    printStuff(stageTest)
    assert stageTest.length == 5
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 5
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None
    
    print("Also set gradient_flattop:")
    stageTest.nom_accel_gradient_flattop = 10
    #printStuff_internal(stageTest)
    printStuff(stageTest)
    assert stageTest.length == 5
    assert stageTest.nom_energy_gain == 5*10
    assert stageTest.nom_accel_gradient == 10
    assert stageTest.length_flattop == 5
    assert stageTest.nom_energy_gain_flattop == 5*10
    assert stageTest.nom_accel_gradient_flattop == 10
    
    #Try to explicitly set or delete the calculated and thus known values
    #This should crash the calculation
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain = 42
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain = None
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain_flattop = 42
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_energy_gain_flattop = None
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.length = 2
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.length = None
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_accel_gradient = 2
    with pytest.raises(VariablesOverspecifiedError):
        stageTest.nom_accel_gradient = None

    print()
    print("** Flip around length/energy_gain/accelGradient **")
    print()
    printStuff_internal(stageTest)
    
    print("length_flattop out, nom_energy_gain_flattop in")
    stageTest.length_flattop = None
    #printStuff(stageTest)
    stageTest.nom_energy_gain_flattop = 20
    printStuff(stageTest)
    
    print("nom_accel_gradient_flattop out, length_flattop in:")
    stageTest.nom_accel_gradient_flattop = None
    #stageTest.length = 2
    stageTest.length_flattop = 2
    printStuff(stageTest)
    
    print("nom_energy_gain_flattop out, nom_accel_gradient_flattop in:")
    stageTest.nom_energy_gain_flattop = None
    #stageTest.length = 2
    stageTest.nom_accel_gradient_flattop = 2
    printStuff(stageTest)

    #Trigger a test failure and printout
    assert False

@pytest.mark.stageGeometry
def testStageGeom_ramps():
    """
    """
    stageTest_L = StageBasic()
    stageTest_L.upramp = stageTest_L.__class__()
    stageTest_L.downramp = stageTest_L.__class__()
