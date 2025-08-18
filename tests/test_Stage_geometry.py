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
import scipy.constants as SI
import numpy as np

from abel import *
from abel.classes.stage.stage import VariablesOverspecifiedError, VariablesOutOfRangeError, StageError

#Helper function - pretty printed debug outputs of current stage status
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

#Helper function - pretty printed debug outputs of internal variables
def printStuff_internal(stage):
    stage._printLengthEnergyGradient_internal()
    print()

@pytest.mark.stageGeometry
def test_StageGeom_basic():
    "Test the basic functionality of stage geometry (length/energy gain/gradient); no ramps."
    stageTest = StageBasic()
    stageTest.doVerbosePrint_debug = True

    assert stageTest.length == None
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == None
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None

    #Mess around with length and length_flattop
    print("Set length:")
    stageTest.length = 10
    printStuff(stageTest)

    assert stageTest.length == 10 #Explicit
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 10
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None
    
    print("Change length:")
    stageTest.length = 15
    printStuff(stageTest)
    assert stageTest.length == 15 #Explicit
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 15
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None

    print("Also set gradient - this implicitly specifies nom_energy_gain:")
    stageTest.nom_accel_gradient = 10
    printStuff_internal(stageTest)
    printStuff(stageTest)
    assert stageTest.length == 15 #Explicit
    assert stageTest.nom_energy_gain == 10*15
    assert stageTest.nom_accel_gradient == 10 #Explicit
    assert stageTest.length_flattop == 15
    assert stageTest.nom_energy_gain_flattop == 10*15
    assert np.allclose(stageTest.nom_accel_gradient_flattop, 10.0)

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
    assert stageTest.length_flattop == 5 #Explicit
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None
    
    print("Also set gradient_flattop:")
    stageTest.nom_accel_gradient_flattop = 10
    #printStuff_internal(stageTest)
    printStuff(stageTest)
    assert stageTest.length == 5
    assert stageTest.nom_energy_gain == 5*10
    assert stageTest.nom_accel_gradient == 10
    assert stageTest.length_flattop == 5 #Explicit
    assert stageTest.nom_energy_gain_flattop == 5*10
    assert stageTest.nom_accel_gradient_flattop == 10 #Explicit
    
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
    printStuff(stageTest)
    
    assert stageTest.length == None
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == 10
    assert stageTest.length_flattop == None
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == 10 #Explicit

    stageTest.nom_energy_gain_flattop = 20
    printStuff(stageTest)

    assert np.allclose(stageTest.length, 20/10)
    assert stageTest.nom_energy_gain == 20
    assert stageTest.nom_accel_gradient == 10
    assert np.allclose(stageTest.length_flattop, 20/10)
    assert stageTest.nom_energy_gain_flattop == 20 #Explicit
    assert stageTest.nom_accel_gradient_flattop == 10 #Explicit
        
    print("nom_accel_gradient_flattop out, length_flattop in:")
    stageTest.nom_accel_gradient_flattop = None
    printStuff(stageTest)

    assert stageTest.length == None
    assert stageTest.nom_energy_gain == 20
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == None
    assert stageTest.nom_energy_gain_flattop == 20 #Explicit
    assert stageTest.nom_accel_gradient_flattop == None

    #stageTest.length = 2
    stageTest.length_flattop = 2
    printStuff(stageTest)
    
    assert stageTest.length == 2
    assert stageTest.nom_energy_gain == 20
    assert np.allclose(stageTest.nom_accel_gradient, 20/2)
    assert stageTest.length_flattop == 2 #Explicit
    assert stageTest.nom_energy_gain_flattop == 20 #Explicit
    assert np.allclose(stageTest.nom_accel_gradient_flattop, 20/2)

    print("nom_energy_gain_flattop out, nom_accel_gradient_flattop in:")
    stageTest.nom_energy_gain_flattop = None
    printStuff(stageTest)

    assert stageTest.length == 2
    assert stageTest.nom_energy_gain == None
    assert stageTest.nom_accel_gradient == None
    assert stageTest.length_flattop == 2 #Explicit
    assert stageTest.nom_energy_gain_flattop == None
    assert stageTest.nom_accel_gradient_flattop == None

    #stageTest.length = 2
    stageTest.nom_accel_gradient_flattop = 2
    printStuff(stageTest)

    assert stageTest.length == 2
    assert stageTest.nom_energy_gain == 2*2
    assert stageTest.nom_accel_gradient == 2
    assert stageTest.length_flattop == 2 #Explicit
    assert stageTest.nom_energy_gain_flattop == 2*2
    assert stageTest.nom_accel_gradient_flattop == 2 # Explicit

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_ramps():
    "Test the basic functionality of stage geometry (length/energy gain/gradient); with ramps."
    
    stageTest_L = StageBasic()
    stageTest_L.upramp = stageTest_L.__class__()
    stageTest_L.downramp = stageTest_L.__class__()

    printStuff(stageTest_L)
    printStuff_internal(stageTest_L)
    printStuff_internal(stageTest_L.upramp)
    printStuff_internal(stageTest_L.downramp)

    stageTest_L.doVerbosePrint_debug = True
    stageTest_L.upramp.doVerbosePrint_debug = True
    stageTest_L.downramp.doVerbosePrint_debug = True

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == None
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == None
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Set upramp length, show overall and upramp:")
    stageTest_L.upramp.length = 1
    printStuff(stageTest_L)
    printStuff(stageTest_L.upramp)

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set total length")
    stageTest_L.length = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10 #Explicit
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp - now length is fully defined")
    stageTest_L.downramp.length = 2
    printStuff(stageTest_L)

    assert stageTest_L.length == 10 #Explicit
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10-1-2
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting total to flattop - still fully defined")
    stageTest_L.length = None
    printStuff(stageTest_L)

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.length_flattop = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set energy gain - should now have overall gradient")
    stageTest_L.nom_energy_gain = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 10 #Explicit
    assert np.allclose(stageTest_L.nom_accel_gradient, 10/(10+1+2))
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting energy gain to setting gradient")
    stageTest_L.nom_energy_gain = None
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.nom_accel_gradient = 2
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Set the flattop energy gain, should define the flattop gradient")
    stageTest_L.nom_energy_gain_flattop = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 10/10)

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting flattop energy gain to setting flattop gradient")
    stageTest_L.nom_energy_gain_flattop    = None
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.nom_accel_gradient_flattop = 0.5
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_energy_gain_flattop, 10*0.5)
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 0.5) #Explicit

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Add in the upramp energy gain  - now we should be fully defined")
    stageTest_L.upramp.nom_energy_gain = 1
    printStuff(stageTest_L)
    printStuff(stageTest_L.upramp)
    printStuff(stageTest_L.downramp)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_energy_gain_flattop, 10*0.5)
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 0.5) #Explicit

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == 1 #Explicit
    assert np.allclose(stageTest_L.upramp.nom_accel_gradient, 1/1)
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == 1
    assert np.allclose(stageTest_L.upramp.nom_accel_gradient_flattop, 1/1)

    assert stageTest_L.downramp.length == 2 #Explicit
    assert np.allclose(stageTest_L.downramp.nom_energy_gain, ( 2*(10+1+2) - 1 - 10*0.5))
    assert np.allclose(stageTest_L.downramp.nom_accel_gradient, ( 2*(10+1+2) - 1 - 10*0.5)/2 )
    assert stageTest_L.downramp.length_flattop == 2
    assert np.allclose(stageTest_L.downramp.nom_energy_gain_flattop, ( 2*(10+1+2) - 1 - 10*0.5))
    assert np.allclose(stageTest_L.downramp.nom_accel_gradient_flattop, ( 2*(10+1+2) - 1 - 10*0.5)/2 )

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_PlasmaRamps():
    """
    Test the basic functionality of stage geometry 
    (length/energy gain/gradient); with ``PlasmaRamp`` ramps.
    """
    
    
    stageTest_L = StageBasic()
    stageTest_L.upramp = PlasmaRamp()
    stageTest_L.downramp = PlasmaRamp()

    printStuff(stageTest_L)
    printStuff_internal(stageTest_L)
    printStuff_internal(stageTest_L.upramp)
    printStuff_internal(stageTest_L.downramp)

    stageTest_L.doVerbosePrint_debug = True
    stageTest_L.upramp.doVerbosePrint_debug = True
    stageTest_L.downramp.doVerbosePrint_debug = True

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == None
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == None
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Set upramp length, show overall and upramp:")
    stageTest_L.upramp.length = 1
    printStuff(stageTest_L)
    printStuff(stageTest_L.upramp)

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set total length")
    stageTest_L.length = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10 #Explicit
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == None
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == None
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp - now length is fully defined")
    stageTest_L.downramp.length = 2
    printStuff(stageTest_L)

    assert stageTest_L.length == 10 #Explicit
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10-1-2
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting total to flattop - still fully defined")
    stageTest_L.length = None
    printStuff(stageTest_L)

    assert stageTest_L.length == None
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == None
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.length_flattop = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Also set energy gain - should now have overall gradient")
    stageTest_L.nom_energy_gain = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 10 #Explicit
    assert np.allclose(stageTest_L.nom_accel_gradient, 10/(10+1+2))
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting energy gain to setting gradient")
    stageTest_L.nom_energy_gain = None
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == None
    assert stageTest_L.nom_accel_gradient == None
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.nom_accel_gradient = 2
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Set the flattop energy gain, should define the flattop gradient")
    stageTest_L.nom_energy_gain_flattop = 10
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 10/10)

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting flattop energy gain to setting flattop gradient")
    stageTest_L.nom_energy_gain_flattop    = None
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert stageTest_L.nom_energy_gain_flattop == None
    assert stageTest_L.nom_accel_gradient_flattop == None

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    stageTest_L.nom_accel_gradient_flattop = 0.5
    printStuff(stageTest_L)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_energy_gain_flattop, 10*0.5)
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 0.5) #Explicit

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == None
    assert stageTest_L.upramp.nom_accel_gradient == None
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == None
    assert stageTest_L.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L.downramp.length == 2 #Explicit
    assert stageTest_L.downramp.nom_energy_gain == None
    assert stageTest_L.downramp.nom_accel_gradient == None
    assert stageTest_L.downramp.length_flattop == 2
    assert stageTest_L.downramp.nom_energy_gain_flattop == None
    assert stageTest_L.downramp.nom_accel_gradient_flattop == None

    print("Add in the upramp energy gain  - now we should be fully defined")
    stageTest_L.upramp.nom_energy_gain = 1
    printStuff(stageTest_L)
    printStuff(stageTest_L.upramp)
    printStuff(stageTest_L.downramp)

    assert stageTest_L.length == 10+1+2
    assert stageTest_L.nom_energy_gain == 2*(10+1+2)
    assert stageTest_L.nom_accel_gradient == 2 #Explicit
    assert stageTest_L.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_L.nom_energy_gain_flattop, 10*0.5)
    assert np.allclose(stageTest_L.nom_accel_gradient_flattop, 0.5) #Explicit

    assert stageTest_L.upramp.length == 1 #Explicit
    assert stageTest_L.upramp.nom_energy_gain == 1 #Explicit
    assert np.allclose(stageTest_L.upramp.nom_accel_gradient, 1/1)
    assert stageTest_L.upramp.length_flattop == 1
    assert stageTest_L.upramp.nom_energy_gain_flattop == 1
    assert np.allclose(stageTest_L.upramp.nom_accel_gradient_flattop, 1/1)

    assert stageTest_L.downramp.length == 2 #Explicit
    assert np.allclose(stageTest_L.downramp.nom_energy_gain, ( 2*(10+1+2) - 1 - 10*0.5))
    assert np.allclose(stageTest_L.downramp.nom_accel_gradient, ( 2*(10+1+2) - 1 - 10*0.5)/2 )
    assert stageTest_L.downramp.length_flattop == 2
    assert np.allclose(stageTest_L.downramp.nom_energy_gain_flattop, ( 2*(10+1+2) - 1 - 10*0.5))
    assert np.allclose(stageTest_L.downramp.nom_accel_gradient_flattop, ( 2*(10+1+2) - 1 - 10*0.5)/2 )

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_eGain():
    "Test of energy gain"

    stageTest_E = StageBasic()
    stageTest_E.upramp = stageTest_E.__class__()
    stageTest_E.downramp = stageTest_E.__class__()

    stageTest_E.doVerbosePrint_debug = True
    stageTest_E.upramp.doVerbosePrint_debug = True
    stageTest_E.downramp.doVerbosePrint_debug = True

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == None
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == None
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Set upramp energy gain:")
    stageTest_E.upramp.nom_energy_gain = 1
    printStuff(stageTest_E)
    #printStuff(stageTest_G.upramp)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set total")
    #stageTest_E.upramp.nom_energy_gain = 1 #Overwrite upramp's total, as a test
    stageTest_E.nom_energy_gain = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10 #Explicit
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp")
    stageTest_E.downramp.nom_energy_gain = 2
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10 #Explicit
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10-1-2
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting total to flattop")
    stageTest_E.nom_energy_gain = None
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    stageTest_E.nom_energy_gain_flattop = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set overall length - should get overall gradient")
    stageTest_E.length = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == 10 #Explicit
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert np.allclose(stageTest_E.nom_accel_gradient, (10+1+2)/10)
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Unset length, set gradient")
    stageTest_E.length = None
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    stageTest_E.nom_accel_gradient = 20
    printStuff(stageTest_E)

    assert np.allclose(stageTest_E.length, (10+1+2)/20)
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == 20 # Explicit
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_eGain_PlasmaRamps():
    "Test of energy gain"

    stageTest_E = StageBasic()
    stageTest_E.upramp = PlasmaRamp()
    stageTest_E.downramp = PlasmaRamp()
    #printStuff(stageTest_G)

    stageTest_E.doVerbosePrint_debug = True
    stageTest_E.upramp.doVerbosePrint_debug = True
    stageTest_E.downramp.doVerbosePrint_debug = True

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == None
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == None
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Set upramp energy gain:")
    stageTest_E.upramp.nom_energy_gain = 1
    printStuff(stageTest_E)
    #printStuff(stageTest_G.upramp)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set total")
    #stageTest_E.upramp.nom_energy_gain = 1 #Overwrite upramp's total, as a test
    stageTest_E.nom_energy_gain = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10 #Explicit
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == None
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == None
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp")
    stageTest_E.downramp.nom_energy_gain = 2
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10 #Explicit
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10-1-2
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Switch from setting total to flattop")
    stageTest_E.nom_energy_gain = None
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == None
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == None
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    stageTest_E.nom_energy_gain_flattop = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Also set overall length - should get overall gradient")
    stageTest_E.length = 10
    printStuff(stageTest_E)

    assert stageTest_E.length == 10 #Explicit
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert np.allclose(stageTest_E.nom_accel_gradient, (10+1+2)/10)
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    print("Unset length, set gradient")
    stageTest_E.length = None
    printStuff(stageTest_E)

    assert stageTest_E.length == None
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == None
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    stageTest_E.nom_accel_gradient = 20
    printStuff(stageTest_E)

    assert np.allclose(stageTest_E.length, (10+1+2)/20)
    assert stageTest_E.nom_energy_gain == 10+1+2
    assert stageTest_E.nom_accel_gradient == 20 # Explicit
    assert stageTest_E.length_flattop == None
    assert stageTest_E.nom_energy_gain_flattop == 10 #Explicit
    assert stageTest_E.nom_accel_gradient_flattop == None

    assert stageTest_E.upramp.length == None
    assert stageTest_E.upramp.nom_energy_gain == 1 #Explicit
    assert stageTest_E.upramp.nom_accel_gradient == None
    assert stageTest_E.upramp.length_flattop == None
    assert stageTest_E.upramp.nom_energy_gain_flattop == 1
    assert stageTest_E.upramp.nom_accel_gradient_flattop == None

    assert stageTest_E.downramp.length == None
    assert stageTest_E.downramp.nom_energy_gain == 2 #Explicit
    assert stageTest_E.downramp.nom_accel_gradient == None
    assert stageTest_E.downramp.length_flattop == None
    assert stageTest_E.downramp.nom_energy_gain_flattop == 2
    assert stageTest_E.downramp.nom_accel_gradient_flattop == None

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_gradient():
    "Test of gradient"

    stageTest_G = StageBasic()
    stageTest_G.upramp = stageTest_G.__class__()
    stageTest_G.downramp = stageTest_G.__class__()

    stageTest_G.doVerbosePrint_debug = True
    stageTest_G.upramp.doVerbosePrint_debug = True
    stageTest_G.downramp.doVerbosePrint_debug = True

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == None
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == None

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Set upramp, show overall and upramp:")
    stageTest_G.upramp.nom_accel_gradient = 1
    printStuff(stageTest_G)
    printStuff(stageTest_G.upramp) #Should show some of the same as above

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Also set total")
    stageTest_G.upramp.nom_accel_gradient = 1 # Overwrite upramps total, as a test
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    stageTest_G.nom_accel_gradient = 10
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp")
    stageTest_G.downramp.nom_accel_gradient = 2
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Also set lengths - this should give energies")
    stageTest_G.length = 10
    stageTest_G.length_flattop = 8.5
    stageTest_G.upramp.length = 1
    printStuff(stageTest_G)

    assert stageTest_G.length == 10 #Explicit
    assert stageTest_G.nom_energy_gain == 10*10
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert np.allclose(stageTest_G.length_flattop, 8.5) #Explicit
    assert np.allclose(stageTest_G.nom_energy_gain_flattop, (10*10 - 1*1 - (10-8.5-1)*2))
    assert np.allclose(stageTest_G.nom_accel_gradient_flattop, (10*10 - 1*1 - (10-8.5-1)*2)/8.5)

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert np.allclose(stageTest_G.downramp.length, (10-8.5-1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain, (10-8.5-1)*2)
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert np.allclose(stageTest_G.downramp.length_flattop, (10-8.5-1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain_flattop, (10-8.5-1)*2)
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Switch from setting total to flattop")
    stageTest_G.length = None
    stageTest_G.length_flattop = 10
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Also set energy gain - should now have overall gradient")
    stageTest_G.nom_energy_gain = 1000
    printStuff(stageTest_G)

    assert np.allclose(stageTest_G.length, 1000/10)
    assert stageTest_G.nom_energy_gain == 1000 # Explicit
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_G.nom_energy_gain_flattop, (1000 - 1*1 - (1000/10 - 10 - 1)*2))
    assert np.allclose(stageTest_G.nom_accel_gradient_flattop, (1000 - 1*1 - (1000/10 - 10 - 1)*2)/10)

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert np.allclose(stageTest_G.downramp.length, (1000/10 - 10 - 1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain, (1000/10 - 10 - 1)*2)
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert np.allclose(stageTest_G.downramp.length_flattop, (1000/10 - 10 - 1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain_flattop, (1000/10 - 10 - 1)*2)
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Switch from setting energy gain to setting gradient")
    stageTest_G.nom_energy_gain = None
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    stageTest_G.nom_accel_gradient = 2
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_gradient_PlasmaRamps():
    "Test of gradient"

    stageTest_G = StageBasic()
    stageTest_G.upramp = PlasmaRamp()
    stageTest_G.downramp = PlasmaRamp()

    stageTest_G.doVerbosePrint_debug = True
    stageTest_G.upramp.doVerbosePrint_debug = True
    stageTest_G.downramp.doVerbosePrint_debug = True

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == None
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == None

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Set upramp, show overall and upramp:")
    stageTest_G.upramp.nom_accel_gradient = 1
    printStuff(stageTest_G)
    printStuff(stageTest_G.upramp) #Should show some of the same as above

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Also set total")
    stageTest_G.upramp.nom_accel_gradient = 1 # Overwrite upramps total, as a test
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == None
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    stageTest_G.nom_accel_gradient = 10
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == None
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == None

    print("Also set downramp")
    stageTest_G.downramp.nom_accel_gradient = 2
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == None
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == None
    assert stageTest_G.upramp.nom_energy_gain == None
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == None
    assert stageTest_G.upramp.nom_energy_gain_flattop == None
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Also set lengths - this should give energies")
    stageTest_G.length = 10
    stageTest_G.length_flattop = 8.5
    stageTest_G.upramp.length = 1
    printStuff(stageTest_G)

    assert stageTest_G.length == 10 #Explicit
    assert stageTest_G.nom_energy_gain == 10*10
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert np.allclose(stageTest_G.length_flattop, 8.5) #Explicit
    assert np.allclose(stageTest_G.nom_energy_gain_flattop, (10*10 - 1*1 - (10-8.5-1)*2))
    assert np.allclose(stageTest_G.nom_accel_gradient_flattop, (10*10 - 1*1 - (10-8.5-1)*2)/8.5)

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert np.allclose(stageTest_G.downramp.length, (10-8.5-1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain, (10-8.5-1)*2)
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert np.allclose(stageTest_G.downramp.length_flattop, (10-8.5-1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain_flattop, (10-8.5-1)*2)
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Switch from setting total to flattop")
    stageTest_G.length = None
    stageTest_G.length_flattop = 10
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Also set energy gain - should now have overall gradient")
    stageTest_G.nom_energy_gain = 1000
    printStuff(stageTest_G)

    assert np.allclose(stageTest_G.length, 1000/10)
    assert stageTest_G.nom_energy_gain == 1000 # Explicit
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert np.allclose(stageTest_G.nom_energy_gain_flattop, (1000 - 1*1 - (1000/10 - 10 - 1)*2))
    assert np.allclose(stageTest_G.nom_accel_gradient_flattop, (1000 - 1*1 - (1000/10 - 10 - 1)*2)/10)

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert np.allclose(stageTest_G.downramp.length, (1000/10 - 10 - 1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain, (1000/10 - 10 - 1)*2)
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert np.allclose(stageTest_G.downramp.length_flattop, (1000/10 - 10 - 1))
    assert np.allclose(stageTest_G.downramp.nom_energy_gain_flattop, (1000/10 - 10 - 1)*2)
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    print("Switch from setting energy gain to setting gradient")
    stageTest_G.nom_energy_gain = None
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 10 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    stageTest_G.nom_accel_gradient = 2
    printStuff(stageTest_G)

    assert stageTest_G.length == None
    assert stageTest_G.nom_energy_gain == None
    assert stageTest_G.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.length_flattop == 10 #Explicit
    assert stageTest_G.nom_energy_gain_flattop == None
    assert stageTest_G.nom_accel_gradient_flattop == None

    assert stageTest_G.upramp.length == 1 #Explicit
    assert stageTest_G.upramp.nom_energy_gain == 1*1
    assert stageTest_G.upramp.nom_accel_gradient == 1 #Explicit
    assert stageTest_G.upramp.length_flattop == 1
    assert stageTest_G.upramp.nom_energy_gain_flattop == 1*1
    assert stageTest_G.upramp.nom_accel_gradient_flattop == 1

    assert stageTest_G.downramp.length == None
    assert stageTest_G.downramp.nom_energy_gain == None
    assert stageTest_G.downramp.nom_accel_gradient == 2 #Explicit
    assert stageTest_G.downramp.length_flattop == None
    assert stageTest_G.downramp.nom_energy_gain_flattop == None
    assert stageTest_G.downramp.nom_accel_gradient_flattop == 2

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_sanityCheckLengths1():
    "Testing of Stage.sanityCheckLengths logic without ramps"

    stageTest_L1 = StageBasic()

    printStuff(stageTest_L1)
    printStuff_internal(stageTest_L1)

    stageTest_L1.doVerbosePrint_debug = True
    
    assert stageTest_L1.length == None
    assert stageTest_L1.nom_energy_gain == None
    assert stageTest_L1.nom_accel_gradient == None
    assert stageTest_L1.length_flattop == None
    assert stageTest_L1.nom_energy_gain_flattop == None
    assert stageTest_L1.nom_accel_gradient_flattop == None

    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L1.length = -1
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L1.length_flattop = -1

    #The error is caught before setting
    assert stageTest_L1.length == None
    assert stageTest_L1.nom_energy_gain == None
    assert stageTest_L1.nom_accel_gradient == None
    assert stageTest_L1.length_flattop == None
    assert stageTest_L1.nom_energy_gain_flattop == None
    assert stageTest_L1.nom_accel_gradient_flattop == None

    #Disable sanity checking
    stageTest_L1.sanityCheckLengths = False
    stageTest_L1.length = -1

    assert stageTest_L1.length == -1 #Explicit
    assert stageTest_L1.nom_energy_gain == None
    assert stageTest_L1.nom_accel_gradient == None
    assert stageTest_L1.length_flattop == -1
    assert stageTest_L1.nom_energy_gain_flattop == None
    assert stageTest_L1.nom_accel_gradient_flattop == None

    stageTest_L1.length = None
    stageTest_L1.length_flattop = -1

    assert stageTest_L1.length == -1
    assert stageTest_L1.nom_energy_gain == None
    assert stageTest_L1.nom_accel_gradient == None
    assert stageTest_L1.length_flattop == -1 # Explicit
    assert stageTest_L1.nom_energy_gain_flattop == None
    assert stageTest_L1.nom_accel_gradient_flattop == None

    #Reset and test calculations
    stageTest_L1.length_flattop = None
    stageTest_L1.sanityCheckLengths = True

    stageTest_L1.nom_energy_gain = -1
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L1.nom_accel_gradient = 1
    stageTest_L1.sanityCheckLengths = False
    stageTest_L1.nom_accel_gradient = 1

    assert stageTest_L1.length == -1*1
    assert stageTest_L1.nom_energy_gain == -1 #Explicit
    assert stageTest_L1.nom_accel_gradient == 1 #Explicit
    assert stageTest_L1.length_flattop == 1*-1
    assert stageTest_L1.nom_energy_gain_flattop == -1
    assert stageTest_L1.nom_accel_gradient_flattop == 1

    #Reset and test other calculation
    stageTest_L1.nom_energy_gain = None
    stageTest_L1.nom_accel_gradient = None
    stageTest_L1.sanityCheckLengths = True

    stageTest_L1.nom_accel_gradient = -1
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L1.nom_energy_gain = 1
    stageTest_L1.sanityCheckLengths = False
    stageTest_L1.nom_energy_gain = 1

    assert stageTest_L1.length == 1*-1
    assert stageTest_L1.nom_energy_gain == 1 #Explicit
    assert stageTest_L1.nom_accel_gradient == -1 #Explicit
    assert stageTest_L1.length_flattop == 1*-1
    assert stageTest_L1.nom_energy_gain_flattop == 1
    assert stageTest_L1.nom_accel_gradient_flattop == -1

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_sanityCheckLengths2():
    "Testing of Stage.sanityCheckLengths logic with ramps"

    stageTest_L2 = StageBasic()
    stageTest_L2.upramp = stageTest_L2.__class__()
    stageTest_L2.downramp = stageTest_L2.__class__()

    printStuff(stageTest_L2)
    printStuff_internal(stageTest_L2)
    printStuff_internal(stageTest_L2.upramp)
    printStuff_internal(stageTest_L2.downramp)

    stageTest_L2.doVerbosePrint_debug = True
    stageTest_L2.upramp.doVerbosePrint_debug = True
    stageTest_L2.downramp.doVerbosePrint_debug = True

    assert stageTest_L2.length == None
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Try to explicitly set negative values
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length = -1
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = -1

    #Nothing has changed
    assert stageTest_L2.length == None
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Implicitly make the main flattop negative
    stageTest_L2.upramp.length = 3
    stageTest_L2.downramp.length = 3
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length = 5
    printStuff(stageTest_L2)

    assert stageTest_L2.length == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Would be negative
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 # Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Disable sanity checking and retry
    stageTest_L2.sanityCheckLengths = False
    stageTest_L2.length = 5

    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == 5-3-3
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Undo it
    stageTest_L2.length = None
    stageTest_L2.sanityCheckLengths = True

    #Implicitly make the downramp length negative
    stageTest_L2.downramp.length = None
    stageTest_L2.length = 5
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    
    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None #Would be negative
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Disable sanity checking and retry
    # main stage not enough...
    stageTest_L2.sanityCheckLengths = False
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    # Error happens in downramp - disable here
    stageTest_L2.sanityCheckLengths = True
    stageTest_L2.downramp.sanityCheckLengths = False
    stageTest_L2.length_flattop = 3

    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == 3 #Explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 5-3-3 #Calculated negative
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 5-3-3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Undo it
    stageTest_L2.length_flattop = None
    stageTest_L2.downramp.sanityCheckLengths = True

    #Implicitly make the upramp length negative
    stageTest_L2.upramp.length = None
    stageTest_L2.downramp.length = 3
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    
    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None #Would be negative
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_sanityCheckLengths2_PlasmaRamps():
    "Testing of Stage.sanityCheckLengths logic with ``PlasmaRamp``ramps."

    stageTest_L2 = StageBasic()
    stageTest_L2.upramp = PlasmaRamp()
    stageTest_L2.downramp = PlasmaRamp()

    printStuff(stageTest_L2)
    printStuff_internal(stageTest_L2)
    printStuff_internal(stageTest_L2.upramp)
    printStuff_internal(stageTest_L2.downramp)

    stageTest_L2.doVerbosePrint_debug = True
    stageTest_L2.upramp.doVerbosePrint_debug = True
    stageTest_L2.downramp.doVerbosePrint_debug = True

    assert stageTest_L2.length == None
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Try to explicitly set negative values
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length = -1
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = -1

    #Nothing has changed
    assert stageTest_L2.length == None
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Implicitly make the main flattop negative
    stageTest_L2.upramp.length = 3
    stageTest_L2.downramp.length = 3
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length = 5
    printStuff(stageTest_L2)

    assert stageTest_L2.length == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Would be negative
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 # Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Disable sanity checking and retry
    stageTest_L2.sanityCheckLengths = False
    stageTest_L2.length = 5

    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == 5-3-3
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Undo it
    stageTest_L2.length = None
    stageTest_L2.sanityCheckLengths = True

    #Implicitly make the downramp length negative
    stageTest_L2.downramp.length = None
    stageTest_L2.length = 5
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    
    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == None #Would be negative
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == None
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Disable sanity checking and retry
    # main stage not enough...
    stageTest_L2.sanityCheckLengths = False
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    # Error happens in downramp - disable here
    stageTest_L2.sanityCheckLengths = True
    stageTest_L2.downramp.sanityCheckLengths = False
    stageTest_L2.length_flattop = 3

    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == 3 #Explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == 3 #Explicit
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == 3
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 5-3-3 #Calculated negative
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 5-3-3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Undo it
    stageTest_L2.length_flattop = None
    stageTest_L2.downramp.sanityCheckLengths = True

    #Implicitly make the upramp length negative
    stageTest_L2.upramp.length = None
    stageTest_L2.downramp.length = 3
    with pytest.raises(VariablesOutOfRangeError):
        stageTest_L2.length_flattop = 3
    
    assert stageTest_L2.length == 5 #Explicit
    assert stageTest_L2.nom_energy_gain == None
    assert stageTest_L2.nom_accel_gradient == None
    assert stageTest_L2.length_flattop == None #Unwound explicit
    assert stageTest_L2.nom_energy_gain_flattop == None
    assert stageTest_L2.nom_accel_gradient_flattop == None

    assert stageTest_L2.upramp.length == None #Would be negative
    assert stageTest_L2.upramp.nom_energy_gain == None
    assert stageTest_L2.upramp.nom_accel_gradient == None
    assert stageTest_L2.upramp.length_flattop == None
    assert stageTest_L2.upramp.nom_energy_gain_flattop == None
    assert stageTest_L2.upramp.nom_accel_gradient_flattop == None

    assert stageTest_L2.downramp.length == 3 #Explicit
    assert stageTest_L2.downramp.nom_energy_gain == None
    assert stageTest_L2.downramp.nom_accel_gradient == None
    assert stageTest_L2.downramp.length_flattop == 3
    assert stageTest_L2.downramp.nom_energy_gain_flattop == None
    assert stageTest_L2.downramp.nom_accel_gradient_flattop == None

    #Trigger a test failure and printout
    #assert False


@pytest.mark.stageGeometry
def test_StageGeom_ramp_beta_mag_PlasmaRamps():
    "Testing ``Stage.ramp_beta_mag`` and ``PlasmaRamp.ramp_beta_mag``."

    stage = StageBasic()
    stage.ramp_beta_mag = 10.0
    stage.upramp = PlasmaRamp()
    stage.downramp = PlasmaRamp()

    assert np.isclose(stage.ramp_beta_mag, 10.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.ramp_beta_mag, 10.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.ramp_beta_mag, 10.0, rtol=1e-15, atol=0.0)

    stage.upramp.ramp_beta_mag = 5.0
    stage.downramp.ramp_beta_mag = 4.0
    assert np.isclose(stage.ramp_beta_mag, 10.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.ramp_beta_mag, 5.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.ramp_beta_mag, 4.0, rtol=1e-15, atol=0.0)

    stage2 = StageBasic()
    stage2.upramp = PlasmaRamp()
    stage2.downramp = PlasmaRamp()

    assert stage2.ramp_beta_mag is None
    assert stage2.upramp.ramp_beta_mag is None
    assert stage2.downramp.ramp_beta_mag is None

    stage2.upramp.ramp_beta_mag = 9.0
    stage2.downramp.ramp_beta_mag = 8.0
    assert stage2.ramp_beta_mag is None
    assert np.isclose(stage2.upramp.ramp_beta_mag, 9.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.ramp_beta_mag, 8.0, rtol=1e-15, atol=0.0)

    stage2.ramp_beta_mag = 11.0
    assert np.isclose(stage2.ramp_beta_mag, 11.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.ramp_beta_mag, 9.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.ramp_beta_mag, 8.0, rtol=1e-15, atol=0.0)


@pytest.mark.stageGeometry
def test_Stage__prepare_ramps():
    "Testing ``Stage._prepare_ramps()``."

    # TODO: move to a separate file test_Stage.py.

    from abel.utilities.plasma_physics import beta_matched

    stage = StageBasic()
    stage.upramp = PlasmaRamp()
    stage.downramp = PlasmaRamp()

    # Trigger exception for nominal energy is not set
    with pytest.raises(StageError):
        stage._prepare_ramps()

    stage.nom_energy = 5e9                                                          # [eV]
    stage.plasma_density = 7e21                                                     # [m^-3]
    stage.nom_energy_gain = 31.9e9                                                  # [eV]
    stage.nom_accel_gradient = 6.4e9                                                # [GV/m]

    # Trigger exception for not setting any ramp_beta_mag
    with pytest.raises(ValueError):
        stage._prepare_ramps()

    # ========== Set different values for ramp_beta_mag in stage and ramps ==========
    stage.ramp_beta_mag = 10.0
    stage.upramp.ramp_beta_mag = 5.0
    stage.downramp.ramp_beta_mag = 4.0

    assert stage.upramp.plasma_density is None
    assert stage.upramp.length is None
    assert stage.upramp.length_flattop is None
    assert np.isclose(stage.upramp.nom_energy, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert stage.upramp.nom_energy_gain is None
    assert stage.upramp.nom_energy_gain_flattop is None
    assert stage.upramp.nom_accel_gradient is None
    assert stage.upramp.nom_accel_gradient_flattop is None
    assert stage.upramp.driver_source is None
    assert stage.upramp.has_ramp() is False

    assert stage.downramp.plasma_density is None
    assert stage.downramp.length is None
    assert stage.downramp.length_flattop is None
    assert stage.downramp.nom_energy is None
    assert stage.downramp.nom_energy_flattop is None
    assert stage.downramp.nom_energy_gain is None
    assert stage.downramp.nom_energy_gain_flattop is None
    assert stage.downramp.nom_accel_gradient is None
    assert stage.downramp.nom_accel_gradient_flattop is None
    assert stage.downramp.driver_source is None
    assert stage.downramp.has_ramp() is False

    stage._prepare_ramps()

    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy + stage.upramp.nom_energy_gain, rtol=1e-15, atol=0.0)

    assert np.isclose(stage.upramp.plasma_density, stage.plasma_density/stage.upramp.ramp_beta_mag, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.length, stage.upramp.length_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.length_flattop, beta_matched(stage.plasma_density, stage.upramp.nom_energy)*np.pi/(2*np.sqrt(1/stage.upramp.ramp_beta_mag)), rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_energy, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_energy_gain_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.upramp.nom_accel_gradient_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert stage.upramp.driver_source is None
    assert stage.upramp.has_ramp() is False

    assert np.isclose(stage.downramp.plasma_density, stage.plasma_density/stage.downramp.ramp_beta_mag, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.length, stage.downramp.length_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.length_flattop, beta_matched(stage.plasma_density, stage.downramp.nom_energy)*np.pi/(2*np.sqrt(1/stage.downramp.ramp_beta_mag)), rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_energy, stage.nom_energy_flattop + stage.nom_energy_gain_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_energy_flattop, stage.downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_energy_gain_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.downramp.nom_accel_gradient_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert stage.downramp.driver_source is None
    assert stage.downramp.has_ramp() is False


    # ========== Only set ramp_beta_mag in stage ==========
    stage2 = StageBasic()
    stage2.nom_energy = 5e9                                                         # [eV]
    stage2.plasma_density = 7e21                                                    # [m^-3]
    stage2.nom_energy_gain = 31.9e9                                                 # [eV]
    stage2.nom_accel_gradient = 6.4e9                                               # [GV/m]
    stage2.upramp = PlasmaRamp()
    stage2.downramp = PlasmaRamp()
    stage2.ramp_beta_mag = 11.0

    assert stage2.upramp.plasma_density is None
    assert stage2.upramp.length is None
    assert stage2.upramp.length_flattop is None
    assert np.isclose(stage2.upramp.nom_energy, stage2.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_energy_flattop, stage2.nom_energy, rtol=1e-15, atol=0.0)
    assert stage2.upramp.nom_energy_gain is None
    assert stage2.upramp.nom_energy_gain_flattop is None
    assert stage2.upramp.nom_accel_gradient is None
    assert stage2.upramp.nom_accel_gradient_flattop is None
    assert stage2.upramp.driver_source is None
    assert stage2.upramp.has_ramp() is False

    assert stage2.downramp.plasma_density is None
    assert stage2.downramp.length is None
    assert stage2.downramp.length_flattop is None
    assert stage2.downramp.nom_energy is None
    assert stage2.downramp.nom_energy_flattop is None
    assert stage2.downramp.nom_energy_gain is None
    assert stage2.downramp.nom_energy_gain_flattop is None
    assert stage2.downramp.nom_accel_gradient is None
    assert stage2.downramp.nom_accel_gradient_flattop is None
    assert stage2.downramp.driver_source is None
    assert stage2.downramp.has_ramp() is False

    stage2._prepare_ramps()

    assert np.isclose(stage2.nom_energy_flattop, stage2.nom_energy + stage2.upramp.nom_energy_gain, rtol=1e-15, atol=0.0)

    assert np.isclose(stage2.upramp.plasma_density, stage2.plasma_density/stage2.ramp_beta_mag, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.length, stage2.upramp.length_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.length_flattop, beta_matched(stage2.plasma_density, stage2.upramp.nom_energy)*np.pi/(2*np.sqrt(1/stage2.ramp_beta_mag)), rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_energy, stage2.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_energy_flattop, stage2.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_energy_gain_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.upramp.nom_accel_gradient_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert stage2.upramp.driver_source is None
    assert stage2.upramp.has_ramp() is False

    assert np.isclose(stage2.downramp.plasma_density, stage2.plasma_density/stage2.ramp_beta_mag, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.length, stage2.downramp.length_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.length_flattop, beta_matched(stage2.plasma_density, stage2.downramp.nom_energy)*np.pi/(2*np.sqrt(1/stage2.ramp_beta_mag)), rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_energy, stage2.nom_energy_flattop + stage2.nom_energy_gain_flattop, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_energy_flattop, stage2.downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_energy_gain_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stage2.downramp.nom_accel_gradient_flattop, 0.0, rtol=1e-15, atol=0.0)
    assert stage2.downramp.driver_source is None
    assert stage2.downramp.has_ramp() is False