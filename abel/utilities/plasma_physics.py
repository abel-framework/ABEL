# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import scipy.constants as SI
from abel.utilities.relativity import energy2gamma
import numpy as np


# =============================================
def k_p(n0):
    """
    Plasma wavenumber [m^-1].
    """
    return np.sqrt(n0*SI.e**2/(SI.epsilon_0*SI.m_e*SI.c**2))


# =============================================
def omega_p(n0):
    """
    Plasma frequency [s^-1].
    """
    return np.sqrt(n0*SI.e**2/(SI.epsilon_0*SI.m_e))


# =============================================
def beta_matched(n0, E):
    """
    Matched beta function.

    Parameters
    ----------
    n0 : [m^-3] float
        Plasma number density of e.g. a plasma acceleration stage.

    E : [eV] float
        Beam energy

    Returns
    -------
    [m] float
        Matched beta function.
    """
    return np.sqrt(2*energy2gamma(E))/k_p(n0)
    

# =============================================
def wave_breaking_field(n0):
    """
    Wave breaking field [V/m].
    """
    return np.sqrt(n0 * SI.m_e * SI.c**2 / SI.epsilon_0)


# =============================================
def blowout_radius(n0, I_peak):
    """
    Approximate maximum blowout radius.

    Parameters
    ----------
    n0 : [m^-3] float
        Plasma number density of e.g. a plasma acceleration stage.

    I_peak : [A] float
        Beam peak current.

    Returns
    -------
    [m] float
        Approximate maximum blowout radius.
    """
    return (2 / k_p(n0)) * np.sqrt(I_peak * SI.e / (2 * np.pi * SI.m_e * SI.epsilon_0 * SI.c**3))


# =============================================
def mag_field_grad2wave_number(g, q=SI.e, m=SI.m_e):
    """
    Convert magnetic field gradient to plasma wavenumber with kp=sqrt(2qg/mc).

    Parameters
    ----------
    g : [T/m] float
        Magnetic field gradient.

    q : [C] float, optional
        The charge of the physical particle. Default set to SI.e.

    m : [kg] float, optional
        The mass of the physical particle. Default set to SI.m_e.


    Returns
    ----------
    kp : [m^-1] float
        Plasma wavenumber.
    """
    return np.sqrt(2*q*g/(m*SI.c))
