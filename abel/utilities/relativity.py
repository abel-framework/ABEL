# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import scipy.constants as SI
import numpy as np

defaultUnitE = 'eV'

# =============================================
# convert Lorentz beta to Lorentz gamma
def beta2gamma(beta):
    """
    Convert Lorentz beta to Lorentz factor.
    """
    return np.sign(beta)/np.sqrt(1-beta**2)


# =============================================
# convert Lorentz beta to velocity
def beta2velocity(beta):
    """
    Convert Lorentz beta to velocity [m/s].
    """
    return beta * SI.c


# =============================================
# convert Lorentz beta to kinetic energy [eV]
def beta2energy(beta, unitE=defaultUnitE):
    """
    Convert Lorentz beta to kinetic energy.

    Parameters
    ----------
    beta : float array_like
        Lorentz beta.

    unitE : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    Returns
    -------
    [eV or J] float array_like
        Kinetic energy. Unit depends on ``unit``.

    """
    return gamma2energy(beta2gamma(beta), unitE)


# =============================================
# convert Lorentz gamma to Lorentz beta
def gamma2beta(gamma):
    """
    Convert Lorentz gamma to Lorentz beta.
    """
    return np.sign(gamma) * np.sqrt(1-1/gamma**2)


# =============================================
# convert Lorentz gamma to velocity
def gamma2velocity(gamma):
    """
    Convert Lorentz gamma to velocity [m/s].
    """
    return beta2velocity(gamma2beta(gamma))


# =============================================
# convert Lorentz gamma to proper velocity
def gamma2proper_velocity(gamma):
    """
    Convert Lorentz gamma to proper velocity [m/s].
    """
    return abs(gamma) * gamma2velocity(gamma)


# =============================================
# convert Lorentz gamma to momentum [kg m/s]
def gamma2momentum(gamma, m=SI.m_e):
    """
    Convert Lorentz gamma to momentum [kg m/s].
    """
    p = m * gamma2proper_velocity(gamma)
    return p


# =============================================
# convert Lorentz gamma to kinetic energy
def gamma2energy(gamma, unit=defaultUnitE, m=SI.m_e):
    """
    Convert Lorentz factor to kinetic energy.

    Parameters
    ----------
    gamma : float array_like
        Lorentz factor.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    E : [eV or J] float array_like
        Kinetic energy. Unit depends on ``unit``.
    """
    E = gamma * (m * SI.c**2)
    if unit == defaultUnitE:
        E =  E / SI.e
    return E


# =============================================
# convert velocity to Lorentz beta
def velocity2beta(v):
    """
    Convert velocity [m/s] to Lorentz beta.
    """
    return v / SI.c


# =============================================
# convert velocity [m/s] to Lorentz gamma
def velocity2gamma(v):
    """
    Convert velocity [m/s] to Lorentz factor.
    """
    return beta2gamma(velocity2beta(v))


# =============================================
# convert proper velocity to Lorentz gamma
def proper_velocity2gamma(u):
    """
    Convert proper velocity [m/s] to Lorentz factor.
    """
    return np.sign(u) * np.sqrt(1+(u/SI.c)**2)


# =============================================
# convert proper velocity to momentum [kg m/s]
# def proper_velocity2momentum(u):
#     return gamma2momentum(proper_velocity2gamma(u))
def proper_velocity2momentum(u, m=SI.m_e):
    """
    Convert proper velocity to momentum.

    Parameters
    ----------
    u : [m/s] float array_like
        Proper velocity.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    [kg m/s] float array_like
        Particle momentum.
    """
    return m*u


# =============================================
# convert proper velocity to energy
def proper_velocity2energy(u, unit=defaultUnitE, m=SI.m_e):
    """
    Convert proper velocity to energy.

    Parameters
    ----------
    u : [m/s] float array_like
        Proper velocity.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    [eV or J] float array_like
        Particle energy. Unit depends on ``unit``.
    """
    return gamma2energy(proper_velocity2gamma(u), unit=unit, m=m)


# =============================================
# convert momentum [kg m/s] to Lorentz gamma
def momentum2gamma(p, m=SI.m_e):
    """
    Convert momentum to Lorentz factor.

    Parameters
    ----------
    p : [kg m/s] float array_like
        Particle momentum.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    float array_like
        Lorentz factor.
    """
    return np.sign(p) * np.sqrt((p / (m * SI.c))**2 + 1)


# =============================================
# convert momentum [kg m/s] to proper velocity
def momentum2proper_velocity(p, m=SI.m_e):
    """
    Convert momentum to proper velocity.

    Parameters
    ----------
    p : [kg m/s] float array_like
        Particle momentum.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    [m/s] float array_like
        Particle proper velocity.
    """
    return p / m


# =============================================
# convert momentum [kg m/s] to energy [eV]
def momentum2energy(p):
    """
    Convert momentum [kg m/s] to energy [eV].
    """
    return gamma2energy(momentum2gamma(p))


# =============================================
# convert kinetic energy [eV] to Lorentz beta
def energy2beta(E, unit=defaultUnitE):
    """
    Convert particle energy to Lorentz beta.

    Parameters
    ----------
    E : [eV or J] float array_like
        Particle energy. Unit depends on ``unit``.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    Returns
    -------
    float array_like
        Lorentz beta.
    """
    return gamma2beta(energy2gamma(E, unit))


# =============================================
# convert kinetic energy [eV] to Lorentz gamma
def energy2gamma(E, unit=defaultUnitE, m=SI.m_e):
    """
    Convert particle energy to Lorentz factor.

    Parameters
    ----------
    E : [eV or J] float array_like
        Particle energy. Unit depends on ``unit``.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    float array_like
        Lorentz factor.
    """
    if unit == defaultUnitE:
        E = E * SI.e
    return E / (m * SI.c**2)


# =============================================
# convert energy to proper velocity
def energy2proper_velocity(E, unit=defaultUnitE, m=SI.m_e):
    """
    Convert energy to proper velocity.

    Parameters
    ----------
    E : [eV or J] float array_like
        Particle energy. Unit depends on ``unit``.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    [m/s] float array_like
        Particle proper velocity.
    """
    return gamma2proper_velocity(energy2gamma(E, unit=unit, m=m))


# =============================================
def energy2momentum(E, unit=defaultUnitE, m=SI.m_e):
    """
    Convert energy to momentum.

    Parameters
    ----------
    E : [eV or J] float array_like
        Particle energy. Unit depends on ``unit``.

    unit : str, optional
        The output is in electronvolt if ``unit='eV'``. The output energy is in 
        joules otherwise. Defaults to ``'eV'``.

    m : [kg] float, optional
        The particle mass. Defaults to eletron mass.

    Returns
    -------
    [kg m/s] float array_like
        Particle momentum.
    """
    return gamma2momentum(energy2gamma(E, unit=unit, m=m), m=m)

