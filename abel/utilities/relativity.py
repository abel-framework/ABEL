import scipy.constants as SI
import numpy as np

defaultUnitE = 'eV'

# convert Lorentz beta to Lorentz gamma
def beta2gamma(beta):
    return np.sign(beta)/np.sqrt(1-beta**2)

# convert Lorentz beta to velocity
def beta2velocity(beta):
    return beta * SI.c

# convert Lorentz beta to kinetic energy [eV]
def beta2energy(beta, unitE=defaultUnitE):
    return gamma2energy(beta2gamma(beta), unitE)


# convert Lorentz gamma to Lorentz beta
def gamma2beta(gamma):
    return np.sign(gamma) * np.sqrt(1-1/gamma**2)

# convert Lorentz gamma to velocity
def gamma2velocity(gamma):
    return beta2velocity(gamma2beta(gamma))

# convert Lorentz gamma to proper velocity
def gamma2proper_velocity(gamma):
    return abs(gamma) * gamma2velocity(gamma)

# convert Lorentz gamma to momentum [kg m/s]
def gamma2momentum(gamma, m=SI.m_e):
    p = m * gamma2proper_velocity(gamma)
    return p

# convert Lorentz gamma to kinetic energy
def gamma2energy(gamma, unit=defaultUnitE, m=SI.m_e):
    E = gamma * (m * SI.c**2)
    if unit == defaultUnitE:
        E =  E / SI.e
    return E


# convert velocity to Lorentz beta
def velocity2beta(v):
    return v / SI.c

# convert velocity [m/s] to Lorentz gamma
def velocity2gamma(v):
    return beta2gamma(velocity2beta(v))

# convert proper velocity to Lorentz gamma
def proper_velocity2gamma(u):
    return np.sign(u) * np.sqrt(1+(u/SI.c)**2)

# convert proper velocity to momentum [kg m/s]
# def proper_velocity2momentum(u):
#     return gamma2momentum(proper_velocity2gamma(u))
def proper_velocity2momentum(u, m=SI.m_e):
    return m*u

# convert proper velocity to energy
def proper_velocity2energy(u, unit=defaultUnitE, m=SI.m_e):
    return gamma2energy(proper_velocity2gamma(u), unit=unit, m=m)


# convert momentum [kg m/s] to Lorentz gamma
def momentum2gamma(p, m=SI.m_e):
    return np.sign(p) * np.sqrt((p / (m * SI.c))**2 + 1)
    
# convert momentum [kg m/s] to proper velocity
def momentum2proper_velocity(p, m=SI.m_e):
    return p / m

# convert momentum [kg m/s] to energy [eV]
def momentum2energy(p):
    return gamma2energy(momentum2gamma(p))


# convert kinetic energy [eV] to Lorentz beta
def energy2beta(E, unit=defaultUnitE):
    return gamma2beta(energy2gamma(E, unit))

# convert kinetic energy [eV] to Lorentz gamma
def energy2gamma(E, unit=defaultUnitE, m=SI.m_e):
    if unit == defaultUnitE:
        E = E * SI.e
    return E / (m * SI.c**2)

# convert energy to proper velocity
def energy2proper_velocity(E, unit=defaultUnitE, m=SI.m_e):
    return gamma2proper_velocity(energy2gamma(E, unit=unit, m=m))

def energy2momentum(E, unit=defaultUnitE, m=SI.m_e):
    return gamma2momentum(energy2gamma(E, unit=unit, m=m), m=m)

