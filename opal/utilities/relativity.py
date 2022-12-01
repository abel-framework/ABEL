from opal.utilities import SI
import numpy as np

defaultUnitE = 'eV'
defaultUnitP = 'eV/c'
defaultMass = SI.me


# convert Lorentz beta to Lorentz gamma
def beta2gamma(beta):
    return 1/np.sqrt(1-beta**2)

# convert Lorentz beta to velocity
def beta2velocity(beta):
    return beta * SI.c

# convert Lorentz beta to kinetic energy [eV]
def beta2energy(beta, m = defaultMass, unitE = defaultUnitE):
    return gamma2energy(beta2gamma(beta), m, unitE)


# convert Lorentz gamma to Lorentz beta
def gamma2beta(gamma):
    return np.sqrt(1-1/gamma**2)

# convert Lorentz gamma to velocity
def gamma2velocity(gamma):
    return beta2velocity(gamma2beta(gamma))

# convert Lorentz gamma to proper velocity
def gamma2properVelocity(gamma):
    return gamma * gamma2velocity(gamma)

# convert Lorentz gamma to momentum [eV/c]
def gamma2momentum(gamma, m = defaultMass, unit = defaultUnitP):
    p = m * gamma2properVelocity(gamma)
    if unit == defaultUnitP:
        p = p / (SI.e / SI.c)
    return p

# convert Lorentz gamma to kinetic energy
def gamma2energy(gamma, m = defaultMass, unit = defaultUnitE):
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
def properVelocity2gamma(w):
    return np.sqrt(1+(w/SI.c)**2)

# convert proper velocity to energy
def properVelocity2momentum(w, m = defaultMass, unit = defaultUnitP):
    return gamma2momentum(properVelocity2gamma(w), m, unit)

# convert proper velocity to energy
def properVelocity2energy(w, m = defaultMass, unit = defaultUnitE):
    return gamma2energy(properVelocity2gamma(w), m, unit)


# convert momentum [eV/c] to Lorentz gamma
def momentum2gamma(p, m = defaultMass):
    return np.sqrt(1+(p / (m * SI.c))**2)

# convert momentum to energy [eV/c]
def momentum2energy(p, m = defaultMass):
    return gamma2energy(momentum2gamma(p, m), m)


# convert kinetic energy [eV] to Lorentz beta
def energy2beta(E, m = defaultMass, unit = defaultUnitE):
    return gamma2beta(energy2gamma(E, m, unit))

# convert kinetic energy [eV] to Lorentz gamma
def energy2gamma(E, m = defaultMass, unit = defaultUnitE):
    if unit == defaultUnitE:
        E = E * SI.e
    return E / (m * SI.c**2)

# convert energy to proper velocity
def energy2properVelocity(E, m = defaultMass, unit = defaultUnitE):
    return gamma2properVelocity(energy2gamma(E, m, unit))

