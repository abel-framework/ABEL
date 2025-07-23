"""
Contains the implementation for ramps as a Stage class.
Only meant to store information.
"""

from abel.classes.stage.stage import Stage, StageError

class PlasmaRamp(Stage):
    
    # ==================================================
    def __init__(self, nom_energy_gain=None, ramp_plasma_density=None, ramp_length=None, ramp_shape='uniform'):

        super().__init__(nom_accel_gradient=None, nom_energy_gain=nom_energy_gain, plasma_density=ramp_plasma_density, driver_source=None, ramp_beta_mag=1.0)

        # TODO: Need code to handle ramp_shape='from_file' ... need to ignore ramp_length, ramp_plasma_density. These need to be calculated.

        self.ramp_shape = ramp_shape
        self.length = ramp_length

        # Acceleration gradient and length cannot be set until the ramp nominal 
        # energy is known. These calculation are done by Stage._prepare_ramps().


    # ==================================================
    def track(self):
        raise StageError('track() is not implemented for StageRamp. Use track_upramp() or track_downramp() from the Stage class instead.')
    

    # ==================================================
    def set_upramp(self):
        raise StageError('Cannot add a ramp to StageRamp.')
    

    # ==================================================
    def set_downramp(self):
        raise StageError('Cannot add a ramp to StageRamp.')
    

    # ==================================================
    def print_summary(self):
        if self.is_upramp():
            print('Ramp type: \t\t\t\t\t\t upramp')
        if self.is_downramp():
            print('Ramp type: \t\t\t\t\t\t downramp')
        print('Ramp shape: \t\t\t\t\t\t', self.ramp_shape)
        super().print_summary()
        
        
    

