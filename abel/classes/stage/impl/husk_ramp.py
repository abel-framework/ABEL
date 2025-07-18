"""
Contains the implementation for ramps as a Stage class.
Only meant to store information.
"""

from abel.classes.stage.stage import Stage, StageError

class HuskRamp(Stage):
    
    # ==================================================
    def __init__(self, ramp_plasma_density=None, ramp_length=None, ramp_shape='uniform'):

        super().__init__(nom_accel_gradient=None, nom_energy_gain=None, plasma_density=ramp_plasma_density, driver_source=None, ramp_beta_mag=1.0)

        self.ramp_shape = ramp_shape
        self.length = ramp_length


    # ==================================================
    def track(self):
        raise StageError('track() is not implemented for StageRamp. Use track_upramp() or track_downramp() from the Stage class instead.')
    

    # ==================================================
    def set_upramp(self):
        raise StageError('Cannot add a ramp to StageRamp.')
    

    # ==================================================
    def set_downramp(self):
        raise StageError('Cannot add a ramp to StageRamp.')
    

