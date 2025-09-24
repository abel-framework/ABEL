from abel.classes.source.source import Source

class SourceCapsule(Source):
    """
    Source class used for "encapsulating" a predefined beam, used e.g. between ramps in plasma stages.
    """
    
    def __init__(self, length=0, beam=None, energy=None, charge=None, x_offset=0, y_offset=0, x_angle=0, y_angle=0, wallplug_efficiency=1, accel_gradient=None):
        
        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle)

        self.beam = beam
        #print('beam location inside capsule:', beam.location)     #######################

    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):

        import copy
        
        # return the saved beam
        beam = copy.deepcopy(self.beam)
        
        # add jitters and offsets in super function
        return beam
        #return super().track(beam, savedepth, runnable, verbose)
    