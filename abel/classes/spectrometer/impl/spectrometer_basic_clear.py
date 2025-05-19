from abel.classes.spectrometer.spectrometer import Spectrometer

class SpectrometerBasicCLEAR(Spectrometer):
    
    def __init__(self, use_otr_screen=True):
        
        super().__init__()
        
        self.use_otr_screen = use_otr_screen
        self.location_otr_screen = 0.3 # [m]
    
    # lattice length
    def get_length(self):
        if self.use_otr_screen:
            return self.location_otr_screen
        else:
            return 0
    
    # tracking function
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # drift transport
        if self.use_otr_screen:            
            beam.transport(self.location_otr_screen)
        
        return super().track(beam, savedepth, runnable, verbose)
        
    