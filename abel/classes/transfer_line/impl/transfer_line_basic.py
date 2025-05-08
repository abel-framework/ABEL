import scipy.constants as SI
import numpy as np
from abel.classes.transfer_line.transfer_line import TransferLine

class TransferLineBasic(TransferLine):
    
    def __init__(self, nom_energy=None, length=None):
        super().__init__(nom_energy=nom_energy, length=length)
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
