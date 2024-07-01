from abc import ABC, abstractmethod
import numpy as np

class Trackable(ABC):
    
    # constructor
    @abstractmethod
    def __init__(self, num_bunches_in_train=1, bunch_separation=None, rep_rate_trains=None):
        
        # bunch train pattern
        self.bunch_separation = bunch_separation # [s]
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains # [Hz]

        self._name = None


      
    @property
    def name(self):
        if self._name is None:
            return type(self).__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    
    # time structure functions
    
    def get_train_duration(self):
        "Get the duration of the bunch train [s]"
        if self.bunch_separation is not None:
            return self.bunch_separation * (self.num_bunches_in_train-1)
        else:
            return None

    def get_rep_rate_trains(self):
        "Get the train repetition rate [Hz]"
        return self.rep_rate_trains

    def get_rep_rate_average(self):
        "Get the average repetition rate of bunches [Hz]"
        if self.rep_rate_trains is not None:
            return self.num_bunches_in_train * self.rep_rate_trains
        else:
            return None
    
        
    # tracking function
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # remove nan particles
        beam.remove_nans()
        
        # advance beam location
        beam.location += self.get_length()
        
        # iterate trackable number
        beam.trackable_number += 1
        
        # save beam to file
        if savedepth >= 0:
            
            # describe tracking
            if verbose:
                if beam.trackable_number == 0:
                    tracking_word = 'Tracked'
                elif np.mod(beam.trackable_number, 10)==0:
                    tracking_word = '  > ...'
                else:
                    tracking_word = '    ...'

                # formatting
                reset = '\x1b[0m'
                bold = '\x1b[1m'
                reset_bold = '\x1b[21m'
                col_lgray = '\x1b[37m'
                col_dgray = '\x1b[90m'
                col_reset = '\x1b[39m'

                # add stage number if a stage
                from abel.classes.stage.stage import Stage
                if isinstance(self, Stage):
                    stage_word = f" #{beam.stage_number}"
                else:
                    stage_word = ''

                # print string
                print(f"{col_lgray}{tracking_word} #{str(beam.trackable_number).ljust(2)}{col_reset} {bold}{(type(self).__name__+stage_word).ljust(23)}{reset_bold} {col_lgray}(s ={col_reset} {bold}{beam.location:6.1f}{reset_bold} m{col_lgray}){col_reset} :   {col_lgray}E ={col_reset}{bold}{beam.energy()/1e9:6.1f}{reset_bold} GeV{col_lgray}, Q ={col_reset}{bold}{beam.charge()*1e9:6.2f}{reset_bold} nC{col_lgray}, σz ={col_reset} {bold}{beam.bunch_length()/1e-6:5.1f}{reset_bold} µm{col_lgray}, σE ={col_reset}{bold}{beam.rel_energy_spread()/1e-2:5.1f}%{reset_bold}{col_lgray}, ε ={col_reset}{bold}{beam.norm_emittance_x()/1e-6:6.1f}{reset_bold}/{bold}{beam.norm_emittance_y()/1e-6:.1f}{reset_bold} mm-mrad{reset}")
                
            # save to file
            if runnable is not None:
                beam.save(runnable)
            
        return beam
    
    
    # length of the trackable element
    @abstractmethod
    def get_length(self):
        "Length of the trackable element, added to the Beam location after tracking [m]"
        pass

    
    # abbreviation of the get_length() function
    def __len__(self):
        "Alias of the get_length() function [m]"
        return self.get_length()
        
    
    # object for survey plotting
    def survey_object(self):
        #return patches.Rectangle((0, -1), self.get_length(), 2)
        
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = type(self).__name__
        color = 'black'
        return x_points, y_points, final_angle, label, color
        
    