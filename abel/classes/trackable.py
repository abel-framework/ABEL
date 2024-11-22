from abc import ABC, abstractmethod
import warnings
import numpy as np

class Trackable(ABC):
    
    # constructor
    @abstractmethod
    def __init__(self, num_bunches_in_train=None, bunch_separation=None, rep_rate_trains=None, name=None):
        
        # bunch train pattern, through @property setters/getters
        # Actually stored in _bunch_separation, _num_bunches_in_train, _rep_rate_trains.
        # From these we can also derive bunch_frequency, train_duration, and rep_rate_average.
        self.bunch_separation = bunch_separation # [s]
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains # [Hz]
        
        self.name = name

    #-----------------------------------------#
    # Little helper functions                 #
    #=========================================#

    def _ensureFloat(self,r, ensurePos=False) -> float:
        "Little helper function, allowing None or float, autoconverting int to float. If ensurePos is True, then only allow r>0.0."
        if r == None:
            return r
        if type(r) == int or type(r) == np.float64:
            #quietly convert to float
            r = float(r)
        if type(r) != float:
            raise TypeError(f"must be float, int, or None. Got: {type(r)}, {r}")
        if ensurePos and r < 0.0:
            raise ValueError("must be >= 0.0")
        return r

    #-----------------------------------------#
    # Properties                              #
    #=========================================#
      
    @property
    def name(self):
        if self._name is None:
            return type(self).__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    # time structure
    
    @property
    def bunch_separation(self) -> float:
        "The time [s] between each bunch in the train"
        return self._bunch_separation
    @bunch_separation.setter
    def bunch_separation(self, bunch_separation : float):
        self._bunch_separation = bunch_separation

    @property
    def bunch_frequency(self) -> float:
        "The frequency [1/s] of bunches in the train"
        if self.num_bunches_in_train == 1:
            raise ValueError("Bunch frequency undefined when num_bunches_in_train == 1")
        return 1.0/self.bunch_separation
    @bunch_frequency.setter
    def bunch_frequency(self, bunch_frequency):
        self.bunch_separation = 1.0/bunch_frequency
    
    @property
    def num_bunches_in_train(self) -> int:
        """
        The number of bunches in the train.
        When 1, train_duration is 0.0 and average_current_train is None / undefined.
        """
        #if self._num_bunches_in_train == None:
        #    raise TrackableInitializationException("num_bunches_in_train is not yet initialized")
        return self._num_bunches_in_train
    @num_bunches_in_train.setter
    def num_bunches_in_train(self, num_bunches_in_train : int):
        if num_bunches_in_train == None:
            self._num_bunches_in_train = None
            return
        if type(num_bunches_in_train) != int:
            raise TypeError("num_bunches_in_train must be int or None")
        if num_bunches_in_train <= 0:
            raise ValueError("num_bunches_in_train must be > 0")
        self._num_bunches_in_train = num_bunches_in_train
    
    @property
    def train_duration(self) -> float:
        """
        The train duration [s] = beam pulse length [s] = length of RF pulse flat top length [s] for the RF structures.
        0.0 for single-bunch trains.
        Normally populated from Beam in track() but can be set directly for calculator use.
        """
        if self.num_bunches_in_train == 1:
            return 0.0 
        return self.bunch_separation*(self.num_bunches_in_train-1)
    @train_duration.setter
    def train_duration(self, train_duration : float):
        raise NotImplementedError("Cannot directly set train_duration")
    def get_train_duration(self):
        "Get the duration of the bunch train [s] - alias for self.train_duration"
        if self._num_bunches_in_train is None:
            warnings.warn("Bunch separation is unset, trackable.train_duration is undefined", DeprecationWarning)
            return None
        return self.train_duration
        #if self.bunch_separation is not None:
        #    return self.bunch_separation * (self.num_bunches_in_train-1)
        #else:
        #    return None

    @property
    def rep_rate_trains(self):
        "The average repetition rate of trains [Hz]"
        #if self._rep_rate_trains == None:
        #    raise TrackableInitializationException("rep_rate_trains is not yet initialized")
        return self._rep_rate_trains
    @rep_rate_trains.setter
    def rep_rate_trains(self,rep_rate_trains : float):
        self._rep_rate_trains = rep_rate_trains
    def get_rep_rate_trains(self):
        "Get the train repetition rate [Hz] - alias for self.rep_rate_trains"
        return self.rep_rate_trains

    @property
    def rep_rate_average(self):
        "Average repetition rate of bunches [Hz]"
        return self.num_bunches_in_train * self.rep_rate_trains
    @rep_rate_average.setter
    def rep_rate_average(self,rep_rate_average):
        raise NotImplementedError("Cannot directly set rep_rate_average")
    def get_rep_rate_average(self):
        "Get the average repetition rate of bunches [Hz] - alias for self.rep_rate_average"
        if self._rep_rate_trains is None:
            warnings.warn("Rep_rate_trains is unset, trackable.rep_rate_average is undefined", DeprecationWarning)
            return None
        return self.rep_rate_average
        #if self.rep_rate_trains is not None:
        #    return self.num_bunches_in_train * self.rep_rate_trains
        #else:
        #    return None
    
    #-----------------------------------------#
    # Tracking function                       #
    #=========================================#
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # remove nan particles
        beam.remove_nans()
        
        # advance beam location
        beam.location += self.get_length()
        
        # iterate trackable number
        beam.trackable_number += 1

        # set the bunch pattern if not already set
        if self.bunch_separation is None:
            self.bunch_separation = beam.bunch_separation
        if self.num_bunches_in_train is None:
            self.num_bunches_in_train = beam.num_bunches_in_train
        
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
    # Note: Not sure if these really need to be abstract?
    _length = None
    @property
    @abstractmethod
    def length(self) -> float:
        "Length of the trackable element, added to the Beam location after tracking [m]"
        #Simplest possible implementation of the `length` property
        return self._length
    @length.setter
    @abstractmethod
    def length(self, length : float):
        self._length = length
    def get_length(self) -> float:
        "Length of the trackable element, added to the Beam location after tracking [m]"
        return self.length

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
        
class TrackableInitializationException(Exception):
    "An Exception class that is raised when trying to access a uninitialized field"
    pass
