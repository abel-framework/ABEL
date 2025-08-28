from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.CONFIG import CONFIG
from abel.classes.cost_modeled import CostModeled
from abel.classes.source.impl.source_capsule import SourceCapsule
import numpy as np
import copy
import scipy.constants as SI
from types import SimpleNamespace
from abel.utilities.plasma_physics import beta_matched
from typing import Self

class Stage(Trackable, CostModeled):
    
    # ==================================================
    @abstractmethod
    def __init__(self, nom_accel_gradient, nom_energy_gain, plasma_density, driver_source=None, ramp_beta_mag=None):

        super().__init__()
        
        # common variables
        self.nom_accel_gradient = nom_accel_gradient
        self.nom_energy_gain = nom_energy_gain
        self.plasma_density = plasma_density
        self.driver_source = driver_source
        self.ramp_beta_mag = ramp_beta_mag
        
        self.stage_number = None
        
        # nominal initial energy
        self.nom_energy = None 
        self.nom_energy_flattop = None

        self._return_tracked_driver = False
        
        self.evolution = SimpleNamespace()
        self.evolution.beam = SimpleNamespace()
        self.evolution.beam.slices = SimpleNamespace()
        self.evolution.driver = SimpleNamespace()
        self.evolution.driver.slices = SimpleNamespace()
        
        self.efficiency = SimpleNamespace()
        self.efficiency.dumped_power = None
        
        self.initial = SimpleNamespace()
        self.initial.driver = SimpleNamespace()
        self.initial.driver.instance = SimpleNamespace()
        self.initial.beam = SimpleNamespace()
        self.initial.beam.instance = SimpleNamespace()
        self.initial.beam.current = SimpleNamespace()
        self.initial.beam.density = SimpleNamespace()
        self.initial.plasma = SimpleNamespace()
        self.initial.plasma.density = SimpleNamespace()
        self.initial.plasma.wakefield = SimpleNamespace()
        self.initial.plasma.wakefield.onaxis = SimpleNamespace()
        
        self.final = SimpleNamespace()
        self.final.driver = SimpleNamespace()
        self.final.driver.instance = SimpleNamespace()
        self.final.beam = SimpleNamespace()
        self.final.beam.instance = SimpleNamespace()
        self.final.beam.current = SimpleNamespace()
        self.final.beam.density = SimpleNamespace()
        self.final.plasma = SimpleNamespace()
        self.final.plasma.density = SimpleNamespace()
        self.final.plasma.wakefield = SimpleNamespace()
        self.final.plasma.wakefield.onaxis = SimpleNamespace()

        self.name = 'Plasma stage'


    # ==================================================
    ## Define upramp and downramp, if present
    @property
    def upramp(self) -> Self | None:
        "The upramp of this stage, which also is a Stage"
        return self._upramp
    @upramp.setter
    def upramp(self, upramp : Self | None):
        if not isinstance(upramp, Stage) and upramp is not None:
            raise StageError("The upramp must be an instance of Stage or None")    
        self._upramp = upramp
        if upramp is not None:
            self._upramp.parent = self

        self._resetLengthEnergyGradient()
        self._recalcLengthEnergyGradient()
    _upramp = None


    # ==================================================
    @property
    def downramp(self) -> Self:
        "The downramp of this stage, which also is a Stage"
        return self._downramp
    @downramp.setter
    def downramp(self, downramp : Self | None):
        if not isinstance(downramp, Stage) and downramp is not None:
            raise StageError("The downramp must be an instance of Stage or None")
        self._downramp = downramp
        if downramp is not None:
            self._downramp.parent = self

        self._resetLengthEnergyGradient()
        self._recalcLengthEnergyGradient()
    _downramp = None


    # ==================================================
    def stage2ramp(self, ramp_plasma_density=None, ramp_length=None):
        """
        Used for copying a predefined stage's settings and configurations to set up flat ramps.
    
        Parameters
        ----------
        ramp_plasma_density : [m^-3] float, optional
            Plasma density for the ramp.

        ramp_length : [m] float, optional
            Length of the ramp.
    
            
        Returns
        ----------
        stage_copy : ``Stage`` object
            A modified deep copy of the original stage.
        """

        stage_copy = copy.deepcopy(self)
        stage_copy.ramp_beta_mag = 1.0

        # Delete any upramps and downramps that might be present
        if stage_copy.upramp is not None:
            stage_copy.upramp = None
        if stage_copy.downramp is not None:
            stage_copy.downramp = None

        # Can set energy gain and gradient parameters to None to let track_upramp() and track_downramp() determine these.
        # Do try/except to allow zeroing everything.
        try:
            stage_copy.nom_accel_gradient = None
        except VariablesOverspecifiedError:
            pass
        try:
            stage_copy.nom_energy_gain = None
        except VariablesOverspecifiedError:
            pass
        try:
            stage_copy.nom_accel_gradient_flattop = None
        except VariablesOverspecifiedError:
            pass
        try:
            stage_copy.nom_energy_gain_flattop = None
        except VariablesOverspecifiedError:
            pass
            
        # Everything else now unset, can set this safely.
        # Will also trigger reset/recalc if needed
        stage_copy.length_flattop = None
        stage_copy.length = ramp_length
        stage_copy.plasma_density = ramp_plasma_density
         
        # Remove the driver source, as this will be replaced with SourceCapsule in track_upramp() and track_downramp()
        stage_copy.driver_source = None

        return stage_copy
    

    # ==================================================
    def _prepare_ramps(self):
        "Set ramp lengths and nominal energy gains if the ramps exist (both upramp and downramp lengths have to be set up before track_upramp())."
        if self.nom_energy is None:
            #Should be set in linac.track()
            raise StageError('Stage nominal energy is None.')

        if self.upramp is not None:
            if self.upramp.nom_energy_gain is None:
                self.upramp.nom_energy_gain = 0.0
            if self.upramp.nom_energy is None:
                self.upramp.nom_energy = self.nom_energy
                self.nom_energy_flattop = self.nom_energy + self.upramp.nom_energy_gain
            if self.upramp.length is None:
                self.upramp.length = self._calc_ramp_length(self.upramp)
        else:
            self.nom_energy_flattop = self.nom_energy
            
        if self.downramp is not None:
            if self.downramp.nom_energy_gain is None:
                self.downramp.nom_energy_gain = 0.0
            if self.downramp.nom_energy is None:
                self.downramp.nom_energy = self.nom_energy_flattop + self.nom_energy_gain_flattop
            if self.downramp.length is None:
                self.downramp.length = self._calc_ramp_length(self.downramp)


    # ==================================================
    def _calc_ramp_length(self, ramp : Self) -> float:
        "Calculate and set the up/down ramp (uniform step ramp) length [m] based on stage nominal energy."
        if ramp.nom_energy is None:
            raise StageError('Ramp nominal energy is None.')
        ramp_length = beta_matched(self.plasma_density, ramp.nom_energy)*np.pi/(2*np.sqrt(1/self.ramp_beta_mag))
        if ramp_length < 0.0:
            raise ValueError(f"ramp_length = {ramp_length} [m] < 0.0")
        return ramp_length
    

    # ==================================================
    @property
    def parent(self) -> Self | None:
        "The parent of this stage (which is then an upramp or downramp), or None"
        return self._parent
    @parent.setter
    def parent(self, parent : Self):
        if parent is None:
            raise StageError("Stage parent cannot be unset")
        if not isinstance(parent, Stage):
            raise StageError("Stage parent must be an instance of Stage")
        self._parent = parent
    _parent = None


    # ==================================================
    def _getOtherRamp(self, aRamp : Self) -> Self:
        "Lets the upramp get hold of the downramp in the same pair, and vise versa"
        if aRamp == self.upramp:
            return self.downramp
        elif aRamp == self.downramp:
            return self.upramp
        else:
            raise StageError("Could not find calling ramp?")


    # ==================================================
    def _getOverallestStage(self) -> Self:
        "Find and return the most overall stage in the hierachy"
        bottom_Stage = self
        itrCtr_pSearch = 0
        while bottom_Stage.parent is not None:
            #Climb down to the bottom
            itrCtr_pSearch += 1
            if itrCtr_pSearch > 20: #Magic number
                raise StageError("Too many levels of parents, giving up")
            bottom_Stage = bottom_Stage.parent
        return bottom_Stage

    
    
    ## Tracking methods

    # ==================================================
    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stage_number += 1
        if self.length < 0.0:
            raise ValueError(f"Length = {self.length} [m] < 0.0")
        if self.length_flattop < 0.0:
            raise ValueError(f"Flattop length = {self.length_flattop} [m] < 0.0")
        return super().track(beam, savedepth, runnable, verbose)


    # ==================================================
    # upramp to be tracked before the main tracking
    def track_upramp(self, beam0, driver0=None):
        if self.upramp is not None:

            # set driver
            self.upramp.driver_source = SourceCapsule(beam=driver0)

            # determine density if not already set
            if self.upramp.plasma_density is None:
                self.upramp.plasma_density = self.plasma_density/self.ramp_beta_mag

            # perform tracking
            self.upramp._return_tracked_driver = True
            beam, driver = self.upramp.track(beam0)
            beam.stage_number -= 1
            driver.stage_number -= 1
            
        else:
            beam = beam0
            driver = driver0
        
        return beam, driver


    # ==================================================
    # downramp to be tracked after the main tracking
    def track_downramp(self, beam0, driver0):
        if self.downramp is not None:

            # set driver
            self.downramp.driver_source = SourceCapsule(beam=driver0)
            
            # determine density if not already set
            if self.downramp.plasma_density is None:
                # set ramp density
                self.downramp.plasma_density = self.plasma_density/self.ramp_beta_mag           
            
            # perform tracking
            self.downramp._return_tracked_driver = True
            beam, driver = self.downramp.track(beam0)
            beam.stage_number -= 1
            driver.stage_number -= 1
            
        else:
            beam = beam0
            driver = driver0
            
        return beam, driver


    ## Mutually consistent calculation for length, nom_accel_gradient, nom_energy gain,
    #  their flattop counterparts, and (if existing) their stage counterparts.
    #
    #  If you try to set something that has already been set or calculated,
    #  we raise a VariablesOverspecifiedError exception with a meaningful error message.
    #  If you try to get something that is unknown, you will get a None.
    #
    # The algorithm when setting a new variable which is unknown is basically:
    #  1. Nuke all calculated variables in the whole Stage hierarchy
    #  2. "Calculate" all variables where the user has set a value by copying this value into the calculated value
    #  3. Try to calculate whatever variables we can in the whole hierarchy from what is currently known
    #  4. If we managed to calculate something, repeat 3. until nothing new comes out.
    #
    #  Step 1/2 is implemented in methods _resetLengthEnergyGradient() and _resetLengthEnergyGradient_helper()
    #  Step 3/4 is implemented in methods _recalcLengthEnergyGradient() and _recalcLengthEnergyGradient_helper()
    #
    #  Functions _printLengthEnergyGradient_internal() and _printVerb() are for debugging.
    #  The last one works like print() and is enabled/disabled by the variable doVerbosePrint_debug.
    #
    # User-specified data is stored in variables with a single underscore, e.g. _length,
    # and calculated variables are stored in variables with also _calc, i.e. _length_calc.
    # Please only access these through their getters and setters i.e. length(self), length(self,length).
    # The only place the _calc variables are touched outside of the getters and setters
    # is when resetting them in _resetLengthEnergyGradient_helper()
    # and when calculating them in _recalcLengthEnergyGradient_helper().
    #
    # If the calculation takes too many iterations, something is probably gone wrong (bug),
    # and we raise a StageError exception.
    #
    # If the calculation goes wrong in some other way (e.g. produces a negative stage length),
    # we raise a VariablesOutOfRangeError and unwind the set.
    #
    # Traversing and keeping track of the hierarchy of Stages is done using the properties parent, upramp, and downramp.
    # These are ensured to be a Stage or None. The parent cannot be unset, but upramp and downramp can be removed (set to None).
    # Helper methods _getOtherRamp() and _getOverallestStage() are used to traverse the hierarchy.


    # Methods for setting and getting the variables

    # ==================================================
    @property
    def length(self) -> float:
        "Total length of the trackable stage element [m], or None if not set/calculateable"
        return self._length_calc # Always return the dynamic version
    @length.setter
    def length(self, length : float):
        if self._length_calc is not None and self._length is None:
            # If there is a known value, and we're not trying to modify a user-set value -> ERROR!
            raise VariablesOverspecifiedError("length already known/calculateable, cannot set.")
        if length is not None:
            if length < 0.0 and self.sanityCheckLengths:
                raise VariablesOutOfRangeError(f"Setting length = {length} < 0; check variables or disable check by setting stage.sanityCheckLengths=False")
        #Set user value and recalculate
        length_old = self._length
        try:
            self._length = length
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            #Something went wrong in the calculation - unwind
            self._length = length_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            #Re-raise the exception so that the user gets the fantastically useful traceback and error message.
            raise

    # Variables are internal to the length/energy/gradient logic
    # do not modify externally!
    _length      = None #User-set value
    _length_calc = None #Dynamic version, current best-known value

    sanityCheckLengths = True # Set to False to disable raising VariablesOutOfRangeError exception on negative lengths.
                              # If it is unclear why negative lengths are happening,
                              # set doVerbosePrint_debug to True in order to spy on the calculation.

    def get_length(self) -> float:
        "Alias of length"
        return self.length

    # ==================================================
    @property
    def nom_energy_gain(self) -> float:
        "Total nominal energy gain of the stage [eV], or None if not set/calculateable"
        return self._nom_energy_gain_calc
    @nom_energy_gain.setter
    def nom_energy_gain(self, nom_energy_gain : float):
        if self._nom_energy_gain_calc is not None and self._nom_energy_gain is None:
            raise VariablesOverspecifiedError("nom_energy_gain already known/calculateable, cannot set.")
        
        nom_energy_gain_old = self._nom_energy_gain
        try:
            self._nom_energy_gain = nom_energy_gain
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            #Make variables consistent again
            self._nom_energy_gain = nom_energy_gain_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            raise
    _nom_energy_gain      = None
    _nom_energy_gain_calc = None

    def get_nom_energy_gain(self):
        "Alias of nom_energy_gain"
        return self.nom_energy_gain

    # ==================================================
    @property
    def nom_accel_gradient(self) -> float:
        "Total nominal accelerating gradient of the stage [eV/m], or None if not set/calculateable"
        return self._nom_accel_gradient_calc
    @nom_accel_gradient.setter
    def nom_accel_gradient(self, nom_accel_gradient : float):
        if self._nom_accel_gradient_calc is not None and self._nom_accel_gradient is None:
            raise VariablesOverspecifiedError("nom_accel_gradient already known/calculateable, cannot set")
        
        nom_accel_gradient_old = self._nom_accel_gradient
        try:
            self._nom_accel_gradient = nom_accel_gradient
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            self._nom_accel_gradient = nom_accel_gradient_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            raise
    _nom_accel_gradient      = None
    _nom_accel_gradient_calc = None


    # ==================================================
    @property
    def length_flattop(self) -> float:
        "Length of the plasma flattop [m], or None if not set/calculateable"
        return self._length_flattop_calc
    @length_flattop.setter
    def length_flattop(self, length_flattop : float):
        if self._length_flattop_calc is not None and self._length_flattop is None:
            raise VariablesOverspecifiedError("length_flattop already known/calculateable, cannot set")
        if length_flattop is not None:
            if length_flattop < 0.0 and self.sanityCheckLengths:
                raise VariablesOutOfRangeError(f"Setting length_flattop = {length_flattop} < 0; check variables or disable check by setting stage.sanityCheckLengths=False")
        
        length_flattop_old = self._length_flattop
        try:
            self._length_flattop = length_flattop
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            self._length_flattop = length_flattop_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            raise
    _length_flattop      = None
    _length_flattop_calc = None

    # ==================================================
    @property
    def nom_energy_gain_flattop(self) -> float:
        "Energy gain of the plasma flattop [eV], or None if not set/calculateable"
        return self._nom_energy_gain_flattop_calc
    @nom_energy_gain_flattop.setter
    def nom_energy_gain_flattop(self,nom_energy_gain_flattop : float):
        if self._nom_energy_gain_flattop_calc is not None and self._nom_energy_gain_flattop is None:
            raise VariablesOverspecifiedError("nom_energy_gain_flattop is already known/calculateable, cannot set")
        
        nom_energy_gain_flattop_old = self._nom_energy_gain_flattop
        try:
            self._nom_energy_gain_flattop = nom_energy_gain_flattop
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            self._nom_energy_gain_flattop = nom_energy_gain_flattop_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            raise
    _nom_energy_gain_flattop      = None
    _nom_energy_gain_flattop_calc = None


    # ==================================================
    @property
    def nom_accel_gradient_flattop(self) -> float:
        "Accelerating gradient of the plasma flattop [eV/m], or None if not set/calculateable"
        return self._nom_accel_gradient_flattop_calc
    @nom_accel_gradient_flattop.setter
    def nom_accel_gradient_flattop(self, nom_accel_gradient_flattop : float):
        if self._nom_accel_gradient_flattop_calc is not None and self._nom_accel_gradient_flattop is None:
            raise VariablesOverspecifiedError("nom_accel_gradient_flattop is already known/calculatable, cannot set")
        
        nom_accel_gradient_flattop_old = self._nom_accel_gradient_flattop
        try:
            self._nom_accel_gradient_flattop = nom_accel_gradient_flattop
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
        except VariablesOutOfRangeError:
            self._nom_accel_gradient_flattop = nom_accel_gradient_flattop_old
            self._resetLengthEnergyGradient()
            self._recalcLengthEnergyGradient()
            raise
    _nom_accel_gradient_flattop      = None
    _nom_accel_gradient_flattop_calc = None

    ## Recalculation methods

    # ==================================================
    def _resetLengthEnergyGradient(self):
        "Reset all the calculated values in the current Stage hierarchy"
        self._getOverallestStage()._resetLengthEnergyGradient_helper()


    # ==================================================
    def _resetLengthEnergyGradient_helper(self):
        #Climb back up from the bottom and set what we can
        # directly / to None, from the stored user input
        self._length_calc              = self._length
        self._nom_energy_gain_calc     = self._nom_energy_gain
        self._nom_accel_gradient_calc  = self._nom_accel_gradient

        self._length_flattop_calc             = self._length_flattop
        self._nom_energy_gain_flattop_calc    = self._nom_energy_gain_flattop
        self._nom_accel_gradient_flattop_calc = self._nom_accel_gradient_flattop

        self._printVerb("length                     (1)>", self.length)
        self._printVerb("nom_energy_gain            (1)>", self.nom_energy_gain)
        self._printVerb("nom_accel_gradient         (1)>", self.nom_accel_gradient)
        self._printVerb("length_flattop             (1)>", self.length_flattop)
        self._printVerb("nom_energy_gain_flattop    (1)>", self.nom_energy_gain_flattop)
        self._printVerb("nom_accel_gradient_flattop (1)>", self.nom_accel_gradient_flattop)

        if self.upramp is not None:
            self._printVerb("Upramp:")
            self.upramp._resetLengthEnergyGradient_helper()
        if self.downramp is not None:
            self._printVerb("Downramp:")
            self.downramp._resetLengthEnergyGradient_helper()


    # ==================================================
    def _recalcLengthEnergyGradient(self):
        #Iteratively calculate everything until stability is reached
        #Note: Before starting calculation, call _resetLengthEnergyGradient() to reset the hierachy
        self._getOverallestStage()._recalcLengthEnergyGradient_helper()
        self._printVerb()


    # ==================================================
    def _recalcLengthEnergyGradient_helper(self):
        itrCtr = 0
        updateCounter_total = 0
        while True:
            itrCtr += 1
            self._printVerb("itrCtr = ", itrCtr)
            if itrCtr > 20:
                self._printLengthEnergyGradient_internal()
                raise StageError("Not able make self-consistent calculation, infinite loop detected")

            updateCounter = 0

            #Indirect setting of overall
            if self.length is None:
                if self.nom_energy_gain is not None and self.nom_accel_gradient is not None:
                    L = self.nom_energy_gain / self.nom_accel_gradient
                    if L < 0.0 and self.sanityCheckLengths:
                        raise VariablesOutOfRangeError(f"Calculated length = {L} < 0; check variables or disable check by setting stage.sanityCheckLengths=False.")
                    self._length_calc = L
                    self._printVerb("length                     (2)>",self.length)
                    updateCounter += 1
            if self.nom_energy_gain is None:
                if self.length is not None and self.nom_accel_gradient is not None:
                    self._nom_energy_gain_calc = self.length * self.nom_accel_gradient
                    self._printVerb("nom_energy_gain            (2)>",self.nom_energy_gain)
                    updateCounter += 1
            if self.nom_accel_gradient is None:
                if self.length is not None and self.nom_energy_gain is not None:
                    self._nom_accel_gradient_calc = self.nom_energy_gain / self.length
                    self._printVerb("nom_accel_gradient         (2)>",self.nom_accel_gradient)
                    updateCounter += 1

            #Indirect setting of flattop
            if self.length_flattop is None:
                if self.nom_energy_gain_flattop is not None and self.nom_accel_gradient_flattop is not None:
                    L = self.nom_energy_gain_flattop / self.nom_accel_gradient_flattop
                    if L < 0.0 and self.sanityCheckLengths:
                        raise VariablesOutOfRangeError(f"Calculated length_flattop = {L} < 0; check variables or disable check by setting stage.sanityCheckLengths=False.")
                    self._length_flattop_calc = L
                    self._printVerb("length_flattop             (2)>",self.length_flattop)
                    updateCounter += 1
            if self.nom_energy_gain_flattop is None:
                if self.length_flattop is not None and self.nom_accel_gradient_flattop is not None:
                    self._nom_energy_gain_flattop_calc = self.length_flattop * self.nom_accel_gradient_flattop
                    self._printVerb("nom_energy_gain_flattop    (2)>",self.nom_energy_gain_flattop)
                    updateCounter += 1
            if self.nom_accel_gradient_flattop is None:
                if self.length_flattop is not None and self.nom_energy_gain_flattop is not None:
                    self._nom_accel_gradient_flattop_calc = self.nom_energy_gain_flattop / self.length_flattop
                    self._printVerb("nom_accel_gradient_flattop (2)>",self.nom_accel_gradient_flattop)
                    updateCounter += 1

            #Relationships total <-> flattop+ramp
            if self.length is None:
                if self.length_flattop is not None:
                    L = self.length_flattop
                    isDef = True
                    if self.upramp is not None:
                        if self.upramp.length is None:
                            isDef = False
                        else:
                            L += self.upramp.length
                    if self.downramp is not None:
                        if self.downramp.length is None:
                            isDef = False
                        else:
                            L += self.downramp.length
                    if isDef:
                        self._printVerb("length                     (3)>",L)
                        if L < 0.0 and self.sanityCheckLengths:
                            raise VariablesOutOfRangeError(f"Calculated length = {L} < 0; check variables or disable check by setting stage.sanityCheckLengths=False.")
                        self._length_calc = L
                        updateCounter += 1

            if self.length_flattop is None:
                if self.length is not None:
                    L = self.length
                    isDef = True
                    if self.upramp is not None:
                        if self.upramp.length is None:
                            isDef = False
                        else:
                            L -= self.upramp.length
                    if self.downramp is not None:
                        if self.downramp.length is None:
                            isDef = False
                        else:
                            L -= self.downramp.length
                    if isDef:
                        self._printVerb("length_flattop             (3)>",L)
                        if L < 0.0 and self.sanityCheckLengths:
                            raise VariablesOutOfRangeError(f"Calculated length_flattop = {L} < 0; check variables or disable check by setting stage.sanityCheckLengths=False.")
                        self._length_flattop_calc = L
                        updateCounter += 1

            if self.nom_energy_gain is None:
                if self.nom_energy_gain_flattop is not None:
                    dE = self.nom_energy_gain_flattop
                    isDef = True
                    if self.upramp is not None:
                        if self.upramp.nom_energy_gain is None:
                            isDef = False
                        else:
                            dE += self.upramp.nom_energy_gain
                    if self.downramp is not None:
                        if self.downramp.nom_energy_gain is None:
                            isDef = False
                        else:
                            dE += self.downramp.nom_energy_gain
                    if isDef:
                        self._printVerb("nom_energy_gain            (3)>",dE)
                        self._nom_energy_gain_calc = dE
                        updateCounter += 1

            if self.nom_energy_gain_flattop is None:
                if self.nom_energy_gain is not None:
                    dE = self.nom_energy_gain
                    isDef = True
                    if self.upramp is not None:
                        if self.upramp.nom_energy_gain is None:
                            isDef = False
                        else:
                            dE -= self.upramp.nom_energy_gain
                    if self.downramp is not None:
                        if self.downramp.nom_energy_gain is None:
                            isDef = False
                        else:
                            dE -= self.downramp.nom_energy_gain
                    if isDef:
                        self._printVerb("nom_energy_gain_flattop    (3)>",dE)
                        self._nom_energy_gain_flattop_calc = dE
                        updateCounter += 1

            #Note:
            #   Nom_accel_gradient from flattop+upramp+downramp gradients is implicitly set
            #       via flattop energy and length, which then sets global energy and length.
            #       G_total = E_total/L_total = (E_up+E_flat+E_dn)/(L_up+L_flat+L_dn)
            #   Nom_accel_gradient_flattop works the same way
            #       G_flat = E_flat/L_flat = (E_total-E_up-E_dn)/L_flat
            #   However this only works if this objects has ramps; if not then just copy flattop<->overall
            if self.nom_accel_gradient is None:
                if self.nom_accel_gradient_flattop is not None and \
                    self.upramp is None and self.downramp is None:
                    self._printVerb("nom_accel_gradient         (3)>",self.nom_accel_gradient_flattop)
                    self._nom_accel_gradient_calc = self.nom_accel_gradient_flattop
                    updateCounter += 1
            if self.nom_accel_gradient_flattop is None:
                if self.nom_accel_gradient is not None and \
                    self.upramp is None and self.downramp is None:
                    self._printVerb("nom_accel_gradient_flattop (3)>",self.nom_accel_gradient)
                    self._nom_accel_gradient_flattop_calc = self.nom_accel_gradient
                    updateCounter += 1

            #Use data from parent to update itself (i.e. if self is a ramp)
            if self.parent is not None:
                if self.length is None:
                    if self.parent.length is not None and \
                       self.parent.length_flattop is not None:

                        otherRamp = self.parent._getOtherRamp(self)
                        if otherRamp is not None:
                            if otherRamp.length is not None:
                                L = self.parent.length - self.parent.length_flattop - otherRamp.length
                                if L < 0.0 and self.sanityCheckLengths:
                                    raise VariablesOutOfRangeError(f"Calculated length = {L} < 0; check variables or disable check by setting stage.sanityCheckLengths=False.")
                                self._length_calc = L
                                self._printVerb("length          (4)>", self.length)
                                updateCounter += 1

                if self.nom_energy_gain is None:
                    if self.parent.nom_energy_gain is not None and \
                       self.parent.nom_energy_gain_flattop is not None:

                        otherRamp = self.parent._getOtherRamp(self)
                        if otherRamp is not None:
                            if otherRamp.nom_energy_gain is not None:
                                self._nom_energy_gain_calc = self.parent.nom_energy_gain - self.parent.nom_energy_gain_flattop - otherRamp.nom_energy_gain
                                self._printVerb("nom_energy_gain (4)>", self.nom_energy_gain)
                                updateCounter += 1

            #Dig down and try to calculate more
            if self.upramp is not None:
                self._printVerb("-> upramp")
                updateCounter += self.upramp._recalcLengthEnergyGradient_helper()
                self._printVerb("<-")
            if self.downramp is not None:
                self._printVerb("-> downramp")
                updateCounter += self.downramp._recalcLengthEnergyGradient_helper()
                self._printVerb("<-")

            #Are we done yet?
            updateCounter_total += updateCounter
            if updateCounter == 0:
                self._printVerb("[break]")
                break

        return updateCounter_total


    # ==================================================
    doVerbosePrint_debug = False
    def _printVerb(self, *args, **kwargs):
        "Print() if doVerbosePrint_debug == True, else NOP."
        if self.doVerbosePrint_debug:
            print(*args, **kwargs)


    # ==================================================
    def _printLengthEnergyGradient_internal(stage):
        "For debugging"
        print("parent/upramp/downramp:    ", stage.parent, stage.upramp, stage.downramp)
        print("length:                    ", stage._length, stage._length_calc)
        print("length_flattop:            ", stage._length_flattop, stage._length_flattop_calc)
        print("nom_energy_gain:           ", stage._nom_energy_gain, stage._nom_energy_gain_calc)
        print("nom_energy_gain_flattop:   ", stage._nom_energy_gain_flattop, stage._nom_energy_gain_flattop_calc)
        print("nom_accel_gradient:        ", stage._nom_accel_gradient, stage._nom_accel_gradient_calc)
        print("nom_accel_gradient_flattop:", stage._nom_accel_gradient_flattop, stage._nom_accel_gradient_flattop_calc)
        print()



    ## Various calculations / plots / etc

    # ==================================================
    def get_cost_breakdown(self):
        breakdown = []
        breakdown.append(('Plasma cell', self.get_length() * CostModeled.cost_per_length_plasma_stage))
        #breakdown.append(('Driver dump', CostModeled.cost_per_driver_dump))
        return (self.name, breakdown)


    # ==================================================
    def matched_beta_function(self, energy_incoming):
        if self.ramp_beta_mag is not None:
            return beta_matched(self.plasma_density, energy_incoming)*self.ramp_beta_mag
        else:
            return beta_matched(self.plasma_density, energy_incoming)

    
    # ==================================================
    def matched_beta_function_flattop(self, energy):
        return beta_matched(self.plasma_density, energy)
    

    # ==================================================
    def energy_usage(self):
        return self.driver_source.energy_usage()
    

    # ==================================================
    def energy_efficiency(self):
        return self.efficiency


    # ==================================================
    #@abstractmethod   # TODO: calculate the dumped power and use it for the dump cost model.
    def dumped_power(self):
        return self.efficiency.dumped_power


    # ==================================================
    def calculate_efficiency(self, beam0, driver0, beam, driver):
        Etot0_beam = beam0.total_energy()
        Etot_beam = beam.total_energy()
        Etot0_driver = driver0.total_energy()
        Etot_driver = driver.total_energy()
        self.efficiency.driver_to_wake = (Etot0_driver-Etot_driver)/Etot0_driver
        self.efficiency.wake_to_beam = (Etot_beam-Etot0_beam)/(Etot0_driver-Etot_driver)
        self.efficiency.driver_to_beam = self.efficiency.driver_to_wake*self.efficiency.wake_to_beam
        if self.get_rep_rate_average() is not None:
            self.efficiency.dumped_power = Etot_driver*self.get_rep_rate_average()
        else:    
            self.efficiency.dumped_power = None


    # ==================================================
    def calculate_beam_current(self, beam0, driver0, beam=None, driver=None):
        
        dz = 40*np.mean([driver0.bunch_length(clean=True)/np.sqrt(len(driver0)), beam0.bunch_length(clean=True)/np.sqrt(len(beam0))])
        num_sigmas = 6
        z_min = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        z_max = driver0.z_offset() + num_sigmas * driver0.bunch_length()
        tbins = np.arange(z_min, z_max, dz)/SI.c
        
        Is0, ts0 = (driver0 + beam0).current_profile(bins=tbins)
        self.initial.beam.current.zs = ts0*SI.c
        self.initial.beam.current.Is = Is0

        if beam is not None and driver is not None:
            Is, ts = (driver + beam).current_profile(bins=tbins)
            self.final.beam.current.zs = ts*SI.c
            self.final.beam.current.Is = Is

    
    # ==================================================
    def save_driver_to_file(self, driver, runnable):
        driver.save(runnable, beam_name='driver_stage' + str(driver.stage_number+1))

    
    # ==================================================
    def save_evolution_to_file(self, bunch='beam'):
    
        # select bunch
        if bunch == 'beam':
            evol = self.evolution.beam
        elif bunch == 'driver':
            evol = self.evolution.driver

        # arrange numbers into a matrix
        matrix = np.empty((len(evol.location),14))
        matrix[:,0] = evol.location
        matrix[:,1] = evol.charge
        matrix[:,2] = evol.energy
        matrix[:,3] = evol.x
        matrix[:,4] = evol.y
        matrix[:,5] = evol.rel_energy_spread
        matrix[:,6] = evol.rel_energy_spread_fwhm
        matrix[:,7] = evol.beam_size_x
        matrix[:,8] = evol.beam_size_y
        matrix[:,9] = evol.emit_nx
        matrix[:,10] = evol.emit_ny
        matrix[:,11] = evol.beta_x
        matrix[:,12] = evol.beta_y
        matrix[:,13] = evol.peak_spectral_density

        # save to CSV file
        filename = bunch + '_evolution.csv'
        np.savetxt(filename, matrix, delimiter=',')
        
    
    # ==================================================
    def plot_driver_evolution(self):
        self.plot_evolution(bunch='driver')
    

    # ==================================================
    def plot_evolution(self, bunch='beam'):

        from matplotlib import pyplot as plt
        
        # select bunch
        if bunch == 'beam':
            evol = copy.deepcopy(self.evolution.beam)
        elif bunch == 'driver':
            evol = copy.deepcopy(self.evolution.driver)
            
        # extract wakefield if not already existing
        if not hasattr(evol, 'location'):
            print('No evolution calculated')
            return

        # add upramp evolution
        if self.upramp is not None and hasattr(self.upramp.evolution.beam, 'location'):
            if bunch == 'beam':
                upramp_evol = self.upramp.evolution.beam
            elif bunch == 'driver':
                upramp_evol = self.upramp.evolution.driver
            evol.location = np.append(upramp_evol.location, evol.location-np.min(evol.location)+np.max(upramp_evol.location))
            evol.energy = np.append(upramp_evol.energy, evol.energy)
            evol.charge = np.append(upramp_evol.charge, evol.charge)
            evol.emit_nx = np.append(upramp_evol.emit_nx, evol.emit_nx)
            evol.emit_ny = np.append(upramp_evol.emit_ny, evol.emit_ny)
            evol.rel_energy_spread = np.append(upramp_evol.rel_energy_spread, evol.rel_energy_spread)
            evol.beam_size_x = np.append(upramp_evol.beam_size_x, evol.beam_size_x)
            evol.beam_size_y = np.append(upramp_evol.beam_size_y, evol.beam_size_y)
            evol.bunch_length = np.append(upramp_evol.bunch_length, evol.bunch_length)
            evol.x = np.append(upramp_evol.x, evol.x)
            evol.y = np.append(upramp_evol.y, evol.y)
            evol.z = np.append(upramp_evol.z, evol.z)
            evol.beta_x = np.append(upramp_evol.beta_x, evol.beta_x)
            evol.beta_y = np.append(upramp_evol.beta_y, evol.beta_y)
            evol.plasma_density = np.append(upramp_evol.plasma_density, evol.plasma_density)

        # add downramp evolution
        if self.downramp is not None and hasattr(self.downramp.evolution.beam, 'location'):
            if bunch == 'beam':
                downramp_evol = self.downramp.evolution.beam
            elif bunch == 'driver':
                downramp_evol = self.downramp.evolution.driver
            evol.location = np.append(evol.location, downramp_evol.location-np.min(downramp_evol.location)+np.max(evol.location))
            evol.energy = np.append(evol.energy, downramp_evol.energy)
            evol.charge = np.append(evol.charge, downramp_evol.charge)
            evol.emit_ny = np.append(evol.emit_ny, downramp_evol.emit_ny)
            evol.emit_nx = np.append(evol.emit_nx, downramp_evol.emit_nx)
            evol.rel_energy_spread = np.append(evol.rel_energy_spread, downramp_evol.rel_energy_spread)
            evol.beam_size_x = np.append(evol.beam_size_x, downramp_evol.beam_size_x)
            evol.beam_size_y = np.append(evol.beam_size_y, downramp_evol.beam_size_y)
            evol.bunch_length = np.append(evol.bunch_length, downramp_evol.bunch_length)
            evol.x = np.append(evol.x, downramp_evol.x)
            evol.y = np.append(evol.y, downramp_evol.y)
            evol.z = np.append(evol.z, downramp_evol.z)
            evol.beta_x = np.append(evol.beta_x, downramp_evol.beta_x)
            evol.beta_y = np.append(evol.beta_y, downramp_evol.beta_y)
            evol.plasma_density = np.append(evol.plasma_density, downramp_evol.plasma_density)
        
        # preprate plot
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]

        # plot energy
        axs[0,0].plot(evol.location, evol.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy [GeV]')
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_xlim(long_limits)
        
        # plot charge
        axs[0,1].plot(evol.location, abs(evol.charge[0]) * np.ones(evol.location.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(evol.location, abs(evol.charge) * 1e9, color=col1)
        axs[0,1].set_ylabel('Charge [nC]')
        axs[0,1].set_xlim(long_limits)
        axs[0,1].set_ylim(0, abs(evol.charge[0]) * 1.3 * 1e9)
        
        # plot normalized emittance
        axs[0,2].plot(evol.location, evol.emit_ny*1e6, color=col2)
        axs[0,2].plot(evol.location, evol.emit_nx*1e6, color=col1)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_xlim(long_limits)
        axs[0,2].set_yscale('log')
        
        # plot energy spread
        axs[1,0].plot(evol.location, evol.rel_energy_spread*1e2, color=col1)
        axs[1,0].set_ylabel('Energy spread, rms [%]')
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_xlim(long_limits)
        axs[1,0].set_yscale('log')

        # plot bunch length
        axs[1,1].plot(evol.location, evol.bunch_length*1e6, color=col1)
        axs[1,1].set_ylabel(r'Bunch length, rms [$\mathrm{\mu}$m]')
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_xlim(long_limits)

        # plot beta function
        axs[1,2].plot(evol.location, evol.beta_y*1e3, color=col2)  
        axs[1,2].plot(evol.location, evol.beta_x*1e3, color=col1)
        axs[1,2].set_ylabel('Beta function [mm]')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)
        axs[1,2].set_yscale('log')
        
        # plot longitudinal offset
        axs[2,0].plot(evol.location, evol.plasma_density / 1e6, color=col1)
        axs[2,0].set_ylabel(r'Plasma density [$\mathrm{cm}^{-3}$]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        axs[2,0].set_yscale('log')
        
        # plot longitudinal offset
        axs[2,1].plot(evol.location, evol.z * 1e6, color=col1)
        axs[2,1].set_ylabel(r'Longitudinal offset [$\mathrm{\mu}$m]')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_xlim(long_limits)
        
        # plot transverse offset
        axs[2,2].plot(evol.location, np.zeros(evol.location.shape), ':', color=col0)
        axs[2,2].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,2].plot(evol.location, evol.x*1e6, color=col1)
        axs[2,2].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)


        if self.stage_number is not None:
            fig.suptitle('Stage ' + str(self.stage_number+1) + ', ' + bunch)
        
        plt.show()


    # ==================================================
    def plot_spin_evolution(self, bunch='beam'):

        from matplotlib import pyplot as plt
        
        # select bunch
        if bunch == 'beam':
            evol = copy.deepcopy(self.evolution.beam)
        elif bunch == 'driver':
            evol = copy.deepcopy(self.evolution.driver)
            
        # extract wakefield if not already existing
        if not hasattr(evol, 'location'):
            print('No evolution calculated')
            return

        if not hasattr(evol, 'spin_x') or evol.spin_x is None:
            print('No spin evolution calculated')
            return
        
        # add upramp evolution
        if self.upramp is not None and hasattr(self.upramp.evolution.beam, 'location'):
            if bunch == 'beam':
                upramp_evol = self.upramp.evolution.beam
            elif bunch == 'driver':
                upramp_evol = self.upramp.evolution.driver
            evol.location = np.append(upramp_evol.location, evol.location-np.min(evol.location)+np.max(upramp_evol.location))
            evol.spin_x = np.append(upramp_evol.spin_x, evol.spin_x)
            evol.spin_y = np.append(upramp_evol.spin_y, evol.spin_y)
            evol.spin_z = np.append(upramp_evol.spin_z, evol.spin_z)
            

        # add downramp evolution
        if self.downramp is not None and hasattr(self.downramp.evolution.beam, 'location'):
            if bunch == 'beam':
                downramp_evol = self.downramp.evolution.beam
            elif bunch == 'driver':
                downramp_evol = self.downramp.evolution.driver
            evol.location = np.append(evol.location, downramp_evol.location-np.min(downramp_evol.location)+np.max(evol.location))
            evol.spin_x = np.append(evol.spin_x, downramp_evol.spin_x)
            evol.spin_y = np.append(evol.spin_y, downramp_evol.spin_y)
            evol.spin_z = np.append(evol.spin_z, downramp_evol.spin_z)
        
        # preprate plot
        fig, axs = plt.subplots(1,1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(CONFIG.plot_width_default*0.5)
        colx = "tab:blue"
        coly = "tab:orange"
        colz = "tab:green"
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]

        # plot energy
        axs.plot(evol.location, evol.spin_x, color=colx, label='x')
        axs.plot(evol.location, evol.spin_y, color=coly, label='y')
        axs.plot(evol.location, evol.spin_z, color=colz, label='z')
        axs.set_ylabel('Spin polarization')
        axs.set_xlabel(long_label)
        axs.set_xlim(long_limits)
        axs.set_ylim(-1.02, 1.02)
        axs.legend()
        
        if self.stage_number is not None:
            fig.suptitle('Stage ' + str(self.stage_number+1) + ', ' + bunch)
        
        plt.show()


    # ==================================================  
    def plot_wakefield(self):

        from matplotlib import pyplot as plt
        
        # extract wakefield if not already existing
        if not hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'):
            print('No wakefield calculated')
            return
        if not hasattr(self.initial.beam.current, 'Is'):
            print('No beam current calculated')
            return

        # preprate plot
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*1)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        
        # extract wakefields and beam currents
        zs0 = self.initial.plasma.wakefield.onaxis.zs
        Ezs0 = self.initial.plasma.wakefield.onaxis.Ezs
        has_final = hasattr(self.final.plasma.wakefield.onaxis, 'Ezs')
        if has_final:
            zs = self.final.plasma.wakefield.onaxis.zs
            Ezs = self.final.plasma.wakefield.onaxis.Ezs
        zs_I = self.initial.beam.current.zs
        Is = self.initial.beam.current.Is

        # find field at the driver and beam
        z_mid = zs_I.min() + (zs_I.max()-zs_I.min())*0.3
        mask = zs_I < z_mid
        zs_masked = zs_I[mask]
        z_beam = zs_masked[np.abs(Is[mask]).argmax()]
        Ez_driver = Ezs0[zs0 > z_mid].max()
        Ez_beam = np.interp(z_beam, zs0, Ezs0)
        
        # get wakefield
        axs[0].plot(zs0*1e6, np.zeros(zs0.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs0*1e6, -self.nom_energy_gain/self.length_flattop*np.ones(zs0.shape)/1e9, ':', color=col2)
        if self.driver_source.energy is not None:
            Ez_driver_max = self.driver_source.energy/self.length_flattop
            axs[0].plot(zs0*1e6, Ez_driver_max*np.ones(zs0.shape)/1e9, ':', color=col0)
        if has_final:
            axs[0].plot(zs*1e6, Ezs/1e9, '-', color=col1, alpha=0.2)
        axs[0].plot(zs0*1e6, Ezs0/1e9, '-', color=col1)
        axs[0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[0].set_ylabel('Longitudinal electric field [GV/m]')
        zlims = [min(zs0)*1e6, max(zs0)*1e6]
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-1.7*np.max([np.abs(Ez_beam), Ez_driver])/1e9, top=1.3*Ez_driver/1e9)
        
        # plot beam current
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[1].set_ylabel('Beam current [kA]')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=1.2*min(-Is)/1e3, top=1.2*max(-Is)/1e3)

        
    # ==================================================
    # plot wake
    def plot_wake(self, aspect='equal', show_beam=True, savefig=None):
        """
        Plot the wake structure (2D plot) as a new pyplot.figure.

        Other parameters
        ----------------
        aspect : str
            The aspect ratio of the plots.
            Defaults to 'equal' which is also the matplotlib default; can also use 'auto'.
            Set to 'auto' to plot the entire simulation box.
        savefig : str or None
            If not None, the path to save the figure.
            Defaults to None

        Returns:
        --------
          None
        """

        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm
        
        # extract density if not already existing
        if not hasattr(self.initial.plasma.density, 'rho'):
            print('No wake calculated')
            return
        if not hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'):
            print('No wakefield calculated')
            return
        
        # make figures
        has_final_step = hasattr(self.final.plasma.density, 'rho')
        num_plots = 1 + int(has_final_step)
        fig, ax = plt.subplots(num_plots,1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*0.5*num_plots)

        # cycle through initial and final step
        for i in range(num_plots):
            if not has_final_step:
                ax1 = ax
            else:
                ax1 = ax[i]

            # extract initial or final
            if i==0:
                data_struct = self.initial
                title = 'Initial step'
            elif i==1:
                data_struct = self.final
                title = 'Final step'

            # get data
            extent = data_struct.plasma.density.extent
            zs0 = data_struct.plasma.wakefield.onaxis.zs
            Ezs0 = data_struct.plasma.wakefield.onaxis.Ezs
            rho0_plasma = data_struct.plasma.density.rho
            rho0_beam = data_struct.beam.density.rho

            # find field at the driver and beam
            if i==0:
                zs_I = self.initial.beam.current.zs
                Is = self.initial.beam.current.Is
                z_mid = zs_I.max()-(zs_I.max()-zs_I.min())*0.3
                z_beam = zs_I[np.abs(Is[zs_I < z_mid]).argmax()]
                Ez_driver = Ezs0[zs0 > z_mid].max()
                Ez_beam = np.interp(z_beam, zs0, Ezs0)
                Ezmax = 2.3*1.7*np.max([np.abs(Ez_driver), np.abs(Ez_beam)])
            
            # plot on-axis wakefield and axes
            ax2 = ax1.twinx()
            ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
            ax2.set_ylabel(r'$E_{z}$' ' [GV/m]')
            ax2.set_ylim(bottom=-Ezmax/1e9, top=Ezmax/1e9)
            axpos = ax1.get_position()
            pad_fraction = 0.13  # Fraction of the figure width to use as padding between the ax and colorbar
            cbar_width_fraction = 0.015  # Fraction of the figure width for the colorbar width
    
            # create colorbar axes based on the relative position and size
            if show_beam:
                cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax3 = fig.add_axes([axpos.x1 + pad_fraction + 2*cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            clims = np.array([1e-2, 1e3])*self.plasma_density
            
            # plot plasma ions
            p_ions = ax1.imshow(-rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Greens', alpha=np.array(-rho0_plasma>clims.min(), dtype=float), aspect=aspect)
            p_ions.set_clim(clims/1e6)
            cb_ions = plt.colorbar(p_ions, cax=cax3)
            cb_ions.set_label(label=r'Beam/plasma-electron/ion density [$\mathrm{cm^{-3}}$]', size=10)
            cb_ions.ax.tick_params(axis='y',which='both', direction='in')
            
            # plot plasma electrons
            p_electrons = ax1.imshow(rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Blues', alpha=np.array(rho0_plasma>clims.min()*2, dtype=float), aspect=aspect)
            p_electrons.set_clim(clims/1e6)
            cb_electrons = plt.colorbar(p_electrons, cax=cax2)
            cb_electrons.ax.tick_params(axis='y',which='both', direction='in')
            cb_electrons.set_ticklabels([])
            
            # plot beam electrons
            if show_beam:
                p_beam = ax1.imshow(rho0_beam/1e6, extent=extent*1e6,  norm=LogNorm(), origin='lower', cmap='Oranges', alpha=np.array(rho0_beam>clims.min()*2, dtype=float), aspect=aspect)
                p_beam.set_clim(clims/1e6)
                cb_beam = plt.colorbar(p_beam, cax=cax1)
                cb_beam.set_ticklabels([])
                cb_beam.ax.tick_params(axis='y', which='both', direction='in')
            
            # set labels
            if i==(num_plots-1):
                ax1.set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            ax1.set_ylabel(r'$x$ [$\mathrm{\mu}$m]')
            ax1.set_title(title)
            ax1.grid(False)
            ax2.grid(False)
            
        # save the figure
        if savefig is not None:
            fig.savefig(str(savefig), format="pdf", bbox_inches="tight")
        
        return 

    
    # ==================================================
    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'Plasma stage'
        color = 'red'
        return x_points, y_points, final_angle, label, color
        

class VariablesOverspecifiedError(Exception):
    "Exception class to throw when trying to set too many overlapping variables."
    pass
class VariablesOutOfRangeError(Exception):
    "Exception class to throw when calculated or set variables are out of allowed range."
class StageError(Exception):
    "Exception class for ``Stage`` to throw in other cases."

class SimulationDomainSizeError(Exception):
    "Exception class to throw when the simulation domain size is too small."
    pass

    
