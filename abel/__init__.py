# suppress numba warnings from Ocelot
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# import configuration class
from .CONFIG import CONFIG

# import all other classes
from .classes.beam import Beam
from .classes.event import Event
from .classes.trackable import Trackable
from .classes.runnable import Runnable

from .classes.ip.ip import InteractionPoint
from .classes.beamline.beamline import Beamline

from .classes.source.source import Source
from .classes.stage.stage import Stage
from .classes.plasma_lens.plasma_lens import PlasmaLens
from .classes.interstage.interstage import Interstage
from .classes.rf_accelerator.rf_accelerator import RFAccelerator
from .classes.bds.bds import BeamDeliverySystem
from .classes.spectrometer.spectrometer import Spectrometer
from .classes.driver_complex.driver_complex import DriverComplex

from .classes.beamline.impl.linac.linac import Linac
from .classes.beamline.impl.linac.impl.plasma_linac import PlasmaLinac
from .classes.beamline.impl.linac.impl.conventional_linac import ConventionalLinac
from .classes.beamline.impl.experiment.experiment import Experiment
from .classes.beamline.impl.experiment.impl.experiment_pwfa import ExperimentPWFA
from .classes.beamline.impl.experiment.impl.experiment_apl import ExperimentAPL

from .classes.RFlinac.RFlinac import *
from .classes.RFlinac.impl.RFlinac_CLICG import *
from .classes.RFlinac.impl.RFlinac_CLIC502 import *


from .classes.collider.collider import Collider

from .classes.source.impl.source_basic import *
from .classes.source.impl.source_trapezoid import *
from .classes.source.impl.source_combiner import *
from .classes.source.impl.source_from_file import *
from .classes.stage.impl.stage_basic import *
from .classes.stage.impl.stage_nonlinear_1d import *
from .classes.stage.impl.stage_hipace import *
from .classes.stage.impl.stage_wake_t import *
from .classes.stage.impl.stage_quasistatic_2d import *
from .classes.stage.impl.stage_slice_transverse_wake_instability import *
from .classes.stage.impl.stage_particle_transverse_wake_instability import *
from .classes.interstage.impl.interstage_null import *
from .classes.interstage.impl.interstage_basic import *
from .classes.interstage.impl.interstage_elegant import *
from .classes.interstage.impl.interstage_ocelot import *
from .classes.plasma_lens.impl.plasma_lens_basic import *
from .classes.plasma_lens.impl.plasma_lens_nonlinear_thin import *
from .classes.rf_accelerator.impl.rf_accelerator_basic import *
from .classes.driver_complex.impl.driver_complex_basic import *
from .classes.bds.impl.bds_basic import *
from .classes.bds.impl.bds_fbt import *
from .classes.spectrometer.impl.spectrometer_facet_ocelot import *
from .classes.spectrometer.impl.spectrometer_basic_clear import *
from .classes.ip.impl.ip_basic import *
from .classes.ip.impl.ip_guineapig import *
