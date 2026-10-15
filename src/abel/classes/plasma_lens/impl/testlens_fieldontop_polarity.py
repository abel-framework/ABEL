# =============================================================================
# ABEL plasma-lens tracking example
#
# Adapted to use:
#     PlasmaLensImportedFieldMap
#
# ABEL 2026.3.0
# ImpactX 25.12
#
# The imported field map contains:
#
#     x   y   z   Bx(t0)   Bx(t1)   Bx(t2) ...
#     x   y   z   By(t0)   By(t1)   By(t2) ...
#
# One discharge timestep is selected and the corresponding transverse
# field map is used throughout the complete plasma-lens length. A uniform
# +Y field of 300 mT is added to that imported field (Pierre sims input)
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

from abel import *
import numpy as np
import matplotlib.pyplot as plt


# Import your custom field-map plasma lens class.
#
# If the class is in a separate Python file, e.g.
#
#     plasma_lens_imported_field_map.py
#
# then use:
#
#     from plasma_lens_imported_field_map import PlasmaLensImportedFieldMap
#
# If the class is defined in an earlier notebook cell, this import is not
# required.

from abel.classes.plasma_lens.impl.retired.plasma_lens_from_file import PlasmaLensImportedFieldMap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

# =============================================================================
# FIELD MAP FILES
# =============================================================================
# Folder containing this Python script
HERE = Path(__file__).resolve().parent
BX_FILE = HERE / "Bx.txt"
BY_FILE = HERE / "By.txt"


# =============================================================================
# IMPORTED FIELD-MAP PARAMETERS
# =============================================================================

# Plasma lens length [m]
PLASMA_LENS_LENGTH = 0.02

# Radius over which particles are tracked [m]
#
# This must not be larger than the radius covered by the imported field map.
PLASMA_LENS_RADIUS = 500e-6

# Number of longitudinal slices used by the imported-lens tracker.
#
# The transverse Bx/By map is repeated along the full lens length.
NUM_SLICES = 30

# The field-map time values are in nanoseconds. Select the 195 ns snapshot.
#
# If None, the first available timestep is selected automatically.
# The code will select the closest timestep available in the field map.
DISCHARGE_TIME_NS = 195.0

# Convert nanoseconds in the field-map header to seconds for the lens class.
TIME_SCALE = 1e-9
DISCHARGE_TIME = DISCHARGE_TIME_NS * TIME_SCALE

# Uniform background field added to the imported map [T].
UNIFORM_BX = 0.0
UNIFORM_BY = 300e-3  # Set to 0.0 to disable the added field.

# Imported-map polarity: +1 uses the file values; -1 reverses Bx and By.
# The uniform background field keeps its configured direction either way.
FIELD_POLARITY = -1 

# Position offset of the plasma lens relative to the field-map origin.
OFFSET_X = 0.0
OFFSET_Y = 0.0


# =============================================================================
# DEFINE THE BEAM
# =============================================================================

source = SourceBasic()

source.charge = -100e-12          # [C]
source.energy = 50e6              # [eV]
source.rel_energy_spread = 0.01

source.bunch_length = 400e-6      # [m]

source.emit_nx = 5e-6             # [m rad]
source.emit_ny = 5e-6             # [m rad]

source.beta_x = 10               # [m]
source.beta_y = source.beta_x

source.num_particles = 10000

source.length = 3                 # [m]

# Initial position jitter
source.jitter.x = 100e-6          # [m]
source.jitter.y = 100e-6          # [m]

# Initial angular jitter
source.jitter.xp = 0.01e-3        # [rad]
source.jitter.yp = 0.01e-3        # [rad]


# =============================================================================
# DEFINE THE RF ACCELERATOR
# =============================================================================

rf_accelerator = RFAcceleratorBasic()

rf_accelerator.nom_energy_gain = 200e6     # [eV]
rf_accelerator.length = 200                # [m]


# =============================================================================
# DEFINE THE BEAM DELIVERY SYSTEM
# =============================================================================

bds = BeamDeliverySystemBasic()

bds.beta_x = 0.2      # [m]
bds.beta_y = 0.2      # [m]

bds.length = 5        # [m]


# =============================================================================
# DEFINE THE LINAC
# =============================================================================

linac = ConventionalLinac()

linac.source = source
linac.rf_accelerator = rf_accelerator
linac.bds = bds


# =============================================================================
# DEFINE THE PLASMA LENS USING THE IMPORTED FIELD MAP
# =============================================================================

plasma_lens = PlasmaLensImportedFieldMap(

    # Field-map files
    bx_file=BX_FILE,
    by_file=BY_FILE,

    # Select discharge timestep
    discharge_time=DISCHARGE_TIME,

    # Plasma lens geometry
    length=PLASMA_LENS_LENGTH,
    radius=PLASMA_LENS_RADIUS,

    # Longitudinal slicing
    num_slice=NUM_SLICES,

    # Field-map/lens alignment
    offset_x=OFFSET_X,
    offset_y=OFFSET_Y,

    # Time conversion
    time_scale=TIME_SCALE,

    # Add a uniform 300 mT field in +Y to the imported field map.
    uniform_bx=UNIFORM_BX,
    uniform_by=UNIFORM_BY,
    # Flip only the imported map with -1; leave the uniform field unchanged.
    field_polarity=FIELD_POLARITY,
)


# =============================================================================
# PRINT INFORMATION ABOUT THE IMPORTED FIELD MAP
# =============================================================================

print()
print("Available discharge times [ns]:")
print(plasma_lens.get_discharge_times() * 1e9)

print()
print("Currently selected discharge time [ns]:")
print(plasma_lens.selected_time * 1e9)

print()
print("Field-map radius:")
print(plasma_lens.get_map_radius())

print()
print("Tracking radius:")
print(plasma_lens.get_radius())

print()
print("Field-map nonlinearity:")
print(plasma_lens.get_relativistic_nonlinearity())


# Print the complete field information
plasma_lens.field_info()


# =============================================================================
# PLOT THE IMPORTED FIELD AND THE FIELD WITH THE UNIFORM FIELD ADDED
# =============================================================================

x_map, y_map, Bx_total, By_total, selected_time = (
    plasma_lens.get_field_map()
)

Bx_imported = Bx_total - plasma_lens.uniform_bx
By_imported = By_total - plasma_lens.uniform_by
B_imported = np.sqrt(Bx_imported**2 + By_imported**2)
B_total = np.sqrt(Bx_total**2 + By_total**2)

print(
    "Maximum imported/total |B| difference [T]:",
    np.max(np.abs(B_total - B_imported)),
)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_fields = [
    (Bx_imported, r"Imported $B_x$", "$B_x$ [T]"),
    (By_imported, r"Imported $B_y$", "$B_y$ [T]"),
    (B_imported, r"Imported $|B|$", "$|B|$ [T]"),
    (Bx_total, r"Total $B_x$", "$B_x$ [T]"),
    (By_total, r"Total $B_y$", "$B_y$ [T]"),
    (B_total, r"Total $|B|$", "$|B|$ [T]"),
]

for ax, (field, title, colorbar_label) in zip(axes.flat, plot_fields):
    sc = ax.scatter(x_map * 1e6, y_map * 1e6, c=field, s=10)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"{title}, t={selected_time * 1e9:g} ns")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label=colorbar_label)

external_field_strength = np.hypot(
    plasma_lens.uniform_bx,
    plasma_lens.uniform_by,
)
if external_field_strength == 0.0:
    external_field_label = "Added external field: 0 mT (orientation undefined)"
else:
    external_field_angle = np.degrees(
        np.arctan2(plasma_lens.uniform_by, plasma_lens.uniform_bx)
    )
    external_field_label = (
        f"Added external field: {external_field_strength * 1e3:.1f} mT, "
        f"{external_field_angle:.1f}° counterclockwise from +X"
    )

fig.suptitle(
    f"Imported-map polarity: {plasma_lens.field_polarity:+d}\n"
    f"{external_field_label}"
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

plt.show()

# =============================================================================
# DEFINE THE SPECTROMETER
# =============================================================================

spectrometer = SpectrometerBasicCLEAR()

spectrometer.use_otr_screen = True


# =============================================================================
# ASSEMBLE EXPERIMENT
# =============================================================================

experiment = ExperimentAPL()

experiment.linac = linac
experiment.plasma_lens = plasma_lens
experiment.spectrometer = spectrometer


# =============================================================================
# SCAN WITH THE IMPORTED FIELD ONLY
# =============================================================================

scan_values = np.linspace(-plasma_lens.radius, plasma_lens.radius, 30)
offset_scan = lambda obj, val: (
    setattr(obj.plasma_lens, "offset_x", 0.0)
    or setattr(obj.plasma_lens, "offset_y", val)
    or obj
)

# Baseline: imported map with no uniform field added.
experiment.plasma_lens.uniform_bx = 0.0
experiment.plasma_lens.uniform_by = 0.0
experiment.scan(
    "plasmalens_imported_field_only_scan",
    offset_scan,
    scan_values,
    label="Plasma lens offset, vertical (um)",
    scale=1e-6,
    num_shots_per_step=1,
    parallel=False,
    overwrite=True,
)
imported_scan = experiment.load(shot=0)


# =============================================================================
# SCAN WITH THE IMPORTED FIELD PLUS THE CONFIGURED UNIFORM FIELD
# =============================================================================

experiment.plasma_lens.uniform_bx = UNIFORM_BX
experiment.plasma_lens.uniform_by = UNIFORM_BY
experiment.scan(
    "plasmalens_total_field_scan",
    offset_scan,
    scan_values,
    label="Plasma lens offset, vertical (um)",
    scale=1e-6,
    num_shots_per_step=1,
    parallel=False,
    overwrite=True,
)


# =============================================================================
# PLOT BOTH SCANS
# =============================================================================

plot_functions = [
    (Beam.x_offset, "x offset [um]", 1e-6),
    (Beam.y_offset, "y offset [um]", 1e-6),
    (Beam.abs_charge, "Charge [nC]", 1e-9),
]

for beam_function, y_label, y_scale in plot_functions:
    imported_values = imported_scan.scan_steps_beam_params(beam_function).mean(axis=0)
    total_values = experiment.scan_steps_beam_params(beam_function).mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(
        scan_values / 1e-6,
        imported_values / y_scale,
        "o-",
        label="Imported field",
    )
    ax.plot(
        scan_values / 1e-6,
        total_values / y_scale,
        "s--",
        label="Total field",
    )
    ax.set_xlabel("Plasma lens offset, vertical [um]")
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()

# Plot each scan's y-profile waterfall separately.
imported_scan.plot_waterfall_y()
plt.title("Imported field")
experiment.plot_waterfall_y()
plt.title("Total field")
plt.show()
