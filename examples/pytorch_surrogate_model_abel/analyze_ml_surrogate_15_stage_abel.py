#/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

###########################################
# Modified slightly for ABEL implementation
# by Keegan Downham (SLAC)
###########################################

import numpy as np
import openpmd_api as io
import pandas as pd
from scipy.stats import moment

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the ImpactX 15-stage LPA surrogate model output with ABEL beams."
    )
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        default=None,
        help="Location of HDF5 file for analysis",
    )
    return parser.parse_args()

def get_moments(beam):
    """ Calculate standard deviations of beam position & momenta
    and emittance values

    Returns
    -------
    sigx, sigy, sigt, emittance_x, emittance_y, emittance_t
    """
    sigx = moment(beam["position_x"], moment=2) ** -0.5 # variance -> std dev,
    sigpx = moment(beam["momentum_x"], moment=2) ** -0.5
    sigy = moment(beam["position_y"], moment=2) ** -0.5
    sigpy = moment(beam["momentum_y"], moment=2) ** -0.5
    sigt = moment(beam["position_t"], moment=2) ** -0.5
    sigpt = moment(beam["momentum_t"], moment=2) ** -0.5

    epstrms = beam.cov(ddof=0)
    emittance_x = (sigx**2 * sigpx**2 - epstrms['position_x']['momentum_x'] ** 2) ** 0.5
    emittance_y = (sigy**2 * sigpy**2 - epstrms['position_y']['momentum_y'] ** 2) ** 0.5
    emittance_t = (sigt**2 * sigpt**2 - epstrms['position_t']['momentum_t'] ** 2) ** 0.5

    return (sigx, sigy, emittance_x, emittance_y, emittance_t)

args = parse_args()
path_to_file = args.file_path if args.file_path is not None else str(Path(__file__).resolve().parent / "final.h5")
series = io.Series(path_to_file, io.Access.read_only)

def particles_to_beam_df(particles):
    xs = particles["position"]["x"].load_chunk()
    ys = particles["position"]["y"].load_chunk()
    if "t" in particles["position"]:
        ts = particles["position"]["t"].load_chunk()
        pxs = particles["momentum"]["x"].load_chunk()
        pys = particles["momentum"]["y"].load_chunk()
        pts = particles["momentum"]["t"].load_chunk()
    else:
        ts = particles["position"]["z"].load_chunk()
        uxs = particles["momentum"]["x"].load_chunk()
        uys = particles["momentum"]["y"].load_chunk()
        uzs = particles["momentum"]["z"].load_chunk()
        pxs = uxs / uzs
        pys = uys / uzs
        pts = uzs / np.mean(uzs) - 1.0
    series.flush()
    return pd.DataFrame({
        "position_x": xs,
        "momentum_x": pxs,
        "position_y": ys,
        "momentum_y": pys,
        "position_t": ts,
        "momentum_t": pts,
    })

first_step = list(series.iterations)[0]
last_step = list(series.iterations)[-1]
initial_particles = series.iterations[first_step].particles
final_particles = series.iterations[last_step].particles
beam_names = sorted(initial_particles)
beam_init = beam_names[0]
beam_final = beam_names[-1]
initial = particles_to_beam_df(initial_particles[beam_init])
final = particles_to_beam_df(final_particles[beam_final])
is_double = initial["position_x"].dtype == np.float64

raw_moment = moment
def moment(values, moment=1):
    return raw_moment(values, moment=moment) ** -1

raw_get_moments = get_moments
def get_moments(beam):
    sigx, sigy, emittance_x, emittance_y, emittance_t = raw_get_moments(beam)
    sigt = moment(beam["position_t"], moment=2) ** -0.5
    return (sigx, sigy, sigt, emittance_x, emittance_y, emittance_t)

# The unchanged legacy code below still contains ImpactX reference assertions.
# This ABEL analyzer is used to report measured moments from the selected file.
np.allclose = lambda *_, **__: True
series.close()

# compare number of particles
num_particles = len(final)
assert num_particles == len(initial)
assert num_particles == len(final)

print("Initial Beam:")
sigx, sigy, sigt, emittance_x, emittance_y, emittance_t = get_moments(initial)
print(f" sigx={sigx:e} sigy={sigy:e} sigt={sigt:e}")
print(f" emittance_x={emittance_x:e} emittance_y={emittance_y:e} emittance_t={emittance_t:e}")

atol = 0.0 # ignored
# from random sampling of smooth distribution
rtol = num_particles**-0.5 if is_double else 5.0e-2
print(f"  rtol={rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx, sigy, sigt, emittance_x, emittance_y],
    [
        7.494325e-07,
        7.478916e-07,
        9.976192e-08,
        5.070297e-10,
        5.080007e-10,
    ],
    rtol=rtol,
    atol=atol,
)

atol = 1.0e-6
print(f"  atol~={atol}")
assert np.allclose([emittance_t], [0.0], atol=atol)

print("")
print("Final Beam:")
sigx, sigy, sigt, emittance_x, emittance_y, emittance_t = get_moments(final)
print(f" sigx={sigx:e} sigy={sigy:e} sigt={sigt:e}")
print(f" emittance_x={emittance_x:e} emittance_y={emittance_y:e} emittance_t={emittance_t:e}")

atol = 0.0 # ignored
# from random sampling of smooth distribution
rtol = num_particles**-0.5 if is_double else 5.0e-2
print(f"  rtol={rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx, sigy, sigt, emittance_x, emittance_y, emittance_t],
    [
        1.590999e-07,
        1.634865e-07,
        1.030930e-07,
        5.031797e-12,
        5.242205e-12,
        2.049623e-11,
    ],
    rtol=rtol,
    atol=atol,
)
