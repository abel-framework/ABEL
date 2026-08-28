#!/usr/bin/env python3
#
# ABEL version of the ImpactX 15-stage PyTorch surrogate example:
# https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
import openpmd_api as io
import scipy.constants as SI
import scipy.optimize as opt
import torch

from abel import Beam, Trackable
from surrogate_model_definitions import surrogate_model


MODEL_URL = "https://zenodo.org/records/10810754/files/models.zip?download=1"

REF_U = 1957.0
ENERGY_GAMMA = np.sqrt(1.0 + REF_U**2)
ENERGY_EV = 0.510998950e6 * ENERGY_GAMMA
BUNCH_CHARGE_C = -10.0e-15

EBEAM_LPA_Z0 = -107e-6
L_PLASMA = 0.28
L_TRANSPORT = 0.03
L_STAGE_PERIOD = L_PLASMA + L_TRANSPORT
DRIFT_AFTER_LPA = 43e-6
L_SURROGATE = abs(EBEAM_LPA_Z0) + L_PLASMA + DRIFT_AFTER_LPA

L_LENS = 0.003
L_FOCAL = 0.5 * L_TRANSPORT
L_DRIFT = 0.5 * (L_TRANSPORT - L_LENS)
L_DRIFT_1 = L_DRIFT - DRIFT_AFTER_LPA
L_DRIFT_BEFORE_2ND_STAGE = abs(EBEAM_LPA_Z0)
L_DRIFT_2 = L_DRIFT - L_DRIFT_BEFORE_2ND_STAGE
K_INITIAL = np.sqrt(2.0 / L_FOCAL / L_LENS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ImpactX 15-stage LPA surrogate models with ABEL beams."
    )
    parser.add_argument(
        "--num-particles",
        "-N",
        type=int,
        default=100_000,
        help="number of ABEL macroparticles to track",
    )
    parser.add_argument(
        "--n-stages",
        "-ns",
        type=int,
        default=15,
        choices=range(1, 16),
        metavar="{1..15}",
        help="number of LPA surrogate stages to run",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="directory containing beam_stage_<i>_model.pt files",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail if model files are absent instead of downloading them from Zenodo",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for model evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="number of particles per Torch surrogate batch",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="number of Torch CPU threads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="NumPy random seed for the generated ABEL source",
    )
    parser.add_argument(
        "--save-h5",
        type=Path,
        default=None,
        help="optional OpenPMD/HDF5 file containing the initial beam and plasma-stage exit beams",
    )
    return parser.parse_args()


def ensure_models(models_dir: Path, n_stages: int, download: bool) -> None:
    missing = [
        models_dir / f"beam_stage_{stage_i}_model.pt" for stage_i in range(n_stages)
    ]
    missing = [model_path for model_path in missing if not model_path.exists()]
    if not missing:
        return

    if not download:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing surrogate model files:\n{missing_text}")

    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir.parent / "models.zip"
    print("Downloading trained surrogate models from Zenodo...")
    request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_dataset:
        zip_dataset.extractall(models_dir.parent)

    missing_after_download = [
        models_dir / f"beam_stage_{stage_i}_model.pt" for stage_i in range(n_stages)
    ]
    missing_after_download = [
        model_path for model_path in missing_after_download if not model_path.exists()
    ]
    if missing_after_download:
        missing_text = "\n".join(str(path) for path in missing_after_download)
        raise FileNotFoundError(
            "Model archive downloaded, but expected files are still missing:\n"
            f"{missing_text}"
        )


def select_device(name: str) -> torch.device | None:
    if name == "cpu":
        return None
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda is unavailable.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return None


def make_initial_beam(num_particles: int) -> Beam:
    # Match the ImpactX distribution.Gaussian inputs used in
    # run_ml_surrogate_15_stage.py. For zero correlations, lambdaX/Y/T and
    # lambdaPx/Py/Pt are the RMS sizes and normalized momenta/angles.
    lambda_x = 0.75e-6
    lambda_y = 0.75e-6
    lambda_z = 0.1e-6
    lambda_px = 1.33 / ENERGY_GAMMA
    lambda_py = 1.33 / ENERGY_GAMMA
    lambda_pt = 1.0e-8

    xs = np.random.normal(loc=0.0, scale=lambda_x, size=num_particles)
    ys = np.random.normal(loc=0.0, scale=lambda_y, size=num_particles)
    zs = np.random.normal(loc=0.0, scale=lambda_z, size=num_particles)
    rel_pxs = np.random.normal(loc=0.0, scale=lambda_px, size=num_particles)
    rel_pys = np.random.normal(loc=0.0, scale=lambda_py, size=num_particles)
    rel_pts = np.random.normal(loc=0.0, scale=lambda_pt, size=num_particles)

    ref_gamma = np.sqrt(1.0 + REF_U**2)
    pxs = rel_pxs * REF_U
    pys = rel_pys * REF_U
    gammas = ref_gamma - rel_pts * REF_U
    pzs = np.sqrt(gammas**2 - 1.0 - pxs**2 - pys**2)

    beam = Beam()
    beam.set_phase_space(
        Q=BUNCH_CHARGE_C,
        xs=xs,
        ys=ys,
        zs=zs,
        uxs=pxs * SI.c,
        uys=pys * SI.c,
        uzs=pzs * SI.c,
    )
    beam.location = 0.0
    beam.trackable_number = 0
    beam.reference_beta_gamma = REF_U
    return beam


def lens_eqn(k: float, lens_length: float, alpha: float, beta: float, gamma: float) -> float:
    return np.tan(k * lens_length) + 2.0 * alpha / (k * beta - gamma / k)


def fixed_s_relative_from_beam(beam, ref_pz: float) -> tuple[np.ndarray, ...]:
    px_abs = beam.uxs() / SI.c
    py_abs = beam.uys() / SI.c
    pz_abs = beam.uzs() / SI.c
    gamma_abs = np.sqrt(1.0 + px_abs**2 + py_abs**2 + pz_abs**2)
    ref_gamma = np.sqrt(1.0 + ref_pz**2)

    return (
        beam.xs().copy(),
        beam.ys().copy(),
        beam.zs().copy(),
        px_abs / ref_pz,
        py_abs / ref_pz,
        (ref_gamma - gamma_abs) / ref_pz,
    )


def set_beam_from_fixed_s_relative(
    beam,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    rel_pxs: np.ndarray,
    rel_pys: np.ndarray,
    rel_pts: np.ndarray,
    ref_pz: float,
) -> None:
    ref_gamma = np.sqrt(1.0 + ref_pz**2)
    px_abs = rel_pxs * ref_pz
    py_abs = rel_pys * ref_pz
    gamma_abs = ref_gamma - rel_pts * ref_pz
    pz_abs = np.sqrt(gamma_abs**2 - 1.0 - px_abs**2 - py_abs**2)

    beam.set_xs(xs)
    beam.set_ys(ys)
    beam.set_zs(ts)
    beam.set_uxs(px_abs * SI.c)
    beam.set_uys(py_abs * SI.c)
    beam.set_uzs(pz_abs * SI.c)


def twiss_from_phase_space(
    positions: np.ndarray, momenta: np.ndarray
) -> tuple[float, float, float]:
    cov = np.cov(positions, momenta, ddof=0)
    emittance = np.sqrt(np.linalg.det(cov))
    beta = cov[0, 0] / emittance
    alpha = -cov[1, 0] / emittance
    gamma = cov[1, 1] / emittance
    return alpha, beta, gamma


def fixed_s_to_fixed_t(beam, ref_pz: float) -> np.ndarray:
    ref_pt = -np.sqrt(1.0 + ref_pz**2)
    x_s = beam.xs()
    y_s = beam.ys()
    t_s = beam.zs()
    px_abs = beam.uxs() / SI.c
    py_abs = beam.uys() / SI.c
    pz_abs = beam.uzs() / SI.c
    gamma_abs = np.sqrt(1.0 + px_abs**2 + py_abs**2 + pz_abs**2)
    dpx_s = px_abs / ref_pz
    dpy_s = py_abs / ref_pz
    dpt_s = (np.sqrt(1.0 + ref_pz**2) - gamma_abs) / ref_pz
    denominator = ref_pt + ref_pz * dpt_s

    x_t = x_s + ref_pz * dpx_s * t_s / denominator
    y_t = y_s + ref_pz * dpy_s * t_s / denominator
    t_t = t_s * pz_abs / denominator

    return np.stack([x_t, y_t, t_t, px_abs, py_abs, pz_abs], axis=1)


def fixed_t_to_fixed_s(model_output: np.ndarray, ref_pz: float, ref_z_f: float) -> tuple:
    x_t = model_output[:, 0]
    y_t = model_output[:, 1]
    t_t = model_output[:, 2] - ref_z_f
    px_abs = model_output[:, 3]
    py_abs = model_output[:, 4]
    pz_abs = model_output[:, 5]
    gamma_abs = np.sqrt(1.0 + px_abs**2 + py_abs**2 + pz_abs**2)

    x_s = x_t - px_abs * t_t / pz_abs
    y_s = y_t - py_abs * t_t / pz_abs
    t_s = -gamma_abs * t_t / pz_abs

    return x_s, y_s, t_s, px_abs, py_abs, pz_abs


class ABELDrift(Trackable):
    def __init__(self, length: float, name: str | None = None):
        super().__init__(name=name)
        self.length = float(length)

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        reference_beta_gamma = getattr(beam, "reference_beta_gamma", REF_U)
        xs, ys, ts, rel_pxs, rel_pys, rel_pts = fixed_s_relative_from_beam(
            beam, reference_beta_gamma
        )

        xs += self.length * rel_pxs
        ys += self.length * rel_pys
        ts += self.length * rel_pts / reference_beta_gamma**2

        set_beam_from_fixed_s_relative(
            beam,
            xs,
            ys,
            ts,
            rel_pxs,
            rel_pys,
            rel_pts,
            reference_beta_gamma,
        )
        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self) -> float:
        return self.length


class RetunedConstFLens(Trackable):
    def __init__(
        self,
        length: float,
        tune_axis: str = "x",
        k_initial: float = K_INITIAL,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if tune_axis not in ("x", "y"):
            raise ValueError("tune_axis must be 'x' or 'y'.")
        self.length = float(length)
        self.tune_axis = tune_axis
        self.k = float(k_initial)
        self.k_history: list[float] = []

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        reference_beta_gamma = getattr(beam, "reference_beta_gamma", REF_U)
        xs, ys, ts, rel_pxs, rel_pys, rel_pts = fixed_s_relative_from_beam(
            beam, reference_beta_gamma
        )
        if self.tune_axis == "x":
            alpha, beta, gamma = twiss_from_phase_space(xs, rel_pxs)
        else:
            alpha, beta, gamma = twiss_from_phase_space(ys, rel_pys)

        sol = opt.root_scalar(
            lens_eqn, bracket=[100.0, 300.0], args=(self.length, alpha, beta, gamma)
        )
        self.k = float(sol.root)
        self.k_history.append(self.k)

        phase = self.k * self.length
        cphase = np.cos(phase)
        sphase = np.sin(phase)

        x0 = xs.copy()
        px0 = rel_pxs.copy()
        y0 = ys.copy()
        py0 = rel_pys.copy()
        t0 = ts.copy()
        pt0 = rel_pts.copy()

        xs = cphase * x0 + sphase * px0 / self.k
        rel_pxs = -self.k * sphase * x0 + cphase * px0
        ys = cphase * y0 + sphase * py0 / self.k
        rel_pys = -self.k * sphase * y0 + cphase * py0

        kt = 1.0e-11
        cphase_t = np.cos(kt * self.length)
        sphase_t = np.sin(kt * self.length)
        ts = cphase_t * t0 + sphase_t * pt0 / kt / reference_beta_gamma**2
        rel_pts = -kt * reference_beta_gamma**2 * sphase_t * t0 + cphase_t * pt0

        set_beam_from_fixed_s_relative(
            beam,
            xs,
            ys,
            ts,
            rel_pxs,
            rel_pys,
            rel_pts,
            reference_beta_gamma,
        )

        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self) -> float:
        return self.length


class LPASurrogateStage(Trackable):
    def __init__(
        self,
        stage_i: int,
        model,
        device: torch.device | None,
        batch_size: int,
        length: float = L_SURROGATE,
        stage_start: float = 0.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.stage_i = stage_i
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)
        self.length = float(length)
        self.stage_start = float(stage_start)

    def _tensor(self, data: np.ndarray) -> torch.Tensor:
        if self.device is None:
            return torch.as_tensor(data, dtype=torch.float64)
        return torch.as_tensor(data, dtype=torch.float64, device=self.device)

    def _run_model(self, data: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for start in range(0, data.shape[0], self.batch_size):
                stop = start + self.batch_size
                outputs.append(self.model(data[start:stop]))
        return torch.cat(outputs, dim=0)

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        reference_beta_gamma = getattr(beam, "reference_beta_gamma", REF_U)
        reference_input = np.array(
            [[0.0, 0.0, EBEAM_LPA_Z0, 0.0, 0.0, reference_beta_gamma]]
        )
        reference_output = self._run_model(self._tensor(reference_input)).detach().cpu().numpy()
        reference_beta_gamma_final = float(reference_output[0, 5])

        data = fixed_s_to_fixed_t(beam, reference_beta_gamma)
        data[:, 2] += EBEAM_LPA_Z0

        model_input = self._tensor(data)
        model_output = self._run_model(model_input).detach().cpu().numpy()

        x_s, y_s, t_s, px_abs, py_abs, pz_abs = fixed_t_to_fixed_s(
            model_output, reference_beta_gamma_final, EBEAM_LPA_Z0 + self.length
        )
        beam.set_xs(x_s)
        beam.set_ys(y_s)
        beam.set_zs(t_s)
        beam.set_uxs(px_abs * SI.c)
        beam.set_uys(py_abs * SI.c)
        beam.set_uzs(pz_abs * SI.c)
        beam.reference_beta_gamma = reference_beta_gamma_final

        return super().track(beam, savedepth, runnable, verbose)

    def get_length(self) -> float:
        return self.length


def beam_row(label: str, beam) -> tuple[str, float, float, float, float, float, float]:
    return (
        label,
        beam.location,
        beam.energy() / 1.0e9,
        beam.rel_energy_spread(),
        beam.beam_size_x() / 1.0e-6,
        beam.beam_size_y() / 1.0e-6,
        beam.bunch_length() / 1.0e-6,
    )


def safe_beam_name(index: int, label: str) -> str:
    safe_label = "".join(char.lower() if char.isalnum() else "_" for char in label)
    safe_label = "_".join(part for part in safe_label.split("_") if part)
    return f"beam_{index:03d}_{safe_label}"


class H5SnapshotRecorder:
    def __init__(self, filename: Path):
        self.filename = filename
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.series = io.Series(str(filename), io.Access.create)
        self.series.author = "ABEL (the Adaptable Beginning-to-End Linac simulation framework)"
        self.index = 0
        self.names: list[str] = []

    def record(self, label: str, beam) -> None:
        beam_name = safe_beam_name(self.index, label)
        particles = self.series.iterations[0].particles[beam_name]

        reference_beta_gamma = getattr(beam, "reference_beta_gamma", REF_U)
        rel_px = (beam.uxs() / SI.c) / reference_beta_gamma
        rel_py = (beam.uys() / SI.c) / reference_beta_gamma
        px_abs = beam.uxs() / SI.c
        py_abs = beam.uys() / SI.c
        pz_abs = beam.uzs() / SI.c
        gamma_abs = np.sqrt(1.0 + px_abs**2 + py_abs**2 + pz_abs**2)
        reference_gamma = np.sqrt(1.0 + reference_beta_gamma**2)
        rel_pt = (reference_gamma - gamma_abs) / reference_beta_gamma
        n_particles = len(beam)

        def write_component(record_name, component_name, values):
            values = np.asarray(values)
            dataset = io.Dataset(values.dtype, extent=values.shape)
            particles[record_name][component_name].reset_dataset(dataset)
            particles[record_name][component_name].store_chunk(values)

        scalar = io.Record_Component.SCALAR
        write_component("position", "x", beam.xs())
        write_component("position", "y", beam.ys())
        write_component("position", "t", beam.zs())
        write_component("positionOffset", "x", np.zeros(n_particles))
        write_component("positionOffset", "y", np.zeros(n_particles))
        write_component("positionOffset", "t", np.zeros(n_particles))
        write_component("momentum", "x", rel_px)
        write_component("momentum", "y", rel_py)
        write_component("momentum", "t", rel_pt)
        write_component("weighting", scalar, np.full(n_particles, abs(BUNCH_CHARGE_C) / SI.e / n_particles))
        write_component("id", scalar, np.arange(n_particles, dtype=np.uint64))
        write_component("qm", scalar, np.full(n_particles, -SI.e / SI.m_e))
        write_component("spin", "x", beam.spxs())
        write_component("spin", "y", beam.spys())
        write_component("spin", "z", beam.spzs())

        particles["position"].unit_dimension = {io.Unit_Dimension.L: 1}
        particles["positionOffset"].unit_dimension = {io.Unit_Dimension.L: 1}
        particles["momentum"].unit_dimension = {}
        particles["weighting"].unit_dimension = {}
        particles["id"].unit_dimension = {}
        particles["qm"].unit_dimension = {
            io.Unit_Dimension.I: 1,
            io.Unit_Dimension.T: 1,
            io.Unit_Dimension.M: -1,
        }
        particles["spin"].unit_dimension = {}

        particles.set_attribute("abel_snapshot_index", self.index)
        particles.set_attribute("abel_snapshot_label", label)
        particles.set_attribute("abel_location_m", beam.location)
        particles.set_attribute("abel_trackable_number", beam.trackable_number)
        particles.set_attribute("abel_reference_beta_gamma", reference_beta_gamma)
        particles.set_attribute("abel_coordinate_convention", "impactx_fixed_s_relative")
        self.series.flush()

        self.names.append(beam_name)
        self.index += 1

    def close(self) -> None:
        self.series.flush()
        self.series.close()


def print_summary(rows: list[tuple[str, float, float, float, float, float, float]]) -> None:
    print()
    print("ABEL surrogate tracking summary")
    print("label          s [m]     E [GeV]    rel dE      sx [um]    sy [um]    sz [um]")
    print("------------  --------  ---------  ---------   --------   --------   --------")
    for label, location, energy, rel_spread, sx, sy, sz in rows:
        print(
            f"{label:<12}  {location:8.4f}  {energy:9.4f}  "
            f"{rel_spread:9.3e}   {sx:8.3f}   {sy:8.3f}   {sz:8.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    if device is None:
        print("Torch device: CPU")
    else:
        print(f"Torch device: {device}")

    ensure_models(args.models_dir, args.n_stages, download=not args.no_download)
    models = [
        surrogate_model(args.models_dir / f"beam_stage_{stage_i}_model.pt", device=device)
        for stage_i in range(args.n_stages)
    ]

    beam = make_initial_beam(args.num_particles)
    recorder = H5SnapshotRecorder(args.save_h5) if args.save_h5 is not None else None
    lenses: list[RetunedConstFLens] = []

    try:
        rows = [beam_row("source", beam)]
        if recorder is not None:
            recorder.record("source", beam)

        for stage_i, model in enumerate(models):
            stage_label = f"stage {stage_i + 1}"
            stage = LPASurrogateStage(
                stage_i=stage_i,
                model=model,
                device=device,
                batch_size=args.batch_size,
                stage_start=L_STAGE_PERIOD * stage_i,
                name=f"LPA surrogate {stage_i}",
            )
            beam = stage.track(beam, savedepth=-1)
            rows.append(beam_row(stage_label, beam))
            if recorder is not None:
                recorder.record(stage_label, beam)

            if stage_i != args.n_stages - 1:
                drift_1_label = f"drift {stage_i + 1}a"
                beam = ABELDrift(L_DRIFT_1, name=drift_1_label).track(beam, savedepth=-1)

                lens_label = f"lens {stage_i + 1}"
                lens = RetunedConstFLens(L_LENS, tune_axis="x", name=lens_label)
                beam = lens.track(beam, savedepth=-1)
                lenses.append(lens)

                drift_2_label = f"drift {stage_i + 1}b"
                beam = ABELDrift(L_DRIFT_2, name=drift_2_label).track(beam, savedepth=-1)
    finally:
        if recorder is not None:
            recorder.close()

    print_summary(rows)

    if lenses:
        k_values = ", ".join(f"{lens.k:.6g}" for lens in lenses)
        print(f"\nRetuned lens k values [1/m]: {k_values}")

    if recorder is not None:
        print(f"\nABEL beam snapshots written to {args.save_h5}")
        print("Snapshots are stored as particle species under /data/0/particles/.")
        print("Saved snapshots include the initial beam and each plasma-stage exit beam.")
        print("Each species has an abel_location_m attribute for its snapshot location.")
        print(f"Saved {len(recorder.names)} particle species: {', '.join(recorder.names)}")


if __name__ == "__main__":
    main()
