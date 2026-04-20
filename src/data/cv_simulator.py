"""ElectroKitty-based cyclic voltammogram simulator for CMU.49.012.

Generates simulated CVs for 5 electrochemical mechanisms (E, EC, CE, ECE,
DISP1) with parametric variation across scan rates, concentrations, diffusion
coefficients, and rate constants.  Used to produce training data for a deep
learning mechanism classifier.

Backend: ElectroKitty (``pip install electrokitty``), a peer-reviewed
simulator validated against MECSim (Vodeb et al., ACS Electrochemistry 2025).

Usage::

    from src.data.cv_simulator import CVSimulator, DEFAULT_SCAN_RATES, generate_dataset

    sim = CVSimulator()
    data = sim.generate("E", n_samples=100, scan_rates=DEFAULT_SCAN_RATES)
    # data["signals"].shape == (100, 6, 500)

    generate_dataset(config=None, output_dir="data/simulated", n_per_class=3000)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from electrokitty import ElectroKitty

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Module-level worker state for multiprocessing
# ------------------------------------------------------------------ #

_worker_sim: Optional["CVSimulator"] = None


def _init_worker(signal_length: int) -> None:
    """Initialize a per-process :class:`CVSimulator` instance."""
    global _worker_sim
    _worker_sim = CVSimulator(signal_length=signal_length)


def _generate_one_sample(args: Tuple) -> Optional[Dict[str, Any]]:
    """Worker function: simulate one multi-rate sample.

    Returns ``None`` on failure so the caller can filter and retry.
    """
    mechanism, params, scan_rates = args
    assert _worker_sim is not None
    try:
        result = _worker_sim.simulate_multi_rate(mechanism, params, scan_rates)
        if not np.all(np.isfinite(result["signals"])):
            return None
        return result
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Mechanism definitions
# ------------------------------------------------------------------ #

MECHANISMS: Dict[str, str] = {
    "E":     "E(1): a = b",
    "EC":    "E(1): a = b \n C: b = c",
    "CE":    "C: c = a \n E(1): a = b",
    "ECE":   "E(1): a = b \n C: b = c \n E(1): c = d",
    "DISP1": "E(1): a = b \n C: b + b = a + c",
}

# Number of dissolved species per mechanism (no adsorbed species used).
_N_SPECIES: Dict[str, int] = {
    "E": 2, "EC": 3, "CE": 3, "ECE": 4, "DISP1": 3,
}

# Default parameter ranges for random sampling.
DEFAULT_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "scan_rate": (0.01, 2.0),        # V/s  (covers Zenodo max ~2.0 V/s)
    "concentration": (0.1, 10.0),    # mol/m^3
    "k0": (0.01, 100.0),             # m/s  (log-uniform)
    "E0": (-0.3, 0.3),               # V
    "alpha": (0.3, 0.7),             # dimensionless
    "D": (1e-10, 1e-8),              # m^2/s (log-uniform)
    "kf_chem": (0.1, 1000.0),        # 1/s for C steps (log-uniform)
}

# Per-mechanism parameter overrides applied on top of DEFAULT_PARAM_RANGES.
MECHANISM_PARAM_OVERRIDES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "E": {"k0": (1e-6, 0.1)},  # quasi-reversible range for shape diversity
}

# Default scan rates for multi-rate input (V/s, geometric progression).
# Matches Zenodo fig2a series and Hoar et al. (2022) 6-rate approach.
DEFAULT_SCAN_RATES: List[float] = [0.1, 0.16, 0.25, 0.4, 0.63, 1.0]

# Fixed cell constants: [temperature_K, R_uncomp_Ohm, C_dl_F/m2, area_m2]
_CELL_CONST = [293.0, 0.0, 0.0, 0.283e-4]

# Spatial discretisation: [dx_fraction, n_points, viscosity_m2/s, rotation_Hz]
_SPATIAL_INFO = [0.001, 20, 1e-5, 0.0]


# ------------------------------------------------------------------ #
# CVSimulator
# ------------------------------------------------------------------ #


class CVSimulator:
    """ElectroKitty-based cyclic voltammogram simulator.

    Args:
        signal_length: Number of data points per CV (default 500).
        e_window: Half-width of the potential window around E0 (V).
        cell_const: Fixed cell constants [T, R, C_dl, area].
        spatial_info: Spatial discretisation parameters.
    """

    def __init__(
        self,
        signal_length: int = 500,
        e_window: float = 0.5,
        cell_const: Optional[List[float]] = None,
        spatial_info: Optional[List[float]] = None,
    ) -> None:
        self.signal_length = signal_length
        self.e_window = e_window
        self.cell_const = cell_const or list(_CELL_CONST)
        self.spatial_info = spatial_info or list(_SPATIAL_INFO)

    # ------------------------------------------------------------ #
    # Single-CV simulation
    # ------------------------------------------------------------ #

    def simulate_one(
        self,
        mechanism: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate a single CV for the given mechanism and parameters.

        Args:
            mechanism: One of ``"E"``, ``"EC"``, ``"CE"``, ``"ECE"``,
                ``"DISP1"``.
            params: Dict with keys ``scan_rate``, ``concentration``,
                ``k0``, ``E0``, ``alpha``, ``D``, and optionally
                ``kf_chem`` (for mechanisms with a C step).

        Returns:
            Dict with ``"potential"`` (1-D array), ``"current"`` (1-D array),
            ``"mechanism"`` label, and ``"params"`` copy.
        """
        if mechanism not in MECHANISMS:
            raise ValueError(
                f"Unknown mechanism {mechanism!r}. "
                f"Choose from {list(MECHANISMS)}"
            )

        mech_str = MECHANISMS[mechanism]
        n_species = _N_SPECIES[mechanism]

        scan_rate = params["scan_rate"]
        conc = params["concentration"]
        k0 = params["k0"]
        E0 = params["E0"]
        alpha = params["alpha"]
        D_val = params["D"]
        kf_chem = params.get("kf_chem", 10.0)

        # --- Build ElectroKitty parameter lists ---
        kin = _build_kin(mechanism, alpha, k0, E0, kf_chem)
        D_list = [D_val] * n_species
        init_conc = [conc] + [0.0] * (n_species - 1)
        # CE: ElectroKitty parses "C: c = a \n E(1): a = b" producing
        # species order [c, a, b]. Precursor species c is at index 0.
        # Note: CE still has a residual ElectroKitty boundary-condition bug
        # (kf saturates at ~1), limiting CE shape diversity.
        if mechanism == "CE":
            init_conc = [conc, 0.0, 0.0]
        species_info: List[Any] = [[], init_conc]

        # Potential window centred on E0
        Ei = E0 + self.e_window
        Ef = E0 - self.e_window

        sim = ElectroKitty(mech_str)
        sim.V_potential(Ei, Ef, scan_rate, 0, 0, self.signal_length)
        sim.create_simulation(
            kin, self.cell_const, D_list, [],
            self.spatial_info, species_info, kinetic_model="BV",
        )
        sim.simulate()

        potential = np.array(sim.E_generated, dtype=np.float64)
        current = np.array(sim.current, dtype=np.float64)

        return {
            "potential": potential,
            "current": current,
            "mechanism": mechanism,
            "params": dict(params),
        }

    # ------------------------------------------------------------ #
    # Multi-rate simulation
    # ------------------------------------------------------------ #

    def simulate_multi_rate(
        self,
        mechanism: str,
        params: Dict[str, Any],
        scan_rates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Simulate one chemical system at multiple scan rates.

        Each "sample" is a single parameter set measured at *n* different
        scan rates, producing a 2-D current matrix.

        Args:
            mechanism: Mechanism label.
            params: Shared parameter dict (``scan_rate`` key is overridden
                per rate).
            scan_rates: List of scan rates (V/s).  Defaults to
                :data:`DEFAULT_SCAN_RATES`.

        Returns:
            Dict with ``"signals"`` ``(n_rates, L)`` array,
            ``"scan_rates"`` ``(n_rates,)`` array, ``"mechanism"`` str,
            and ``"params"`` dict.
        """
        if scan_rates is None:
            scan_rates = list(DEFAULT_SCAN_RATES)

        currents: List[np.ndarray] = []
        for rate in scan_rates:
            p = dict(params)
            p["scan_rate"] = rate
            result = self.simulate_one(mechanism, p)
            currents.append(result["current"])

        signals = np.stack(currents, axis=0)  # (n_rates, L)

        # Cross-rate normalization: divide by global absolute max
        abs_max = np.max(np.abs(signals))
        if abs_max > 1e-30:
            signals = signals / abs_max

        return {
            "signals": signals,
            "scan_rates": np.array(scan_rates, dtype=np.float64),
            "mechanism": mechanism,
            "params": dict(params),
        }

    # ------------------------------------------------------------ #
    # Batch generation
    # ------------------------------------------------------------ #

    def generate(
        self,
        mechanism: str,
        n_samples: int = 100,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        scan_rates: Optional[List[float]] = None,
        n_workers: int = 1,
        retry_factor: float = 1.5,
    ) -> Dict[str, Any]:
        """Generate *n_samples* multi-rate CV sets for a single mechanism.

        Parameters are sampled uniformly (or log-uniformly for ``k0``,
        ``D``, ``kf_chem``) within the given ranges.  Each sample is
        simulated at all scan rates, producing a 3-D output.

        Args:
            mechanism: Mechanism label (``"E"``, ``"EC"``, etc.).
            n_samples: Number of multi-rate samples to generate.
            param_ranges: Override default parameter ranges.
            seed: Random seed for reproducibility.
            scan_rates: List of scan rates (V/s).  Defaults to
                :data:`DEFAULT_SCAN_RATES`.
            n_workers: Number of parallel workers.  ``1`` = sequential
                (default).  Values >1 use :mod:`multiprocessing.Pool`.
            retry_factor: Oversample ratio for parallel mode — extra
                params pre-sampled to compensate for failures (default
                1.5 = 50 % extra).

        Returns:
            Dict with keys:
            - ``"signals"``: ``np.ndarray`` of shape ``(N, n_rates, L)``
            - ``"scan_rates"``: ``np.ndarray`` of shape ``(N, n_rates)``
            - ``"potentials"``: ``np.ndarray`` of shape ``(L,)``
            - ``"labels"``: list of mechanism strings
            - ``"params"``: list of parameter dicts
        """
        if scan_rates is None:
            scan_rates = list(DEFAULT_SCAN_RATES)

        ranges = dict(DEFAULT_PARAM_RANGES)
        # Apply per-mechanism overrides (e.g. E gets quasi-reversible k0)
        if mechanism in MECHANISM_PARAM_OVERRIDES:
            ranges.update(MECHANISM_PARAM_OVERRIDES[mechanism])
        if param_ranges:
            ranges.update(param_ranges)

        rng = np.random.default_rng(seed)

        signals: List[np.ndarray] = []
        scan_rates_list: List[np.ndarray] = []
        params_list: List[Dict[str, Any]] = []
        potentials: Optional[np.ndarray] = None
        failed = 0
        progress_interval = max(1, n_samples // 10)

        if n_workers > 1:
            # ----- Parallel path (multiprocessing Pool) ----- #
            n_total = int(n_samples * retry_factor)
            all_params = [_sample_params(ranges, rng) for _ in range(n_total)]
            args_list = [(mechanism, p, scan_rates) for p in all_params]

            logger.info(
                "  %s: launching Pool with %d workers (%d candidate samples)",
                mechanism, n_workers, n_total,
            )
            with mp.Pool(
                n_workers,
                initializer=_init_worker,
                initargs=(self.signal_length,),
            ) as pool:
                results = pool.map(_generate_one_sample, args_list)

            # Collect successful results up to n_samples
            for r in results:
                if r is None:
                    failed += 1
                    continue
                if len(signals) >= n_samples:
                    break
                signals.append(r["signals"])
                scan_rates_list.append(r["scan_rates"])
                params_list.append(r["params"])
                if potentials is None:
                    p = dict(r["params"])
                    p["scan_rate"] = scan_rates[0]
                    pot_result = self.simulate_one(mechanism, p)
                    potentials = pot_result["potential"]
                if len(signals) % progress_interval == 0:
                    logger.info(
                        "  %s: %d/%d samples collected",
                        mechanism, len(signals), n_samples,
                    )

            if len(signals) < n_samples:
                raise RuntimeError(
                    f"{mechanism}: only {len(signals)}/{n_samples} samples "
                    f"succeeded from {n_total} candidates "
                    f"({failed} failures). Increase retry_factor or check "
                    f"parameter ranges."
                )
        else:
            # ----- Sequential path (original behaviour) ----- #
            for i in range(n_samples):
                params = _sample_params(ranges, rng)
                try:
                    result = self.simulate_multi_rate(mechanism, params, scan_rates)
                except Exception:
                    failed += 1
                    logger.warning(
                        "Simulation %d/%d failed for %s, resampling",
                        i + 1, n_samples, mechanism,
                    )
                    # Retry with new params (up to 3 extra attempts)
                    for _ in range(3):
                        params = _sample_params(ranges, rng)
                        try:
                            result = self.simulate_multi_rate(
                                mechanism, params, scan_rates,
                            )
                            break
                        except Exception:
                            failed += 1
                            continue
                    else:
                        continue  # skip this sample entirely

                sig = result["signals"]  # (n_rates, L)
                if not np.all(np.isfinite(sig)):
                    failed += 1
                    continue

                signals.append(sig)
                scan_rates_list.append(result["scan_rates"])
                params_list.append(result["params"])
                if potentials is None:
                    # Get potential grid from a single-rate call
                    p = dict(params)
                    p["scan_rate"] = scan_rates[0]
                    pot_result = self.simulate_one(mechanism, p)
                    potentials = pot_result["potential"]

                if (i + 1) % progress_interval == 0:
                    logger.info(
                        "  %s: %d/%d samples collected",
                        mechanism, len(signals), n_samples,
                    )

        if failed > 0:
            logger.info(
                "%s: %d simulations failed or produced non-finite values",
                mechanism, failed,
            )

        signals_arr = np.stack(signals, axis=0)  # (N, n_rates, L)
        scan_rates_arr = np.stack(scan_rates_list, axis=0)  # (N, n_rates)
        labels = [mechanism] * len(signals)

        return {
            "signals": signals_arr,
            "scan_rates": scan_rates_arr,
            "potentials": potentials if potentials is not None else np.array([]),
            "labels": labels,
            "params": params_list,
        }


# ------------------------------------------------------------------ #
# Batch dataset generation
# ------------------------------------------------------------------ #


def generate_dataset(
    config: Optional[Dict[str, Any]],
    output_dir: str,
    n_per_class: int = 3000,
    seed: int = 42,
    n_workers: int = 1,
) -> Path:
    """Generate a full simulated multi-rate CV dataset for all mechanisms.

    Saves the dataset as a single ``.npz`` file with arrays ``signals``
    ``(N, n_rates, 500)``, ``scan_rates`` ``(N, n_rates)``, ``potentials``
    ``(500,)``, and ``labels`` ``(N,)``.

    Supports per-mechanism **checkpointing**: if the run is interrupted,
    re-running the same command resumes from where it left off (completed
    mechanisms are loaded from checkpoint files).

    Args:
        config: Project config dict.  If ``None``, uses defaults.  Reads
            ``data.cv.signal_length``, ``data.cv.n_per_class``,
            ``data.cv.mechanisms``, ``data.cv.scan_rates``, and
            ``data.cv.generation.{n_workers, retry_factor}`` when available.
        output_dir: Directory to write the ``.npz`` file into.
        n_per_class: Number of CVs per mechanism (overridden by config).
        seed: Base random seed (offset per mechanism for independence).
        n_workers: Number of parallel workers (1 = sequential).
            Overridden by ``data.cv.generation.n_workers`` in *config*.

    Returns:
        Path to the saved ``.npz`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read config overrides
    retry_factor = 1.5
    if config is not None:
        cv_cfg = config.get("data", {}).get("cv", {})
        signal_length = cv_cfg.get("signal_length", 500)
        n_per_class = cv_cfg.get("n_per_class", n_per_class)
        mechanisms = cv_cfg.get("mechanisms", list(MECHANISMS.keys()))
        scan_rates = cv_cfg.get("scan_rates", DEFAULT_SCAN_RATES)
        gen_cfg = cv_cfg.get("generation", {})
        n_workers = gen_cfg.get("n_workers", n_workers)
        retry_factor = gen_cfg.get("retry_factor", retry_factor)
    else:
        signal_length = 500
        mechanisms = list(MECHANISMS.keys())
        scan_rates = list(DEFAULT_SCAN_RATES)

    # n_workers=0 means auto (nproc - 4, at least 1)
    if n_workers == 0:
        n_workers = max(1, mp.cpu_count() - 4)
        logger.info("Auto-detected %d workers", n_workers)

    sim = CVSimulator(signal_length=signal_length)

    all_signals: List[np.ndarray] = []
    all_scan_rates: List[np.ndarray] = []
    all_labels: List[str] = []
    all_params: List[Dict[str, Any]] = []
    potentials: Optional[np.ndarray] = None

    for idx, mech in enumerate(mechanisms):
        checkpoint_path = out / f"_checkpoint_{mech}.npz"
        params_checkpoint = out / f"_checkpoint_{mech}_params.npy"

        # Resume from checkpoint if it exists
        if checkpoint_path.exists():
            logger.info("Resuming %s from checkpoint", mech)
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            data = {
                "signals": ckpt["signals"],
                "scan_rates": ckpt["scan_rates"],
                "labels": list(ckpt["labels"]),
                "params": list(
                    np.load(params_checkpoint, allow_pickle=True)
                ),
            }
            if potentials is None:
                potentials = ckpt["potentials"]
        else:
            logger.info(
                "Generating %d multi-rate CVs (%d rates) for mechanism %s ...",
                n_per_class, len(scan_rates), mech,
            )
            data = sim.generate(
                mech,
                n_samples=n_per_class,
                seed=seed + idx,
                scan_rates=scan_rates,
                n_workers=n_workers,
                retry_factor=retry_factor,
            )
            if potentials is None:
                potentials = data["potentials"]

            # Save checkpoint
            np.savez(
                checkpoint_path,
                signals=data["signals"],
                scan_rates=data["scan_rates"],
                potentials=potentials,
                labels=np.array(data["labels"]),
            )
            np.save(params_checkpoint, data["params"], allow_pickle=True)
            logger.info("Checkpoint saved for %s", mech)

        all_signals.append(data["signals"])
        all_scan_rates.append(data["scan_rates"])
        all_labels.extend(data["labels"])
        all_params.extend(data["params"])
        logger.info(
            "  %s: %d samples ready", mech, len(data["labels"]),
        )

    signals_arr = np.concatenate(all_signals, axis=0)  # (N_total, n_rates, L)
    scan_rates_arr = np.concatenate(all_scan_rates, axis=0)  # (N_total, n_rates)
    labels_arr = np.array(all_labels)

    save_path = out / "simulated_cvs.npz"
    np.savez(
        save_path,
        signals=signals_arr,
        scan_rates=scan_rates_arr,
        potentials=potentials,
        labels=labels_arr,
    )
    # Save params separately (list of dicts → .npy with allow_pickle)
    params_path = out / "simulated_params.npy"
    np.save(params_path, all_params, allow_pickle=True)

    # Clean up checkpoints after successful save
    for mech in mechanisms:
        for suffix in [f"_checkpoint_{mech}.npz", f"_checkpoint_{mech}_params.npy"]:
            ckpt = out / suffix
            if ckpt.exists():
                ckpt.unlink()
    logger.info("Checkpoints cleaned up")

    logger.info(
        "Dataset saved to %s  |  %d total samples  |  %d mechanisms  |  %d rates",
        save_path, len(labels_arr), len(mechanisms), len(scan_rates),
    )
    return save_path


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _build_kin(
    mechanism: str, alpha: float, k0: float, E0: float, kf_chem: float,
) -> List[List[float]]:
    """Build the ElectroKitty kinetic constant list for a mechanism."""
    e_step = [alpha, k0, E0]
    c_step = [kf_chem, 0.0]  # [kf, kb] — irreversible chemical step

    if mechanism == "E":
        return [e_step]
    elif mechanism == "EC":
        return [e_step, c_step]
    elif mechanism == "CE":
        return [c_step, e_step]
    elif mechanism == "ECE":
        # Second E step at a different potential (shifted by -0.3 V)
        e_step2 = [alpha, k0, E0 - 0.3]
        return [e_step, c_step, e_step2]
    elif mechanism == "DISP1":
        return [e_step, c_step]
    else:
        raise ValueError(f"Unknown mechanism: {mechanism!r}")


def _sample_params(
    ranges: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample a random parameter set from the given ranges."""
    lo, hi = ranges["scan_rate"]
    scan_rate = rng.uniform(lo, hi)

    lo, hi = ranges["concentration"]
    concentration = rng.uniform(lo, hi)

    # Log-uniform for k0, D, kf_chem
    lo, hi = ranges["k0"]
    k0 = 10 ** rng.uniform(np.log10(lo), np.log10(hi))

    lo, hi = ranges["E0"]
    E0 = rng.uniform(lo, hi)

    lo, hi = ranges["alpha"]
    alpha = rng.uniform(lo, hi)

    lo, hi = ranges["D"]
    D = 10 ** rng.uniform(np.log10(lo), np.log10(hi))

    lo, hi = ranges["kf_chem"]
    kf_chem = 10 ** rng.uniform(np.log10(lo), np.log10(hi))

    return {
        "scan_rate": scan_rate,
        "concentration": concentration,
        "k0": k0,
        "E0": E0,
        "alpha": alpha,
        "D": D,
        "kf_chem": kf_chem,
    }
