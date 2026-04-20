"""Post-processing and standardization pipeline for real-world CV data.

Loads digitized CV traces (JSON from figure extraction) and Zenodo instrument
data, standardizes to a uniform 500-point potential grid, groups traces by
chemical system (for multi-scan-rate input), and produces a consolidated
``real_cvs.npz`` file with shape ``(N_systems, n_rates, 500)`` matching the
multi-rate format from :func:`cv_simulator.generate_dataset`.

No external API calls — all digitization happens upstream (Claude Code session
extracts data from figure images). This module handles post-extraction
processing only.

Usage::

    from src.data.cv_digitizer import CVDigitizer

    digitizer = CVDigitizer(signal_length=500, manifest_path="configs/real_data_manifest.json")
    output_path = digitizer.process_all_sources(raw_root="data/real/raw/", output_dir="data/real/processed/")
    # Produces data/real/processed/real_cvs.npz with shape (N_systems, 6, 500)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CVDigitizer:
    """Standardize and consolidate real CV data from multiple sources.

    Args:
        signal_length: Target number of data points per CV (default 500).
        manifest_path: Path to ``real_data_manifest.json`` for provenance.
    """

    def __init__(
        self,
        signal_length: int = 500,
        manifest_path: Optional[str] = None,
    ) -> None:
        self.signal_length = signal_length
        self.manifest: Optional[Dict[str, Any]] = None
        if manifest_path is not None:
            self.manifest = self._load_manifest(manifest_path)

    @staticmethod
    def _load_manifest(path: str) -> Dict[str, Any]:
        """Load the real data provenance manifest."""
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_digitized_json(json_path: str) -> Dict[str, Any]:
        """Load a per-trace JSON file produced by the digitization session.

        Expected JSON schema::

            {
                "potential": [float, ...],
                "current": [float, ...],
                "mechanism": "E",
                "source": "hoar_2022_fig3a",
                "scan_rate": 0.1,
                "notes": "optional"
            }

        Args:
            json_path: Path to a single-trace JSON file.

        Returns:
            Dict with ``potential``, ``current`` (as numpy arrays),
            ``mechanism``, ``source``, and optional metadata.
        """
        with open(json_path) as f:
            data = json.load(f)

        current = np.array(data["current"], dtype=np.float64)

        # Standardize to IUPAC convention (anodic positive).
        # US convention (cathodic positive) requires negation.
        convention = data.get("current_convention", "iupac")
        if convention == "us":
            current = -current

        return {
            "potential": np.array(data["potential"], dtype=np.float64),
            "current": current,
            "mechanism": data["mechanism"],
            "source": data.get("source", Path(json_path).stem),
            "scan_rate": data.get("scan_rate"),
            "notes": data.get("notes", ""),
        }

    @staticmethod
    def load_zenodo_csv(csv_path: str) -> Dict[str, Any]:
        """Load a CSV file from the Zenodo dataset (Sheng et al. 2024).

        Expects columns: ``potential`` (V), ``current`` (A), plus metadata
        columns ``mechanism`` and ``scan_rate``.  Adapts to the actual Zenodo
        format once inspected.

        Args:
            csv_path: Path to a single Zenodo CSV file.

        Returns:
            Dict with ``potential``, ``current`` arrays, ``mechanism``,
            ``source``, and ``scan_rate``.
        """
        import csv

        rows: List[Dict[str, str]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"Empty CSV file: {csv_path}")

        # Try common column names
        potential_col = None
        current_col = None
        for col in rows[0]:
            col_lower = col.lower().strip()
            if col_lower in ("potential", "e", "voltage", "potential_v", "e_v"):
                potential_col = col
            elif col_lower in ("current", "i", "current_a", "i_a"):
                current_col = col

        if potential_col is None or current_col is None:
            # Fall back to first two numeric columns
            cols = list(rows[0].keys())
            if len(cols) >= 2:
                potential_col, current_col = cols[0], cols[1]
            else:
                raise ValueError(
                    f"Cannot identify potential/current columns in {csv_path}. "
                    f"Found columns: {list(rows[0].keys())}"
                )

        potential = np.array([float(r[potential_col]) for r in rows], dtype=np.float64)
        current = np.array([float(r[current_col]) for r in rows], dtype=np.float64)

        # Extract mechanism from filename using word-boundary matching
        # to avoid false positives (e.g. "E" matching "ELECTRODE").
        # Check longer names first (DISP1 before E) for correct precedence.
        stem = Path(csv_path).stem.upper()
        mechanism = "unknown"
        for mech in ["DISP1", "ECE", "EC", "CE", "E"]:
            if re.search(r'(?:^|[_\-\s])' + mech + r'(?:$|[_\-\s])', stem):
                mechanism = mech
                break

        return {
            "potential": potential,
            "current": current,
            "mechanism": mechanism,
            "source": f"zenodo_sheng_2024/{Path(csv_path).stem}",
            "scan_rate": None,
        }

    # ------------------------------------------------------------------ #
    # Signal processing
    # ------------------------------------------------------------------ #

    def standardize_signal(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        target_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate CV trace to a uniform potential grid.

        Aligns sweep direction to the simulator convention (start at most
        positive potential, sweep negative, return) before interpolating to
        a uniform-length parametric grid.

        Args:
            potential: Raw potential array (variable length).
            current: Raw current array (same length as potential).
            target_length: Number of output points (default: ``self.signal_length``).

        Returns:
            Tuple of ``(standardized_potential, standardized_current)``
            each of shape ``(target_length,)``.
        """
        if target_length is None:
            target_length = self.signal_length

        if len(potential) != len(current):
            raise ValueError(
                f"Potential ({len(potential)}) and current ({len(current)}) "
                "arrays must have the same length"
            )

        if len(potential) < 2:
            raise ValueError("Need at least 2 data points to interpolate")

        # Align to simulator convention: start at maximum potential.
        # CVs are closed loops (start ≈ end potential), so np.roll
        # wraps cleanly.  This converts IUPAC (neg→pos→neg) traces to
        # the ElectroKitty convention (pos→neg→pos).
        max_idx = int(np.argmax(potential))
        if max_idx > 0:
            # Validate loop closure before rolling
            gap = abs(potential[0] - potential[-1])
            if gap > 0.05:  # >50 mV gap
                logger.warning(
                    "CV trace has %.0f mV endpoint gap — roll may "
                    "introduce discontinuity", gap * 1000,
                )
            potential = np.roll(potential, -max_idx)
            current = np.roll(current, -max_idx)

        # Parametric interpolation: resample to target_length points
        # along the trace path (preserving the loop shape).
        t_original = np.linspace(0, 1, len(potential))
        t_target = np.linspace(0, 1, target_length)

        std_potential = np.interp(t_target, t_original, potential)
        std_current = np.interp(t_target, t_original, current)

        return std_potential, std_current

    @staticmethod
    def normalize_current(
        current: np.ndarray,
        method: str = "peak_to_peak",
    ) -> np.ndarray:
        """Normalize current signal.

        Args:
            current: 1-D current array.
            method: Normalization method. Options:

                - ``"peak_to_peak"``: Scale to [0, 1] based on min/max.
                - ``"zscore"``: Zero mean, unit variance.

        Returns:
            Normalized current array (same shape).
        """
        if method == "peak_to_peak":
            cmin, cmax = current.min(), current.max()
            span = cmax - cmin
            if span < 1e-30:
                logger.warning("Near-zero current range; returning zeros")
                return np.zeros_like(current)
            return (current - cmin) / span
        elif method == "zscore":
            std = current.std()
            if std < 1e-30:
                return np.zeros_like(current)
            return (current - current.mean()) / std
        else:
            raise ValueError(f"Unknown normalization method: {method!r}")

    # ------------------------------------------------------------------ #
    # Multi-rate grouping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_system_id(source: str) -> str:
        """Infer a chemical-system identifier by stripping scan-rate suffixes.

        Examples::

            "zenodo_sheng_2024_fig2a_10mM_0.16Vs" → "zenodo_fig2a_EC_10mM"
            "sandford_2019_fig5b_0.01Vs"          → "sandford_2019_fig5b"
            "hoar_2022_fig1a_slow"                → "hoar_2022_fig1a"

        Args:
            source: Raw source string from trace metadata.

        Returns:
            System identifier for grouping.
        """
        s = source
        # Strip common scan-rate suffixes
        s = re.sub(r'_?\d+\.?\d*\s*[Vv]/?\s*[Ss]$', '', s)   # _0.16Vs, _0.01V/s
        s = re.sub(r'_?\d+\.?\d*Vs$', '', s)                   # _0.16Vs
        s = re.sub(r'_(slow|fast|medium)$', '', s, flags=re.IGNORECASE)
        # Strip trailing whitespace/underscores
        s = s.rstrip('_').strip()
        return s

    @staticmethod
    def group_by_system(
        traces: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group loaded traces by chemical system ID.

        Args:
            traces: List of trace dicts (each with ``source``, ``scan_rate``,
                ``mechanism``, ``current``, ``potential``).

        Returns:
            Dict mapping system_id → list of traces, sorted by scan_rate
            (ascending) within each group.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trace in traces:
            system_id = trace.get(
                "system_id",
                CVDigitizer._infer_system_id(trace.get("source", "")),
            )
            trace["system_id"] = system_id
            groups[system_id].append(trace)

        # Assign placeholder scan rates for traces with None
        # (e.g. Hoar 2022 digitized traces labeled slow/fast)
        for sys_id in groups:
            group = groups[sys_id]
            null_count = sum(1 for t in group if t.get("scan_rate") is None)
            if null_count > 0 and null_count == len(group):
                # All null — assign evenly spaced placeholders [0.1 .. 1.0]
                placeholders = np.linspace(0.1, 1.0, len(group)).tolist()
                for t, rate in zip(group, placeholders):
                    t["scan_rate"] = rate

        # Sort each group by scan_rate ascending
        for sys_id in groups:
            groups[sys_id].sort(
                key=lambda t: t.get("scan_rate") or 0.0,
            )

        return dict(groups)

    def standardize_multi_rate(
        self,
        traces: List[Dict[str, Any]],
        n_rates: int = 6,
    ) -> Dict[str, Any]:
        """Standardize a group of traces for one system into multi-rate format.

        Each trace is standardized to ``signal_length`` points, then
        stacked into ``(n_rates, signal_length)``.  If fewer traces
        than ``n_rates``, **replicates the last measured trace** to fill
        remaining channels (avoids zero-padding artifacts that create
        an artificial domain signal).  If more, subsamples evenly.
        Applies cross-rate normalization.

        Args:
            traces: List of trace dicts for one chemical system.
            n_rates: Target number of scan rates (default 6).

        Returns:
            Dict with ``"signals"`` ``(n_rates, L)``, ``"scan_rates"``
            ``(n_rates,)``, ``"mechanism"`` str, ``"source"`` str.
        """
        # Standardize each trace to uniform length
        standardized: List[np.ndarray] = []
        rates: List[float] = []
        for trace in traces:
            _, std_cur = self.standardize_signal(
                trace["potential"], trace["current"],
            )
            standardized.append(std_cur)
            rates.append(trace.get("scan_rate") or 0.0)

        n_available = len(standardized)

        if n_available >= n_rates:
            # Subsample evenly
            indices = np.linspace(0, n_available - 1, n_rates, dtype=int)
            selected = [standardized[i] for i in indices]
            selected_rates = [rates[i] for i in indices]
        else:
            # Use all available, replicate last trace to fill remaining
            # channels. Replication avoids zero-padding artifacts where
            # the model could trivially distinguish sim (all channels
            # have signal) from real (mostly zeros).
            selected = list(standardized)
            selected_rates = list(rates)
            while len(selected) < n_rates:
                selected.append(standardized[-1].copy())
                selected_rates.append(rates[-1])

        signals = np.stack(selected, axis=0)  # (n_rates, L)

        # Cross-rate normalization
        abs_max = np.max(np.abs(signals))
        if abs_max > 1e-30:
            signals = signals / abs_max

        mechanism = traces[0].get("mechanism", "unknown")
        source = traces[0].get("system_id", traces[0].get("source", "unknown"))

        return {
            "signals": signals,
            "scan_rates": np.array(selected_rates, dtype=np.float64),
            "mechanism": mechanism,
            "source": source,
        }

    # ------------------------------------------------------------------ #
    # Batch processing
    # ------------------------------------------------------------------ #

    def process_all_sources(
        self,
        raw_root: str,
        output_dir: str,
        n_rates: int = 6,
    ) -> Path:
        """Load all raw real CV data, group by system, and save multi-rate .npz.

        Scans ``raw_root/digitized/`` for JSON files and
        ``raw_root/zenodo_sheng_2024/`` for CSV files.  Groups traces by
        chemical system and outputs ``(N_systems, n_rates, 500)`` format.

        Args:
            raw_root: Root directory for raw data (e.g., ``data/real/raw/``).
            output_dir: Directory for processed output (e.g., ``data/real/processed/``).
            n_rates: Target number of scan rates per system (default 6).

        Returns:
            Path to the saved ``real_cvs.npz`` file.
        """
        raw_path = Path(raw_root)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_traces: List[Dict[str, Any]] = []
        potentials: Optional[np.ndarray] = None

        # --- Load digitized JSONs ---
        digitized_dir = raw_path / "digitized"
        if digitized_dir.exists():
            json_files = sorted(digitized_dir.glob("*.json"))
            logger.info("Found %d digitized JSON files", len(json_files))

            for jf in json_files:
                try:
                    trace = self.load_digitized_json(str(jf))
                except Exception as e:
                    logger.warning("Skipping %s: %s", jf.name, e)
                    continue
                all_traces.append(trace)
                if potentials is None:
                    std_pot, _ = self.standardize_signal(
                        trace["potential"], trace["current"],
                    )
                    potentials = std_pot

        # --- Load Zenodo CSVs ---
        zenodo_dir = raw_path / "zenodo_sheng_2024"
        if zenodo_dir.exists():
            csv_files = sorted(zenodo_dir.glob("*.csv"))
            logger.info("Found %d Zenodo CSV files", len(csv_files))

            for cf in csv_files:
                try:
                    trace = self.load_zenodo_csv(str(cf))
                except Exception as e:
                    logger.warning("Skipping %s: %s", cf.name, e)
                    continue
                all_traces.append(trace)
                if potentials is None:
                    std_pot, _ = self.standardize_signal(
                        trace["potential"], trace["current"],
                    )
                    potentials = std_pot

        if not all_traces:
            logger.warning("No real CV data found in %s", raw_root)
            save_path = out_path / "real_cvs.npz"
            np.savez(
                save_path,
                signals=np.empty((0, n_rates, self.signal_length)),
                scan_rates=np.empty((0, n_rates)),
                potentials=np.linspace(-0.5, 0.5, self.signal_length),
                labels=np.array([], dtype="U10"),
                sources=np.array([], dtype="U100"),
            )
            return save_path

        # --- Group by chemical system and build multi-rate arrays ---
        groups = self.group_by_system(all_traces)
        logger.info("Grouped %d traces into %d systems", len(all_traces), len(groups))

        all_signals: List[np.ndarray] = []
        all_scan_rates: List[np.ndarray] = []
        all_labels: List[str] = []
        all_sources: List[str] = []

        for sys_id, traces in sorted(groups.items()):
            result = self.standardize_multi_rate(traces, n_rates=n_rates)
            all_signals.append(result["signals"])
            all_scan_rates.append(result["scan_rates"])
            all_labels.append(result["mechanism"])
            all_sources.append(result["source"])
            logger.info(
                "  %s: %d traces → (%d, %d) | mechanism=%s",
                sys_id, len(traces),
                result["signals"].shape[0], result["signals"].shape[1],
                result["mechanism"],
            )

        signals_arr = np.stack(all_signals, axis=0)  # (N_systems, n_rates, L)
        scan_rates_arr = np.stack(all_scan_rates, axis=0)  # (N_systems, n_rates)
        labels_arr = np.array(all_labels)
        sources_arr = np.array(all_sources)

        save_path = out_path / "real_cvs.npz"
        np.savez(
            save_path,
            signals=signals_arr,
            scan_rates=scan_rates_arr,
            potentials=potentials,
            labels=labels_arr,
            sources=sources_arr,
        )

        unique_labels, counts = np.unique(labels_arr, return_counts=True)
        logger.info(
            "Saved %d systems to %s  |  shape=%s  |  Mechanisms: %s",
            len(labels_arr), save_path, signals_arr.shape,
            dict(zip(unique_labels, counts)),
        )

        return save_path

    def dry_run(self, raw_root: str) -> Dict[str, Any]:
        """Validate manifest and count available data without processing.

        Args:
            raw_root: Root directory for raw data.

        Returns:
            Dict with counts per source and mechanism coverage summary.
        """
        raw_path = Path(raw_root)
        summary: Dict[str, Any] = {"digitized": 0, "zenodo": 0, "mechanisms": {}}

        digitized_dir = raw_path / "digitized"
        if digitized_dir.exists():
            json_files = list(digitized_dir.glob("*.json"))
            summary["digitized"] = len(json_files)
            for jf in json_files:
                try:
                    trace = self.load_digitized_json(str(jf))
                    mech = trace["mechanism"]
                    summary["mechanisms"][mech] = summary["mechanisms"].get(mech, 0) + 1
                except Exception as e:
                    logger.warning("Invalid JSON %s: %s", jf.name, e)

        zenodo_dir = raw_path / "zenodo_sheng_2024"
        if zenodo_dir.exists():
            csv_files = list(zenodo_dir.glob("*.csv"))
            summary["zenodo"] = len(csv_files)

        summary["total"] = summary["digitized"] + summary["zenodo"]
        return summary
