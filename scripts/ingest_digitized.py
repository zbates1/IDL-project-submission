"""Ingest WebPlotDigitizer CSVs into the real_cvs.npz dataset.

Naming convention for input CSVs:
    {source}_{mechanism}_{scanrate}Vs.csv

    Examples:
        hoar_2022_fig3a_E_0.1Vs.csv
        sandford_2019_fig4a_EC_0.05Vs.csv
        brown_2015_fig2_E_0.1Vs.csv

Each CSV should have two columns (no header, or header will be skipped):
    potential (V),  current (A or normalized)

The script:
    1. Reads all CSVs from an input directory
    2. Groups them by source (everything before the mechanism label)
    3. Interpolates each trace to 500 points on a common potential grid
    4. Normalizes current to [-1, 1] per sample
    5. Pads missing scan rates with zeros if < 6 scan rates
    6. Appends to existing real_cvs.npz (or creates new one)

Usage:
    python scripts/ingest_digitized.py data/real/digitized/ [--output data/real/processed/real_cvs.npz]
    python scripts/ingest_digitized.py data/real/digitized/ --dry-run
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


N_POINTS = 500       # Points per CV trace (must match existing data)
N_SCAN_RATES = 6     # Max scan rates per sample
MECHANISMS = ["DISP1", "E", "EC", "CE", "ECE"]

# Pattern: {source}_{mechanism}_{scanrate}Vs.csv
# e.g., hoar_2022_fig3a_E_0.1Vs.csv
FILENAME_PATTERN = re.compile(
    r"^(.+?)_(" + "|".join(MECHANISMS) + r")_([0-9.]+)Vs\.csv$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    """Parse a digitized CSV filename into source, mechanism, scan_rate."""
    m = FILENAME_PATTERN.match(fname)
    if not m:
        return None
    return {
        "source": m.group(1),
        "mechanism": m.group(2).upper(),
        "scan_rate": float(m.group(3)),
    }


def load_csv(path: Path) -> tuple:
    """Load a WebPlotDigitizer CSV. Returns (potential, current) arrays."""
    try:
        data = np.loadtxt(path, delimiter=",", comments="#")
    except ValueError:
        # Try skipping header row
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=1)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2+ columns in {path}, got shape {data.shape}")

    potential = data[:, 0]
    current = data[:, 1]
    return potential, current


def interpolate_to_grid(potential, current, n_points=N_POINTS):
    """Interpolate a CV trace to a uniform potential grid.

    Handles both forward-then-reverse sweeps (triangle wave) and
    monotonic traces. Returns the full forward+reverse trace.
    """
    # Remove NaN/inf
    mask = np.isfinite(potential) & np.isfinite(current)
    potential, current = potential[mask], current[mask]

    if len(potential) < 10:
        raise ValueError(f"Too few valid points ({len(potential)})")

    # Create uniform grid spanning the potential range
    v_min, v_max = potential.min(), potential.max()
    grid = np.linspace(v_min, v_max, n_points)

    # If the trace is a full CV (forward + reverse), it's non-monotonic.
    # Split at the turning point and interpolate each half separately.
    turn_idx = np.argmax(potential)  # or argmin depending on sweep direction
    if turn_idx == 0 or turn_idx == len(potential) - 1:
        # Monotonic — just interpolate directly
        f = interp1d(potential, current, kind="linear", fill_value="extrapolate")
        return f(grid)

    # Forward sweep: start → turning point
    v_fwd = potential[: turn_idx + 1]
    i_fwd = current[: turn_idx + 1]

    # Reverse sweep: turning point → end
    v_rev = potential[turn_idx:]
    i_rev = current[turn_idx:]

    half = n_points // 2
    grid_fwd = np.linspace(v_min, v_max, half)
    grid_rev = np.linspace(v_max, v_min, n_points - half)

    # Ensure monotonic for interpolation
    if v_fwd[0] > v_fwd[-1]:
        v_fwd, i_fwd = v_fwd[::-1], i_fwd[::-1]
    if v_rev[0] < v_rev[-1]:
        v_rev, i_rev = v_rev[::-1], i_rev[::-1]

    f_fwd = interp1d(v_fwd, i_fwd, kind="linear", fill_value="extrapolate")
    f_rev = interp1d(v_rev[::-1], i_rev[::-1], kind="linear", fill_value="extrapolate")

    trace = np.concatenate([f_fwd(grid_fwd), f_rev(np.sort(grid_rev))])
    return trace


def normalize_sample(signals):
    """Normalize a (n_sr, n_points) array to [-1, 1] based on global max abs."""
    max_abs = np.abs(signals).max()
    if max_abs > 0:
        signals = signals / max_abs
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of digitized CSVs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/processed/real_cvs.npz"),
        help="Output npz path (default: data/real/processed/real_cvs.npz)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without saving")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    # Parse all CSVs
    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        sys.exit(1)

    # Group by source
    groups = defaultdict(list)
    skipped = []
    for f in csv_files:
        parsed = parse_filename(f.name)
        if parsed is None:
            skipped.append(f.name)
            continue
        parsed["path"] = f
        groups[parsed["source"]].append(parsed)

    if skipped:
        print(f"Skipped {len(skipped)} files (naming didn't match pattern):")
        for s in skipped:
            print(f"  {s}")
        print()

    print(f"Found {len(groups)} samples from {len(csv_files) - len(skipped)} CSVs:\n")

    new_signals = []
    new_labels = []
    new_scan_rates = []
    new_sources = []

    for source, entries in sorted(groups.items()):
        mechanism = entries[0]["mechanism"]
        # Verify all entries for this source have the same mechanism
        if not all(e["mechanism"] == mechanism for e in entries):
            print(f"  WARNING: {source} has mixed mechanisms, skipping")
            continue

        entries.sort(key=lambda e: e["scan_rate"])
        n_sr = len(entries)
        print(f"  {source}: {mechanism}, {n_sr} scan rate(s) "
              f"[{', '.join(f'{e[\"scan_rate\"]}' for e in entries)}]")

        sample_signals = np.zeros((N_SCAN_RATES, N_POINTS), dtype=np.float32)
        sample_sr = np.zeros(N_SCAN_RATES, dtype=np.float32)

        for i, entry in enumerate(entries[:N_SCAN_RATES]):
            try:
                potential, current = load_csv(entry["path"])
                trace = interpolate_to_grid(potential, current)
                sample_signals[i] = trace.astype(np.float32)
                sample_sr[i] = entry["scan_rate"]
            except Exception as e:
                print(f"    ERROR loading {entry['path'].name}: {e}")
                continue

        sample_signals = normalize_sample(sample_signals)
        new_signals.append(sample_signals)
        new_labels.append(mechanism)
        new_scan_rates.append(sample_sr)
        new_sources.append(source)

    if not new_signals:
        print("\nNo valid samples to add.")
        sys.exit(0)

    new_signals = np.array(new_signals)
    new_labels = np.array(new_labels)
    new_scan_rates = np.array(new_scan_rates)
    new_sources = np.array(new_sources)

    print(f"\nNew samples: {len(new_signals)}")
    from collections import Counter
    print(f"Per-class:   {dict(Counter(new_labels.tolist()))}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Load existing data and append
    output = args.output
    if output.exists():
        existing = np.load(output, allow_pickle=True)
        old_signals = existing["signals"]
        old_labels = existing["labels"]
        old_scan_rates = existing["scan_rates"]
        old_sources = existing["sources"] if "sources" in existing else np.array(["unknown"] * len(old_labels))
        old_potentials = existing["potentials"]

        # Check for duplicate sources
        existing_sources = set(old_sources.tolist())
        dup_mask = np.array([s not in existing_sources for s in new_sources])
        if not dup_mask.all():
            n_dup = (~dup_mask).sum()
            print(f"\nSkipping {n_dup} duplicate source(s) already in dataset")
            new_signals = new_signals[dup_mask]
            new_labels = new_labels[dup_mask]
            new_scan_rates = new_scan_rates[dup_mask]
            new_sources = new_sources[dup_mask]

        if len(new_signals) == 0:
            print("All samples already exist. Nothing to add.")
            return

        signals = np.concatenate([old_signals, new_signals], axis=0)
        labels = np.concatenate([old_labels, new_labels])
        scan_rates = np.concatenate([old_scan_rates, new_scan_rates], axis=0)
        sources = np.concatenate([old_sources, new_sources])
        potentials = old_potentials

        print(f"\nAppending to existing dataset ({len(old_labels)} -> {len(labels)} samples)")
    else:
        signals = new_signals
        labels = new_labels
        scan_rates = new_scan_rates
        sources = new_sources
        potentials = np.linspace(-1, 1, N_POINTS)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nCreating new dataset at {output}")

    np.savez(
        output,
        signals=signals,
        labels=labels,
        scan_rates=scan_rates,
        sources=sources,
        potentials=potentials,
    )

    print(f"Saved: {output}")
    print(f"Total: {len(labels)} samples")
    print(f"Per-class: {dict(Counter(labels.tolist()))}")


if __name__ == "__main__":
    main()
