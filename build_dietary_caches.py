"""Build the NHANES dietary caches from the raw .XPT files. Run ONCE per machine.

The dietary builder needs two cache files derived from NHANES dietary data:
  - density.pkl     (per-100g nutrient table, from P_DR1IFF.XPT)
  - food_index.pkl  (food code -> description search index, from P_DRXFCD.XPT)

These caches are multi-MB and are intentionally NOT committed to the repo. You
generate them once from your local NHANES files with this script.

Usage:
    venv\\Scripts\\python.exe build_dietary_caches.py ^
        --dr1iff "C:\\path\\to\\P_DR1IFF.XPT" ^
        --drxfcd "C:\\path\\to\\P_DRXFCD.XPT"

By default the caches are written to data\\nhanes\\. Override with --out-dir.
After this runs once, the dietary builder loads the caches at startup:

    from server.builders import DietaryPipeline
    pipeline = DietaryPipeline("data/nhanes/density.pkl", "data/nhanes/food_index.pkl")
"""

import argparse
import sys
from pathlib import Path

from server.builders import build_caches


def main():
    parser = argparse.ArgumentParser(description="Build NHANES dietary caches.")
    parser.add_argument("--dr1iff", required=True,
                        help="Path to NHANES P_DR1IFF.XPT (individual foods file)")
    parser.add_argument("--drxfcd", required=True,
                        help="Path to NHANES P_DRXFCD.XPT (food code dictionary)")
    parser.add_argument("--out-dir", default="data/nhanes",
                        help="Where to write density.pkl and food_index.pkl (default: data/nhanes)")
    args = parser.parse_args()

    dr1iff = Path(args.dr1iff)
    drxfcd = Path(args.drxfcd)
    out_dir = Path(args.out_dir)

    for p, label in ((dr1iff, "P_DR1IFF.XPT"), (drxfcd, "P_DRXFCD.XPT")):
        if not p.exists():
            print(f"ERROR: {label} not found at: {p}")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    density_cache = out_dir / "density.pkl"
    food_index_cache = out_dir / "food_index.pkl"

    print("Building NHANES dietary caches (one-time, ~30s)...")
    print(f"  P_DR1IFF: {dr1iff}")
    print(f"  P_DRXFCD: {drxfcd}")
    print(f"  output:   {out_dir}\\")
    print("-" * 60)

    summary = build_caches(
        dr1iff_path=dr1iff,
        drxfcd_path=drxfcd,
        density_cache_path=density_cache,
        food_index_cache_path=food_index_cache,
        verbose=True,
    )

    print("-" * 60)
    print("Done. Cache summary:")
    print(f"  food codes:          {summary['food_codes']:,}")
    print(f"  nutrient components: {summary['nutrient_components']}")
    print(f"  indexed foods:       {summary['indexed_foods']:,}")
    print(f"  density cache:       {summary['density_cache']}")
    print(f"  food index cache:    {summary['food_index_cache']}")
    print()
    print("The dietary builder can now load these caches:")
    print(f'  DietaryPipeline(r"{density_cache}", r"{food_index_cache}")')
    return 0


if __name__ == "__main__":
    sys.exit(main())
