"""Top-level training orchestrator for Agent 1.

Runs Layer 1 (NHANES -> CRP quantile regressors) and Layer 2 (Rheumatic ->
3 cognitive pillars) in sequence. Use this when setting up the agent from
scratch.

Usage::

    python -m immunosense.agents.biomarker.train \\
        --rheumatic-xlsx ./data/rheumatic/Rheumatic_and_Autoimmune_Disease_Dataset.xlsx \\
        [--nhanes-dir ./data/nhanes] \\
        [--artifact-root ./artifacts]

Produces:
    artifacts/agent1_layer1/
        crp_quantile_*.pkl  (7 files)
        layer1_metadata.pkl
    artifacts/agent1_layer2/
        lgb_model.pkl
        xgb_model.json
        encoder.pt
        centroids.npz
        scaler.pkl
        label_encoder.pkl
        layer2_metadata.pkl

Total training time: 5-10 minutes on a modern laptop (CPU only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from immunosense.agents.biomarker.layer1.train import run_layer1_training
from immunosense.agents.biomarker.layer2.train import run_layer2_training


def run_full_training(
    rheumatic_xlsx: Path,
    nhanes_dir: Path,
    artifact_root: Path,
    encoder_epochs: int = 100,
    random_state: int = 42,
    skip_layer1: bool = False,
    skip_layer2: bool = False,
    verbose: bool = True,
) -> None:
    """Train Layer 1 + Layer 2 end-to-end."""
    layer1_dir = artifact_root / "agent1_layer1"
    layer2_dir = artifact_root / "agent1_layer2"

    if verbose:
        print("=" * 70)
        print("Agent 1 full training (Layer 1 + Layer 2)")
        print("=" * 70)

    if not skip_layer1:
        run_layer1_training(
            data_dir=nhanes_dir,
            artifact_dir=layer1_dir,
            verbose=verbose,
        )
    elif verbose:
        print("\n[skipped] Layer 1 (--skip-layer1)")

    if not skip_layer2:
        run_layer2_training(
            rheumatic_xlsx=rheumatic_xlsx,
            artifact_dir=layer2_dir,
            encoder_epochs=encoder_epochs,
            random_state=random_state,
            verbose=verbose,
        )
    elif verbose:
        print("\n[skipped] Layer 2 (--skip-layer2)")

    if verbose:
        print()
        print("=" * 70)
        print("Full training complete.")
        print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Agent 1 (Biomarker) end-to-end: Layer 1 + Layer 2"
    )
    parser.add_argument(
        "--rheumatic-xlsx", type=Path, required=True,
        help="Path to the Rheumatic and Autoimmune Disease dataset .xlsx",
    )
    parser.add_argument(
        "--nhanes-dir", type=Path, default=Path("./data/nhanes"),
        help="NHANES data directory (for Layer 1)",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("./artifacts"),
        help="Root directory under which agent1_layer1/ and agent1_layer2/ are written",
    )
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-layer1", action="store_true")
    parser.add_argument("--skip-layer2", action="store_true")
    args = parser.parse_args()

    run_full_training(
        rheumatic_xlsx=args.rheumatic_xlsx,
        nhanes_dir=args.nhanes_dir,
        artifact_root=args.artifact_root,
        encoder_epochs=args.encoder_epochs,
        random_state=args.seed,
        skip_layer1=args.skip_layer1,
        skip_layer2=args.skip_layer2,
        verbose=True,
    )


if __name__ == "__main__":
    main()
