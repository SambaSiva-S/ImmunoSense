"""Layer2Bundle: atomic loader for all 6 Layer 2 artifacts.

Loads in one call:
    1. lgb_model.pkl          (Pillar B LightGBM classifier)
    2. xgb_model.json         (Pillar C XGBoost classifier)
    3. encoder.pt             (Pillar A PyTorch contrastive encoder)
    4. centroids.npz          (Per-class normalized centroids on unit sphere)
    5. scaler.pkl             (StandardScaler for encoder input)
    6. label_encoder.pkl      (sklearn LabelEncoder for class names)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Layer2Bundle:
    """Container for all Layer 2 inference-time artifacts."""

    lgb_model: Any                 # LightGBM classifier
    xgb_model: Any                 # XGBoost classifier
    encoder: Any                   # PyTorch DiseaseEncoder
    centroids: dict                # {class_name: np.ndarray}
    scaler: Any                    # sklearn StandardScaler
    label_encoder: Any             # sklearn LabelEncoder
    feature_names: list            # input feature ordering
    class_names: list              # class label ordering

    @classmethod
    def load(cls, artifact_dir: Path) -> "Layer2Bundle":
        """Load all Layer 2 artifacts from disk.

        Args:
            artifact_dir: directory containing the 6 artifact files

        Raises:
            FileNotFoundError if any artifact is missing.
            ImportError if PyTorch is not installed (encoder requires it).
        """
        import joblib
        import numpy as np
        import xgboost as xgb

        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Layer 2 artifacts directory not found: {artifact_dir}. "
                "Train via `python -m immunosense.agents.biomarker.layer2.train`."
            )

        # 1. LightGBM
        lgb_path = artifact_dir / "lgb_model.pkl"
        if not lgb_path.exists():
            raise FileNotFoundError(f"Layer 2: {lgb_path} missing. Re-train Layer 2.")
        lgb_model = joblib.load(lgb_path)

        # 2. XGBoost (use native JSON format for cross-version compatibility)
        xgb_path = artifact_dir / "xgb_model.json"
        if not xgb_path.exists():
            raise FileNotFoundError(f"Layer 2: {xgb_path} missing. Re-train Layer 2.")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))

        # 3. PyTorch encoder (state dict)
        encoder_path = artifact_dir / "encoder.pt"
        meta_path = artifact_dir / "layer2_metadata.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Layer 2: {encoder_path} missing. Re-train Layer 2.")
        if not meta_path.exists():
            raise FileNotFoundError(f"Layer 2: {meta_path} missing. Re-train Layer 2.")

        meta = joblib.load(meta_path)

        # Import torch lazily — raises ImportError with install hint if missing
        from immunosense.agents.biomarker.layer2.encoder import DiseaseEncoder
        import torch

        encoder = DiseaseEncoder(
            input_dim=meta["input_dim"],
            embedding_dim=meta["embedding_dim"],
        )
        encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        encoder.eval()

        # 4. Centroids (saved as npz)
        centroids_path = artifact_dir / "centroids.npz"
        if not centroids_path.exists():
            raise FileNotFoundError(f"Layer 2: {centroids_path} missing. Re-train Layer 2.")
        npz = np.load(centroids_path)
        centroids = {key: npz[key] for key in npz.files}

        # 5. Scaler
        scaler_path = artifact_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Layer 2: {scaler_path} missing. Re-train Layer 2.")
        scaler = joblib.load(scaler_path)

        # 6. Label encoder
        le_path = artifact_dir / "label_encoder.pkl"
        if not le_path.exists():
            raise FileNotFoundError(f"Layer 2: {le_path} missing. Re-train Layer 2.")
        label_encoder = joblib.load(le_path)

        return cls(
            lgb_model=lgb_model,
            xgb_model=xgb_model,
            encoder=encoder,
            centroids=centroids,
            scaler=scaler,
            label_encoder=label_encoder,
            feature_names=meta.get("feature_names", []),
            class_names=list(label_encoder.classes_),
        )

    def save(self, artifact_dir: Path) -> None:
        """Persist all 6 artifacts to artifact_dir.

        Used by layer2/train.py after training.
        """
        import joblib
        import numpy as np
        import torch

        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.lgb_model, artifact_dir / "lgb_model.pkl")
        self.xgb_model.save_model(str(artifact_dir / "xgb_model.json"))
        torch.save(self.encoder.state_dict(), artifact_dir / "encoder.pt")
        np.savez(artifact_dir / "centroids.npz", **self.centroids)
        joblib.dump(self.scaler, artifact_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, artifact_dir / "label_encoder.pkl")
        joblib.dump(
            {
                "input_dim": self.encoder.input_dim,
                "embedding_dim": self.encoder.embedding_dim,
                "feature_names": self.feature_names,
            },
            artifact_dir / "layer2_metadata.pkl",
        )
