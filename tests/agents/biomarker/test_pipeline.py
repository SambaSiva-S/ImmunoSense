"""Tests for Layer2Bundle save/load round-trip."""

from pathlib import Path

import numpy as np
import pytest

from immunosense.agents.biomarker.layer2 import Layer2Bundle


def test_bundle_loads_from_session_artifacts(loaded_layer2_bundle):
    """Bundle should load all 6 artifacts cleanly from session fixture."""
    bundle = loaded_layer2_bundle
    assert bundle.lgb_model is not None
    assert bundle.xgb_model is not None
    assert bundle.encoder is not None
    assert bundle.scaler is not None
    assert bundle.label_encoder is not None
    assert len(bundle.centroids) == 7
    assert len(bundle.class_names) == 7
    assert len(bundle.feature_names) == 28


def test_bundle_class_names_alphabetical(loaded_layer2_bundle):
    """Class names should be alphabetical (matches DISEASE_CLASSES constant)."""
    assert loaded_layer2_bundle.class_names == sorted(loaded_layer2_bundle.class_names)


def test_bundle_centroids_are_unit_norm(loaded_layer2_bundle):
    """Each centroid must lie on the unit sphere (within floating-point tolerance)."""
    for name, c in loaded_layer2_bundle.centroids.items():
        norm = np.linalg.norm(c)
        assert abs(norm - 1.0) < 1e-3, f"centroid {name} norm = {norm}"


def test_bundle_encoder_input_dim_28(loaded_layer2_bundle):
    """Encoder must expect the 28-feature input vector."""
    assert loaded_layer2_bundle.encoder.input_dim == 28


def test_bundle_encoder_embedding_dim_128(loaded_layer2_bundle):
    assert loaded_layer2_bundle.encoder.embedding_dim == 128


def test_bundle_load_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        Layer2Bundle.load(Path("/nonexistent/biomarker/layer2"))


def test_bundle_save_then_reload_round_trip(loaded_layer2_bundle, tmp_path):
    """Saving and reloading a bundle should preserve all artifacts."""
    new_dir = tmp_path / "new_layer2"
    loaded_layer2_bundle.save(new_dir)
    # All 7 artifact files should exist
    for fname in [
        "lgb_model.pkl", "xgb_model.json", "encoder.pt",
        "centroids.npz", "scaler.pkl", "label_encoder.pkl",
        "layer2_metadata.pkl",
    ]:
        assert (new_dir / fname).exists(), f"missing artifact: {fname}"
    # Reload should succeed
    reloaded = Layer2Bundle.load(new_dir)
    assert reloaded.class_names == loaded_layer2_bundle.class_names


def test_bundle_class_names_match_label_encoder(loaded_layer2_bundle):
    """class_names is derived from label_encoder.classes_."""
    assert list(loaded_layer2_bundle.label_encoder.classes_) == loaded_layer2_bundle.class_names
