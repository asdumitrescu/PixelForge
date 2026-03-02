"""Tests for the model registry."""

from src.engine.model_registry import (
    MODEL_REGISTRY,
    get_all_model_ids,
    get_model_entry,
    get_models_for_scale,
)


def test_registry_not_empty() -> None:
    assert len(MODEL_REGISTRY) > 0


def test_all_entries_have_urls() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.url, f"Model {model_id} has no URL"
        assert entry.url.startswith("https://"), f"Model {model_id} URL is not HTTPS"


def test_all_entries_have_valid_scale() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.scale in (2, 3, 4), f"Model {model_id} has invalid scale: {entry.scale}"


def test_all_entries_have_filenames() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.filename, f"Model {model_id} has no filename"
        assert entry.filename.endswith((".pth", ".safetensors")), (
            f"Model {model_id} has unexpected format: {entry.filename}"
        )


def test_no_duplicate_filenames() -> None:
    filenames = [entry.filename for entry in MODEL_REGISTRY.values()]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames in registry"


def test_no_duplicate_ids() -> None:
    ids = get_all_model_ids()
    assert len(ids) == len(set(ids)), "Duplicate IDs in registry"


def test_get_model_entry_found() -> None:
    entry = get_model_entry("realesrgan-x4plus")
    assert entry is not None
    assert entry.display_name == "Real-ESRGAN x4 Plus"


def test_get_model_entry_not_found() -> None:
    entry = get_model_entry("nonexistent-model")
    assert entry is None


def test_get_models_for_scale_4() -> None:
    models = get_models_for_scale(4)
    assert len(models) >= 2  # At least realesrgan-x4plus and anime
    assert all(m.scale == 4 for m in models)


def test_get_models_for_scale_2() -> None:
    models = get_models_for_scale(2)
    assert len(models) >= 1
    assert all(m.scale == 2 for m in models)


def test_all_entries_have_display_names() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.display_name, f"Model {model_id} has no display name"


def test_all_entries_have_descriptions() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.description, f"Model {model_id} has no description"


def test_all_entries_have_file_size() -> None:
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.file_size_mb > 0, f"Model {model_id} has invalid file size"
