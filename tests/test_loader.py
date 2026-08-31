"""
Tests for the dataset loading interface. Focused on the CONTRACT (every
loader returns a consistent DatasetBundle shape) rather than re-testing
Hugging Face's own dataset loading -- that lets us swap datasets later
without rewriting tests.
"""

import pytest

from src.data.loader import load_dataset, DATASET_LOADERS


def test_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


def test_mimic_iii_not_yet_implemented():
    # Documents current state: registered as a placeholder, not usable yet.
    with pytest.raises(NotImplementedError):
        load_dataset("mimic_iii")


def test_all_registered_loaders_are_callable():
    for name, loader_fn in DATASET_LOADERS.items():
        assert callable(loader_fn), f"Loader for '{name}' is not callable"


@pytest.mark.integration
@pytest.mark.parametrize("name", ["pubmed_20k_rct", "pubmed_20k_rct_context"])
def test_pubmed_bundle_shape(name):
    """
    Marked integration: downloads real data from the Hugging Face Hub.
    Run explicitly with: pytest -m integration
    """
    bundle = load_dataset(name)

    assert bundle.text_column in bundle.dataset["train"].column_names
    assert bundle.label_column in bundle.dataset["train"].column_names
    assert len(bundle.label_names) > 0
    assert bundle.source_name
    for split in ("train", "validation", "test"):
        assert split in bundle.dataset


@pytest.mark.integration
def test_context_loader_windows_the_text():
    """The context variant should wrap each sentence with [SEP] markers and
    keep the same number of rows as the plain loader."""
    plain = load_dataset("pubmed_20k_rct")
    ctx = load_dataset("pubmed_20k_rct_context")

    assert len(ctx.dataset["train"]) == len(plain.dataset["train"])
    assert ctx.dataset["train"][0]["text"].count("[SEP]") == 2
