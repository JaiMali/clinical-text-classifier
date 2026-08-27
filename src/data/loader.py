"""
Dataset loading interface.

Design intent: every dataset-specific detail (download source, column names,
label taxonomy, parsing quirks) lives ONLY in this file. The training script,
model code, and API never import a dataset directly -- they call
`load_dataset(name)` and get back a standardized DatasetBundle.

To add a new dataset (e.g. MIMIC-III later), write a new `_load_<name>()`
function that returns a DatasetBundle, and register it in DATASET_LOADERS.
Nothing outside this file needs to change.
"""

from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset as hf_load_dataset, ClassLabel, DatasetDict


@dataclass
class DatasetBundle:
    """Standardized output of any dataset loader in this project."""

    dataset: DatasetDict  # must have 'train', 'validation' (or 'val'), 'test' splits
    text_column: str  # name of the column containing raw text
    label_column: str  # name of the column containing integer label ids
    label_names: list[str]  # index-aligned human-readable label names
    source_name: str  # human-readable name, e.g. "PubMed 20k RCT"


def _load_pubmed_20k_rct() -> DatasetBundle:
    """
    PubMed 20k RCT: sentences from structured medical abstracts, each labeled
    with its rhetorical role (BACKGROUND, OBJECTIVE, METHODS, RESULTS,
    CONCLUSIONS). Hosted on the Hugging Face Hub.
    """
    ds = hf_load_dataset("armanc/pubmed-rct20k")

    # Hugging Face gives us train/validation/test already -- if a given
    # source doesn't, this is the place to carve out a val split.
    #
    # This Hub copy stores `label` as a raw lowercase string ("methods",
    # "objective", ...) rather than a ClassLabel. Cast it to a ClassLabel
    # with one explicit ordering derived from train, so label ids are
    # integers and are consistent across all three splits.
    label_names = sorted(set(ds["train"]["label"]))
    ds = ds.cast_column("label", ClassLabel(names=label_names))

    return DatasetBundle(
        dataset=ds,
        text_column="text",
        label_column="label",
        label_names=label_names,
        source_name="PubMed 20k RCT",
    )


def _load_mimic_iii() -> DatasetBundle:
    """
    Placeholder for the future MIMIC-III variant. Implement once PhysioNet
    credentialing is complete. Must return the same DatasetBundle shape as
    every other loader -- that's the whole point.
    """
    raise NotImplementedError(
        "MIMIC-III loader not yet implemented -- pending PhysioNet credentialing. "
        "See loader.py docstring for the interface every dataset must satisfy."
    )


# Registry: add new datasets here, one line each.
DATASET_LOADERS: dict[str, Callable[[], DatasetBundle]] = {
    "pubmed_20k_rct": _load_pubmed_20k_rct,
    "mimic_iii": _load_mimic_iii,
}


def load_dataset(name: str) -> DatasetBundle:
    """Entry point everything else in the project should use."""
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_LOADERS.keys())}"
        )
    return DATASET_LOADERS[name]()
