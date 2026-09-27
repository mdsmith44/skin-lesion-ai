"""Read the saved Stage 7 records without changing their split or targets.

This map-style dataset can be consumed by a PyTorch DataLoader with a
multimodal collator later. It performs no tokenization or GPU operations.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CATEGORIES = frozenset({"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"})
REQUIRED_COLUMNS = frozenset({
    "image_id", "image_path", "lesion_id", "split",
    "class_abbreviation", "instruction", "response",
})


@dataclass
class MultimodalSample:
    """An RGB image and its metadata; the answer stays separate by default."""

    image: Image.Image
    image_path: Path
    image_id: str
    lesion_id: str
    split: str
    class_abbreviation: str
    instruction: str
    response: str

    def evaluation_messages(self) -> list[dict]:
        """Only the image and instruction are supplied to generation."""
        return [{"role": "user", "content": [
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.instruction},
        ]}]

    def training_messages(self) -> list[dict]:
        """Include the stored assistant answer for supervised training."""
        return self.evaluation_messages() + [{
            "role": "assistant",
            "content": [{"type": "text", "text": self.response}],
        }]


class HAM10000MultimodalDataset:
    """Load one existing development split, opening images only on access.

    ``project_root`` is explicit so host paths and Docker paths work alike.
    Test is intentionally unavailable in this development-only adapter.
    CSV row order, instructions, responses, and split assignments are retained.
    """

    def __init__(self, project_root: str | Path, split: str = "train"):
        if split not in {"train", "val"}:
            raise ValueError("Stage 8 development accepts only 'train' or 'val'.")
        self.project_root = Path(project_root).expanduser().resolve()
        self.split = split
        csv_path = self.project_root / "data/processed/multimodal" / f"{split}.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            self._records = list(reader)
        if not self._records:
            raise ValueError(f"{csv_path}: empty split")
        seen = set()
        for row_number, row in enumerate(self._records, start=2):
            if any(not row.get(key) or not row[key].strip() for key in REQUIRED_COLUMNS):
                raise ValueError(f"{csv_path}: empty required field at record {row_number}")
            if row["split"] != split:
                raise ValueError(f"{csv_path}: unexpected split at record {row_number}")
            if row["image_id"] in seen:
                raise ValueError(f"{csv_path}: duplicate image {row['image_id']}")
            seen.add(row["image_id"])
            if row["class_abbreviation"] not in CATEGORIES:
                raise ValueError(f"{csv_path}: unknown category at record {row_number}")
            if row["response"] != row["class_abbreviation"]:
                raise ValueError(f"{csv_path}: expected saved abbreviation target at record {row_number}")
            self._resolve_image_path(row)

    def _resolve_image_path(self, row: dict) -> Path:
        # Keep the path below data/raw/ham10000; discard the old host prefix.
        parts = Path(row["image_path"]).parts
        marker = ("data", "raw", "ham10000")
        matches = [i for i in range(len(parts) - 2) if parts[i:i + 3] == marker]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized HAM10000 image path: {row['image_path']}")
        suffix = parts[matches[0]:]
        if ".." in suffix or Path(suffix[-1]).name != row["image_id"] + ".jpg":
            raise ValueError(f"Invalid image path for {row['image_id']}")
        return self.project_root.joinpath(*suffix)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> MultimodalSample:
        row = self._records[index]
        path = self._resolve_image_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"Image {row['image_id']} is missing: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        return MultimodalSample(
            image=image, image_path=path,
            **{key: row[key] for key in REQUIRED_COLUMNS - {"image_path"}},
        )

class HAM10000NeMoDataset:
    """Expose the validated Stage 7 dataset in NeMo VLM conversation format."""

    def __init__(
        self,
        project_root: str | Path,
        split: str = "train",
        indices: list[int] | None = None,
    ):
        self.dataset = HAM10000MultimodalDataset(
            project_root=project_root,
            split=split,
        )

        if indices is None:
            self.indices = list(range(len(self.dataset)))
        else:
            self.indices = list(indices)

            if not self.indices:
                raise ValueError("indices must not be empty.")

            if min(self.indices) < 0 or max(self.indices) >= len(self.dataset):
                raise IndexError("Dataset index is out of range.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[self.indices[index]]

        return {
            "conversation": sample.training_messages(),
        }


def make_ham10000_dataset(
    project_root: str | Path = "/workspace/skin-lesion-ai",
    split: str = "train",
    indices: list[int] | None = None,
    **kwargs,
):
    """Factory used by NeMo AutoModel's FinetuneRecipeForVLM."""
    return HAM10000NeMoDataset(
        project_root=project_root,
        split=split,
        indices=indices,
    )


def make_ham10000_balanced_dataset(
    project_root: str | Path = "/workspace/skin-lesion-ai",
    split: str = "train",
    samples_per_class: int = 1000,
    seed: int = 42,
    **kwargs,
):
    """Create a deterministic class-balanced logical training dataset.

    Each class contributes exactly ``samples_per_class`` logical positions.

    If a class has fewer available images, all of its images are used before
    indices are repeated as evenly as possible. If a class has more available
    images, a subset is selected without replacement.

    NeMo may then apply its normal DistributedSampler to this logical dataset.
    """
    import random

    if split != "train":
        raise ValueError(
            "Balanced sampling is intended only for the training split."
        )

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    base = HAM10000MultimodalDataset(
        project_root=project_root,
        split=split,
    )

    by_class = {label: [] for label in sorted(CATEGORIES)}

    for index, record in enumerate(base._records):
        by_class[record["class_abbreviation"]].append(index)

    rng = random.Random(seed)
    balanced_indices = []

    for label in sorted(CATEGORIES):
        candidates = by_class[label]

        if not candidates:
            raise ValueError(f"No training samples found for class {label!r}")

        # Shuffle once so repeated cycling is not tied to CSV ordering.
        candidates = candidates.copy()
        rng.shuffle(candidates)

        if len(candidates) >= samples_per_class:
            selected = candidates[:samples_per_class]
        else:
            full_repeats, remainder = divmod(
                samples_per_class,
                len(candidates),
            )

            selected = candidates * full_repeats
            selected += candidates[:remainder]

        balanced_indices.extend(selected)

    # Randomize class ordering while preserving the exact balanced counts.
    rng.shuffle(balanced_indices)

    return HAM10000NeMoDataset(
        project_root=project_root,
        split=split,
        indices=balanced_indices,
    )

def make_ham10000_tempered_dataset(
    project_root: str | Path = "/workspace/skin-lesion-ai",
    split: str = "train",
    total_samples: int = 7000,
    alpha: float = 0.5,
    seed: int = 42,
    **kwargs,
):
    """Create a deterministic tempered-distribution training dataset.

    Class exposure is proportional to n_c ** alpha, where n_c is the
    natural number of training images for class c.

    alpha = 1.0 preserves the natural class proportions.
    alpha = 0.0 gives equal class proportions.
    Values between 0 and 1 provide intermediate balancing.

    Integer class targets are assigned using the largest-remainder method
    so that their sum is exactly ``total_samples``.

    NeMo may then apply its normal DistributedSampler to this logical dataset.
    """
    import math
    import random

    if split != "train":
        raise ValueError(
            "Tempered sampling is intended only for the training split."
        )

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    base = HAM10000MultimodalDataset(
        project_root=project_root,
        split=split,
    )

    labels = sorted(CATEGORIES)
    by_class = {label: [] for label in labels}

    for index, record in enumerate(base._records):
        by_class[record["class_abbreviation"]].append(index)

    for label in labels:
        if not by_class[label]:
            raise ValueError(f"No training samples found for class {label!r}")

    # Desired class exposure is proportional to n_c ** alpha.
    weights = {
        label: len(by_class[label]) ** alpha
        for label in labels
    }
    weight_sum = sum(weights.values())

    exact_targets = {
        label: total_samples * weights[label] / weight_sum
        for label in labels
    }

    # Start with the integer floor of each target.
    targets = {
        label: math.floor(exact_targets[label])
        for label in labels
    }

    # Assign leftover positions according to largest fractional remainder.
    remaining = total_samples - sum(targets.values())

    remainder_order = sorted(
        labels,
        key=lambda label: (
            exact_targets[label] - targets[label],
            label,
        ),
        reverse=True,
    )

    for label in remainder_order[:remaining]:
        targets[label] += 1

    rng = random.Random(seed)
    tempered_indices = []

    for label in labels:
        candidates = by_class[label].copy()
        rng.shuffle(candidates)

        target = targets[label]

        if len(candidates) >= target:
            selected = candidates[:target]
        else:
            full_repeats, remainder = divmod(
                target,
                len(candidates),
            )

            selected = candidates * full_repeats
            selected += candidates[:remainder]

        tempered_indices.extend(selected)

    rng.shuffle(tempered_indices)

    if len(tempered_indices) != total_samples:
        raise RuntimeError(
            "Tempered dataset construction produced an unexpected size: "
            f"{len(tempered_indices)} != {total_samples}"
        )

    return HAM10000NeMoDataset(
        project_root=project_root,
        split=split,
        indices=tempered_indices,
    )
