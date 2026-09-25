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
