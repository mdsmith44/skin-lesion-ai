"""Reproducible Stage 9 runner; only train/val CSVs can be selected."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .reporting import LABELS, build_evidence, template_report


def select_records(root, split, per_class=False):
    if split not in {"train", "val"}:
        raise ValueError("Only train and val are permitted.")
    path = root / "data/processed/multimodal" / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    seen = set()
    for row in rows:
        if row["split"] != split:
            raise ValueError("CSV contains unexpected split membership.")
        label = row["class_abbreviation"]
        if not per_class:
            selected = [row]
            break
        if label not in seen:
            selected.append(row)
            seen.add(label)
        if seen == set(LABELS):
            break
    if not selected or (per_class and seen != set(LABELS)):
        raise ValueError("Missing development examples.")
    return selected, path


def resolve_image(root, row):
    parts = Path(row["image_path"]).parts
    matches = [i for i in range(len(parts)-2)
               if parts[i:i+3] == ("data", "raw", "ham10000")]
    if len(matches) != 1:
        raise ValueError("Unrecognized HAM10000 image path.")
    suffix = parts[matches[0]:]
    if ".." in suffix or suffix[-1] != row["image_id"] + ".jpg":
        raise ValueError("Invalid image path.")
    return root.joinpath(*suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--missing-case", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path(
        "/workspace/storage/skin-lesion-ai/outputs/stage9/reports"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rows, csv_path = select_records(root, args.split, args.per_class)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    from .resnet18_inference import ResNet18Inference
    from .nemotron_reporting import NemotronReporter
    classifier = ResNet18Inference()
    cases = []
    for row in rows:
        prediction = classifier.predict(resolve_image(root, row))
        evidence = build_evidence(prediction, image_id=row["image_id"],
                                  lesion_id=row["lesion_id"], split=args.split)
        cases.append((row["image_id"], evidence))
    if args.missing_case:
        missing = copy.deepcopy(cases[0][1])
        missing["classifier"] = {"predicted_class": None, "top_softmax_score": None,
                                 "softmax_scores": None, "score_type": "unknown"}
        cases.append(("synthetic_missing_prediction", missing))
    # Manifest records selection but deliberately excludes reference diagnoses.
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root / "src/stage9").glob("*.py"))}
    manifest = {"run_id": run_id, "split": args.split,
                "selection": "first row per reference class" if args.per_class else "first row",
                "source_csv": str(csv_path),
                "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "source_sha256": source_hashes, "case_ids": [c[0] for c in cases]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reporter = NemotronReporter()
    records = []
    for case_id, evidence in cases:
        result = reporter.report(evidence)
        baseline = template_report(evidence)
        artifact = {"case_id": case_id, "evidence": evidence,
                    "template_baseline": baseline, **result}
        (output / f"{case_id}.json").write_text(json.dumps(artifact, indent=2) + "\n")
        records.append({"case_id": case_id, "status": result["status"],
                        "generation_seconds": result["generation_seconds"]})
        print(case_id, result["status"], flush=True)
    summary = {"cases": len(records), "accepted": sum(r["status"] == "accepted" for r in records),
               "rejected": sum(r["status"] == "rejected" for r in records),
               "records": records, "output_directory": str(output),
               "scope": "Small development contract check; not classification or clinical validation."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
