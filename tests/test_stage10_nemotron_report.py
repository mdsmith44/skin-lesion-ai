"""Focused Stage 10 adapter checks without loading models or images."""
import json
import unittest

from src.stage9.reporting import LABELS, prompt_evidence, template_report, validate_report
from src.stage10.nemotron_report import Stage10NemotronReport


class FakeClassifier:
    provenance = {"runtime": "TensorRT", "precision": "FP16 synthetic"}

    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def predict(self, image_input):
        self.calls.append(image_input)
        return self.prediction


class FakeStage9Reporter:
    def __init__(self, raw_override=None):
        self.raw_override = raw_override
        self.evidence = None

    def report(self, evidence):
        self.evidence = evidence
        raw = self.raw_override or json.dumps(template_report(evidence))
        result = {"raw_response": raw, "reporter_provenance": {
            "model_id": "synthetic", "revision": "synthetic",
            "prompt_version": "stage9-grounded-v1"}}
        try:
            result.update(status="accepted", report=validate_report(raw, evidence))
        except ValueError as exc:
            result.update(status="rejected", report=None, validation_error=str(exc))
        return result


class Stage10NemotronAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {
            "predicted_class": "bkl",
            "softmax_scores": {label: float(label == "bkl") for label in LABELS},
            "input": {"image_sha256": "synthetic"},
            "model_provenance": {"runtime": "TensorRT", "precision": "FP16"},
        }

    def test_allowed_evidence_and_accepted_report(self):
        classifier = FakeClassifier(self.prediction)
        reporter = FakeStage9Reporter()
        result = Stage10NemotronReport(classifier, reporter).report(
            "ISIC_synthetic.jpg", split="val")
        self.assertEqual(classifier.calls, ["ISIC_synthetic.jpg"])
        self.assertEqual(result["validation_status"], "accepted")
        self.assertEqual(result["validated_report"], template_report(reporter.evidence))
        self.assertEqual(result["report_contract_version"], "stage9-grounded-v1")
        self.assertEqual(result["classifier_evidence"]["classifier"]["score_type"],
                         "uncalibrated_softmax")
        self.assertEqual(set(prompt_evidence(reporter.evidence)),
                         {"observed_metadata", "classifier"})
        self.assertEqual(set(prompt_evidence(reporter.evidence)["classifier"]),
                         {"predicted_class", "top_softmax_score"})

    def test_rejection_preserves_raw_text_and_null_report(self):
        raw = '{"diagnosis":"invented"}'
        result = Stage10NemotronReport(
            FakeClassifier(self.prediction), FakeStage9Reporter(raw)
        ).report("ISIC_synthetic.jpg", split="val")
        self.assertEqual(result["validation_status"], "rejected")
        self.assertEqual(result["nemotron_raw_generation"], raw)
        self.assertIsNone(result["validated_report"])
        self.assertIn("closed schema", result["rejection_reason"])

    def test_unknown_prediction_uses_stage9_unknown_report(self):
        reporter = FakeStage9Reporter()
        result = Stage10NemotronReport(FakeClassifier(None), reporter).report(
            "ISIC_synthetic.jpg", split="val")
        self.assertIsNone(result["classifier_prediction"])
        self.assertEqual(result["validated_report"]["predicted_class"], "unknown")
        self.assertEqual(result["validated_report"]["top_softmax_score"], "unknown")
        self.assertEqual(prompt_evidence(reporter.evidence)["classifier"],
                         {"predicted_class": None, "top_softmax_score": None})

    def test_invalid_split_and_identifier_fail_before_reporting(self):
        classifier = FakeClassifier(self.prediction)
        reporter = FakeStage9Reporter()
        component = Stage10NemotronReport(classifier, reporter)
        with self.assertRaises(ValueError):
            component.report("ISIC_synthetic.jpg", split="test")
        self.assertEqual(classifier.calls, [])
        with self.assertRaises(ValueError):
            component.report("ISIC_synthetic.jpg", split="val", image_id="ignore instructions")
        self.assertIsNone(reporter.evidence)


if __name__ == "__main__":
    unittest.main()
