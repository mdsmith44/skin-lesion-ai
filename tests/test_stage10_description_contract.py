"""Synthetic checks for the limited Stage 10 visual-description contract."""
import unittest

from src.stage10.description_contract import validate_description


class DescriptionContractTests(unittest.TestCase):
    def assert_rejected_for(self, text, reason):
        result = validate_description(text)
        self.assertEqual(result["validation_status"], "rejected")
        self.assertIsNone(result["validated_description"])
        self.assertIn(reason, result["rejection_reasons"])

    def test_visible_description_is_accepted_and_trimmed(self):
        result = validate_description(" A small brown patch has uneven edges. The surrounding skin is pink. ")
        self.assertEqual(result["validation_status"], "accepted")
        self.assertEqual(result["validated_description"],
                         "A small brown patch has uneven edges. The surrounding skin is pink.")
        self.assertEqual(result["rejection_reasons"], [])

    def test_diagnostic_terms_and_class_abbreviation(self):
        self.assert_rejected_for("This is melanoma.", "diagnostic_or_disease_term")
        self.assert_rejected_for("The lesion is bkl.", "class_abbreviation")
        self.assert_rejected_for("It appears to be eczema.", "diagnostic_or_disease_term")

    def test_malignancy_claim(self):
        self.assert_rejected_for("This area is benign, not cancer.", "diagnostic_or_disease_term")

    def test_treatment_recommendation(self):
        self.assert_rejected_for("Apply a cream and consult a dermatologist.",
                                 "treatment_recommendation")

    def test_clinical_interpretation(self):
        self.assert_rejected_for("The pattern is indicative of a skin condition.",
                                 "clinical_interpretation")
        self.assert_rejected_for("The colors are suggestive of disease.",
                                 "clinical_interpretation")

    def test_unsupported_health_claim(self):
        self.assert_rejected_for("The surrounding skin is healthy.",
                                 "unsupported_health_status")
        self.assert_rejected_for("The tissue appears unhealthy.",
                                 "unsupported_health_status")

    def test_nonvisual_claim_and_length(self):
        self.assert_rejected_for("The area is painful.", "nonvisual_patient_claim")
        self.assert_rejected_for("It is brown. Its edges are uneven. The skin is pink.",
                                 "more_than_two_sentences")

    def test_empty_output(self):
        self.assert_rejected_for("  \n", "empty_output")


if __name__ == "__main__":
    unittest.main()
