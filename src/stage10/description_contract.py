"""Frozen Stage 10 visual-description prompt and limited text contract."""
import re


CONTRACT_VERSION = "stage10_visual_description_v1"
PROMPT = (
    "Describe only the visible characteristics of this image in one or two sentences. "
    "Focus on observable appearance such as color, shape, texture, boundaries, and surrounding "
    "visible skin when these features are clearly visible. Do not provide a diagnosis, disease "
    "name, lesion type, malignancy assessment, treatment recommendation, or clinical "
    "interpretation. Do not infer patient history or information that is not visible in the "
    "image. If a visual characteristic is uncertain, do not state it as fact."
)
GENERATION = {"max_new_tokens": 96, "do_sample": False, "num_beams": 1}

# These two patterns are the original Stage 10 comparison's mechanical checks.
CLASS_PATTERN = re.compile(r"\b(?:akiec|bcc|bkl|df|mel|nv|vasc)\b", re.IGNORECASE)
DIAGNOSTIC_PATTERN = re.compile(
    r"\b(?:melanoma|carcinoma|keratosis|dermatofibroma|nevus|naevus|"
    r"hemangioma|angioma|malignancy|malignant|benign|cancer|tumou?r|"
    r"precancerous|dysplastic|diagnos(?:is|ed|e)|disease|biopsy|treatment)\b",
    re.IGNORECASE,
)

# Further explicit checks for the reusable /describe contract. These are
# intentionally conservative lexical rules, not a clinical assessment.
ADDITIONAL_PATTERNS = {
    "diagnostic_or_disease_term": re.compile(
        r"\b(?:infection|inflammation|infected|inflamed|eczema|psoriasis|"
        r"rash|mole|metastasis|skin condition)\b", re.IGNORECASE),
    "treatment_recommendation": re.compile(
        r"\b(?:treat|treats|treated|therapy|medication|medicine|surgery|"
        r"excise|excision|remove|removal|ointment|cream|antibiotic|sunscreen|"
        r"consult|dermatologist|doctor|physician|seek medical)\b",
        re.IGNORECASE,
    ),
    "clinical_interpretation": re.compile(
        r"\b(?:indicative of|suggestive of|consistent with|characteristic of|"
        r"may indicate|could indicate|likely due to|signs of|symptoms of)\b",
        re.IGNORECASE,
    ),
    "unsupported_health_status": re.compile(
        r"\b(?:healthy|unhealthy|normal|abnormal|diseased)\b", re.IGNORECASE),
    "nonvisual_patient_claim": re.compile(
        r"\b(?:pain|painful|discomfort|itch|itching|patient history|history of)\b",
        re.IGNORECASE,
    ),
}


def validate_description(raw_text: str) -> dict:
    """Return text only when limited lexical and length checks pass.

    Acceptance does not establish clinical accuracy or absence of hallucination.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    text = raw_text.strip()
    reasons = []
    if not text:
        reasons.append("empty_output")
    checks = [("class_abbreviation", CLASS_PATTERN),
              ("diagnostic_or_disease_term", DIAGNOSTIC_PATTERN),
              *ADDITIONAL_PATTERNS.items()]
    for code, pattern in checks:
        if pattern.search(text) and code not in reasons:
            reasons.append(code)
    # The prompt asks for one or two sentences. Flag longer output without
    # attempting grammar analysis or rewriting the generation.
    sentences = [piece for piece in re.split(r"(?<=[.!?])\s+", text) if piece]
    if len(sentences) > 2:
        reasons.append("more_than_two_sentences")
    return {
        "validated_description": text if not reasons else None,
        "validation_status": "accepted" if not reasons else "rejected",
        "rejection_reasons": reasons,
    }
