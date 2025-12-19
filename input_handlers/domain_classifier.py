# input_handlers/domain_classifier.py

import re
from core.model_loader import ModelLoader


class DomainClassifier:
    """
    Uses a language model to classify input text into a domain.
    Falls back to a keyword-based heuristic if the model fails or parsing is unsuccessful.
    """

    def __init__(self):
        """
        Initializes the classifier by creating an instance of the hardcoded ModelLoader.
        """
        # FIX: ModelLoader is now called without arguments as it's hardcoded for Mistral-7B.
        self.loader = ModelLoader()
        self.domains = ["medical", "legal", "general"]

        self.medical_terms = {
            "patient", "diagnosis", "symptom", "treatment", "doctor", "hospital",
            "medication", "disease", "therapy", "clinical", "health", "surgeon",
            "mri", "scan", "x-ray", "follow-up", "prescribed", "antibiotics",
            "nurse", "surgery", "prescription", "exam", "vaccine"
        }
        self.legal_terms = {
            "law", "court", "contract", "legal", "judge", "attorney", "clause",
            "agreement", "lawsuit", "liability", "compliance", "regulation",
            "arbitration", "jurisdiction", "ruling", "parties", "statute", "claim"
        }

    def classify(self, input_text: str, max_new_tokens: int = 40):
        """
        Returns: (prediction:str, confidence:float, reason_one_line:str)
        """
        if not isinstance(input_text, str) or not input_text.strip():
            return "general", 0.50, "Empty or invalid input."

        # A clear, structured prompt for better model adherence
        prompt = (
            "You are a text classification expert. Provide a concise reason and a domain for the input text.\n"
            "Follow this format exactly:\n"
            "Reason: <One-line explanation>\n"
            "Domain: <medical|legal|general>\n\n"
            f"Input: \"{input_text.strip()}\"\n\n"
            "Output:"
        )

        try:
            # Use deterministic settings for consistent classification
            model_output = self.loader.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
            reason, domain = self._parse_model_output(model_output)

            # If parsing was unsuccessful, use the fallback
            if domain is None or reason is None:
                domain, conf, reason = self._heuristic_fallback(input_text)
                return domain, conf, reason

            confidence = self._estimate_confidence(domain, model_output, input_text)
            return domain, round(confidence, 2), reason

        except Exception as e:
            # Log the error and use the fallback
            print(f"❗️ Model generation failed: {e}. Using heuristic fallback.")
            domain, conf, reason = self._heuristic_fallback(input_text)
            return domain, conf, reason

    def _parse_model_output(self, model_output: str):
        """A robust parser to extract Reason and Domain from model output."""
        reason = None
        domain = None
        for line in model_output.strip().split('\n'):
            line_clean = line.strip()
            if line_clean.lower().startswith("reason:"):
                reason = line_clean.split(":", 1)[1].strip()
            elif line_clean.lower().startswith("domain:"):
                potential_domain = line_clean.split(":", 1)[1].strip().lower()
                if potential_domain in self.domains:
                    domain = potential_domain
        return reason, domain

    def _heuristic_fallback(self, text: str):
        t = text.lower()
        med_hits = sorted([w for w in self.medical_terms if w in t])
        leg_hits = sorted([w for w in self.legal_terms if w in t])

        if len(med_hits) > len(leg_hits) and len(med_hits) > 0:
            conf = min(0.95, 0.6 + 0.05 * len(med_hits))
            reason = f"Heuristic: contains medical keywords: {', '.join(med_hits[:6])}"
            return "medical", round(conf, 2), reason
        elif len(leg_hits) > len(med_hits) and len(leg_hits) > 0:
            conf = min(0.95, 0.6 + 0.05 * len(leg_hits))
            reason = f"Heuristic: contains legal keywords: {', '.join(leg_hits[:6])}"
            return "legal", round(conf, 2), reason
        else:
            return "general", 0.60, "Heuristic: no strong medical or legal indicators found."

    def _estimate_confidence(self, domain: str, model_text: str, original_text: str) -> float:
        score = 0.7  # Higher base confidence for successful model prediction
        if domain in model_text.lower():
            score += 0.1
        if domain == "medical":
            hits = sum(1 for w in self.medical_terms if w in original_text.lower())
            score += 0.05 * min(hits, 3)
        elif domain == "legal":
            hits = sum(1 for w in self.legal_terms if w in original_text.lower())
            score += 0.05 * min(hits, 3)
        return min(0.98, score)