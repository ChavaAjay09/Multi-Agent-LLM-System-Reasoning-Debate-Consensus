# core/agent.py

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from input_handlers.domain_classifier import DomainClassifier
from core.model_loader import ModelLoader
import re

class Agent:
    """
    Orchestrates domain classification and role-based response generation
    using a shared, singleton ModelLoader instance.
    Responses are designed to be meaningful, emotionally aware, and multi-sentence.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role.lower()

        self.model_loader = ModelLoader()
        self.domain_classifier = DomainClassifier()

        self.prompts = {
            "medical-analytical": (
                "You are a medical analyst. Respond thoughtfully, explaining the reasoning clearly. "
                "Make your answer meaningful and emotionally aware of the impact on patients. "
                "Write 2-3 sentences."
            ),
            "medical-creative": (
                "You are a creative medical expert. Respond imaginatively, considering patient emotions, "
                "possible future scenarios, and innovative treatments. Use narrative or metaphor if helpful. "
                "Write 2-3 sentences."
            ),
            "medical-critical": (
                "You are a medical critic. Critically evaluate the situation, highlighting ethical, emotional, "
                "and clinical aspects. Write 2-3 sentences."
            ),
            "legal-analytical": (
                "You are a legal analyst. Provide a well-reasoned and meaningful analysis, considering "
                "ethical, societal, and emotional consequences. Write 2-3 sentences."
            ),
            "legal-creative": (
                "You are a creative legal thinker. Offer imaginative solutions or perspectives, "
                "emphasizing the human and societal impact. Write 2-3 sentences."
            ),
            "legal-critical": (
                "You are a legal critic. Critically evaluate the legal matter, addressing emotional, "
                "moral, and societal dimensions. Write 2-3 sentences."
            ),
            "general-analytical": (
                "You are an analytical expert. Analyze the situation with depth, incorporating logical reasoning "
                "and emotional understanding. Write 2-3 sentences."
            ),
            "general-creative": (
                "You are a creative thinker. Offer a meaningful, emotionally rich perspective, using stories, "
                "examples, or metaphors. Write 2-3 sentences."
            ),
            "general-critical": (
                "You are a critical examiner. Evaluate with insight, highlighting logical and emotional implications. "
                "Write 2-3 sentences."
            ),
        }

    def get_prompt(self, domain: str, input_text: str) -> str:
        key = f"{domain}-{self.role}"
        base_prompt = self.prompts.get(key, self.prompts["general-analytical"])
        emotional_prefix = "Respond with empathy, conveying emotions and meaningful reasoning. "
        return emotional_prefix + base_prompt + f"\n{input_text.strip()}\nAnswer:"

    def respond(
        self,
        domain: str,
        input_text: str,
        max_new_tokens: int = None,
        temperature: float = None,
        do_sample: bool = None
    ):
        prompt = self.get_prompt(domain, input_text)
        max_tokens = max_new_tokens or 150
        temp = temperature if temperature is not None else (1.0 if self.role == "creative" else 0.7)
        sample = do_sample if do_sample is not None else True
        top_p = 0.95

        if hasattr(self.model_loader, "clear_kv_cache"):
            try:
                self.model_loader.clear_kv_cache()
            except Exception:
                pass

        try:
            raw_text = self.model_loader.generate_text(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=sample
            )
            formatted = self._format_thought_and_response(raw_text)
        except Exception as e:
            formatted = f'Thought: (generation error)\nResponse: "[Error generating response: {str(e)}]"'

        return {"agent": self.name, "role": self.role, "response": formatted}

    def _format_thought_and_response(self, text: str) -> str:
        """
        Format raw LLM text into:
        Thought: <single concise sentence>
        Response: "<1-2 sentence reply>"
        """
        if not text:
            return 'Thought: (no thought generated)\nResponse: "(no response generated)"'

        s = re.sub(r'\s+', ' ', text).strip()
        # Remove bracketed tags like [SPEAKER], [CORPORATE], etc.
        s = re.sub(r'\[[^\]]*\]', '', s).strip()
        # Remove duplicated "Thought:" prefixes and narrator/meta inserts
        s = re.sub(r'(?:^|\s)(Thought:\s*){2,}', r' Thought: ', s, flags=re.IGNORECASE)
        # Cut off known meta/narration segments
        s = re.split(r'\b(?:Casey is|Jordan is|Alex is|Note:|Also:|Topic:)\b', s)[0].strip()
        # Remove leftover "Topic:" fragments anywhere
        s = re.sub(r'\bTopic:\s*', '', s, flags=re.IGNORECASE).strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', s)
        clean = []
        for sent in sentences:
            t = sent.strip(' \'"“”„‟').strip()
            if t:
                t = re.sub(r'\s+', ' ', t)
                clean.append(t)

        if not clean:
            return 'Thought: (no thought generated)\nResponse: "(no response generated)"'

        thought = clean[0]
        # Build 1–2 sentence response different from thought
        response_parts = []
        for cand in clean[1:3]:
            if cand and cand.lower() != thought.lower():
                response_parts.append(cand)
        if not response_parts:
            # Fallback: split thought by comma to create a short response
            response_parts = [thought.split(',', 1)[-1].strip()] if ',' in thought else [thought]

        response_text = " ".join(response_parts).strip().rstrip(' "\'“”')
        response_text = re.sub(r'"+', '"', response_text)
        response_text = re.sub(r"'+", "'", response_text)

        return f"Thought: {thought}\nResponse: \"{response_text}\""
