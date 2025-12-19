# utils/combined_metrics.py

from __future__ import annotations
import numpy as np
import torch

def ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

class CombinedMetrics:
    def __init__(self, transcript=None):
        """
        Initialize the metrics class. Optionally provide a debate transcript.
        transcript: list[dict] with fields {"agent","role?","response"}
        """
        self.transcript = transcript
        self.metrics = {}

        ensure_nltk()
        import nltk
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.nltk = nltk

        # Try to load GPT-2 for perplexity; gracefully degrade if unavailable
        self.gpt2 = None
        self.tokenizer = None
        try:
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.gpt2.eval()
            if torch.cuda.is_available():
                self.gpt2.to("cuda")
        except Exception:
            self.gpt2 = None
            self.tokenizer = None  # Perplexity will be None

    def set_transcript(self, transcript):
        self.transcript = transcript

    def _only_responses(self):
        if not self.transcript:
            return []
        rs = [e.get("response", "") for e in self.transcript if e.get("agent") != "Topic"]
        return [r for r in rs if isinstance(r, str) and r.strip()]

    def compute_perplexity(self, texts):
        if not self.gpt2 or not self.tokenizer:
            return None
        vals = []
        for text in texts:
            if not text.strip():
                continue
            enc = self.tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                out = self.gpt2(**enc, labels=enc["input_ids"])
                loss = out.loss
            vals.append(torch.exp(loss).item())
        return float(np.mean(vals)) if vals else None

    def compute_lexical_diversity(self, texts):
        tokens = []
        for t in texts:
            tokens.extend(self.nltk.word_tokenize(t.lower()))
        return (len(set(tokens)) / len(tokens)) if tokens else 0.0

    def evaluate(self, transcript=None):
        """
        Compute metrics. If transcript is passed, it overrides self.transcript.
        """
        if transcript is not None:
            self.transcript = transcript
        if not self.transcript:
            raise ValueError("No transcript provided for evaluation.")

        texts = self._only_responses()
        if not texts:
            raise ValueError("Transcript has no agent responses to evaluate.")

        # BLEU
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        cc = SmoothingFunction()
        bleu_scores = []
        for hyp, ref in zip(texts, texts):
            hyp_t = self.nltk.word_tokenize(hyp.lower())
            ref_t = self.nltk.word_tokenize(ref.lower())
            bleu_scores.append(sentence_bleu([ref_t], hyp_t, smoothing_function=cc.method1))
        self.metrics["BLEU"] = float(np.mean(bleu_scores)) if bleu_scores else 0.0

        # ROUGE
        from rouge_score import rouge_scorer
        rouge_inst = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1, rL = [], []
        for hyp, ref in zip(texts, texts):
            sc = rouge_inst.score(ref, hyp)
            r1.append(sc["rouge1"].fmeasure)
            rL.append(sc["rougeL"].fmeasure)
        self.metrics["ROUGE-1"] = float(np.mean(r1)) if r1 else 0.0
        self.metrics["ROUGE-L"] = float(np.mean(rL)) if rL else 0.0

        # BERTScore
        from bert_score import score as bert_score
        device = "cuda" if torch.cuda.is_available() else "cpu"
        P, R, F1 = bert_score(texts, texts, lang="en", rescale_with_baseline=True, device=device)
        self.metrics["BERTScore-F1"] = float(F1.mean().item())

        # Length stats
        wc = [len(self.nltk.word_tokenize(t)) for t in texts]
        self.metrics["Avg Sentence Length"] = float(np.mean(wc)) if wc else 0.0
        self.metrics["Response Lengths"] = {
            "min": int(min(wc)) if wc else 0,
            "avg": float(np.mean(wc)) if wc else 0.0,
            "max": int(max(wc)) if wc else 0,
        }

        # Lexical diversity
        self.metrics["Lexical Diversity"] = float(self.compute_lexical_diversity(texts))

        # Perplexity (optional)
        self.metrics["Perplexity"] = self.compute_perplexity(texts)

        return self.metrics

    def print_summary(self):
        if not self.metrics:
            self.evaluate()
        print("\n📊 Combined Metrics Summary:")
        for k, v in self.metrics.items():
            print(f"- {k}: {v}")
