from typing import List
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class SentimentScorer:
    def __init__(self):
        model_name = "yiyanghkust/finbert-tone"
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Determine safe max length (FinBERT uses 512)
        inferred_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        # Some tokenizers set this to a very large int; cap at 512 for BERT
        self.max_length = int(min(512, inferred_max))
        self.chunk_stride = 64
        self.clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=device,
            return_all_scores=True,
            truncation=True,
        )

    @staticmethod
    def _scores_to_scalar(label_scores) -> float:
        """
        Convert list of dicts [{'label': 'Positive', 'score': p}, ...] into a scalar sentiment
        using (pos - neg). Neutral is ignored in the final scalar.
        """
        by_label = {d["label"].lower(): float(d["score"]) for d in label_scores}
        pos = by_label.get("positive", 0.0)
        neg = by_label.get("negative", 0.0)
        return pos - neg

    def _chunk_text(self, text: str) -> List[str]:
        # Tokenize with overflow to create multiple 512-token windows with stride
        enc = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.chunk_stride,
        )
        input_ids = enc.get("input_ids", [])
        # HF returns list for overflow; ensure list of lists
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            chunks = [
                self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in input_ids
            ]
            chunks = [c for c in chunks if c]
            return chunks or [text]
        return [text]

    def score_article(self, text: str) -> float:
        chunks = self._chunk_text(text)
        outputs = self.clf(chunks)
        if not outputs:
            return 0.0
        scalars = []
        for out in outputs:
            if not isinstance(out, list):
                continue
            scalars.append(self._scores_to_scalar(out))
        if not scalars:
            return 0.0
        return float(np.mean(scalars))

    def score_articles(self, articles: List[str]) -> List[float]:
        if not articles:
            return []
        return [self.score_article(a) for a in articles]
