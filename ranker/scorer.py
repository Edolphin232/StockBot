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
        # Reasonable default; pipeline will internally micro-batch on GPU
        self.batch_size = 16
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
        outputs = self.clf(chunks, batch_size=self.batch_size)
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
        """
        Efficiently score many articles by batching all chunks across the list.
        """
        if not articles:
            return []
        # Build flattened list of chunks with mapping back to article index
        chunk_texts: List[str] = []
        article_indices: List[int] = []
        for idx, text in enumerate(articles):
            if not isinstance(text, str) or not text:
                continue
            chunks = self._chunk_text(text)
            for c in chunks:
                chunk_texts.append(c)
                article_indices.append(idx)
        if not chunk_texts:
            return [0.0] * len(articles)
        outputs = self.clf(chunk_texts, batch_size=self.batch_size)
        # Aggregate per article
        per_article_scores: List[List[float]] = [[] for _ in range(len(articles))]
        for art_idx, out in zip(article_indices, outputs):
            if isinstance(out, list):
                per_article_scores[art_idx].append(self._scores_to_scalar(out))
        sentiments: List[float] = []
        for scores in per_article_scores:
            if scores:
                sentiments.append(float(np.mean(scores)))
            else:
                sentiments.append(0.0)
        return sentiments

    def score_articles_multi(self, articles_by_symbol: List[List[str]]) -> List[float]:
        """
        Score multiple symbols' article lists in a single batched pass.
        articles_by_symbol: list where each element is a list of article texts for one symbol.
        Returns a list of sentiment scores, one per symbol.
        """
        if not articles_by_symbol:
            return []
        chunk_texts: List[str] = []
        symbol_indices: List[int] = []
        for sym_idx, articles in enumerate(articles_by_symbol):
            if not articles:
                continue
            for text in articles:
                if not isinstance(text, str) or not text:
                    continue
                for c in self._chunk_text(text):
                    chunk_texts.append(c)
                    symbol_indices.append(sym_idx)
        if not chunk_texts:
            return [0.0] * len(articles_by_symbol)
        outputs = self.clf(chunk_texts, batch_size=self.batch_size)
        per_symbol_scores: List[List[float]] = [[] for _ in range(len(articles_by_symbol))]
        for sym_idx, out in zip(symbol_indices, outputs):
            if isinstance(out, list):
                per_symbol_scores[sym_idx].append(self._scores_to_scalar(out))
        sentiments: List[float] = []
        for scores in per_symbol_scores:
            if scores:
                sentiments.append(float(np.mean(scores)))
            else:
                sentiments.append(0.0)
        return sentiments
