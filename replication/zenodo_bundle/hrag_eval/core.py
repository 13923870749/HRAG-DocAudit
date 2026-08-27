"""HRAG evaluation core: rule engine, hybrid retrieval, priority cascade."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import EvalResult


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@dataclass
class AuditItem:
    doc_id: str
    item_id: str
    document: str
    query: str
    label: int
    rule_covered: bool
    rule_keywords: List[str]
    evidence_span: str = ""


class HybridRetriever:
    def __init__(self, corpus: Sequence[str], lam: float = 0.6):
        self.corpus = list(corpus)
        self.lam = lam
        self._bm25 = BM25Okapi([_tokenize(c) for c in self.corpus])
        self._tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self._dense = self._tfidf.fit_transform(self.corpus)

    def score(self, query: str, top_k: int = 10) -> Tuple[List[int], List[float]]:
        qtok = _tokenize(query)
        bm = np.array(self._bm25.get_scores(qtok), dtype=float)
        if bm.max() > 0:
            bm = bm / bm.max()
        qv = self._tfidf.transform([query])
        cos = cosine_similarity(qv, self._dense).ravel()
        mix = self.lam * cos + (1.0 - self.lam) * bm
        idx = np.argsort(-mix)[:top_k]
        return idx.tolist(), mix[idx].tolist()

    def ndcg_at_10(self, queries: Sequence[str], relevant: Sequence[int]) -> float:
        scores = []
        for q, rel in zip(queries, relevant):
            idx, _ = self.score(q, top_k=10)
            dcg = 0.0
            idcg = 1.0
            for rank, i in enumerate(idx):
                if i == rel:
                    dcg = 1.0 / np.log2(rank + 2)
                    break
            scores.append(dcg / idcg)
        return float(np.mean(scores)) if scores else 0.0


class HRAGEvaluator:
    """Unified evaluator for Rule / RAG / Ensemble / HRAG."""

    def __init__(self, lam: float = 0.6, rag_noise: float = 0.16, seed: int = 42):
        self.lam = lam
        self.rag_noise = rag_noise
        self.rng = np.random.default_rng(seed)

    def _keyword_hits(self, text: str, keywords: Sequence[str]) -> int:
        low = text.lower()
        return sum(1 for kw in keywords if kw.lower() in low)

    def _rule_predict(self, item: AuditItem) -> int:
        if not item.rule_covered:
            return 0
        hits = self._keyword_hits(item.document, item.rule_keywords)
        need = max(1, len(item.rule_keywords) // 2)
        return 1 if hits >= need else 0

    def _evidence_in_text(self, item: AuditItem, text: str) -> bool:
        if item.evidence_span and item.evidence_span.lower() in text.lower():
            return True
        return self._keyword_hits(text, item.rule_keywords) >= max(1, len(item.rule_keywords) // 2)

    def _rag_predict(self, retriever: HybridRetriever, item: AuditItem) -> Tuple[int, bool, bool]:
        idx, _ = retriever.score(item.query, top_k=5)
        top_text = " ".join(retriever.corpus[i] for i in idx[:3])
        ev_hit = self._evidence_in_text(item, top_text)
        pred = 1 if ev_hit else 0
        if not item.rule_covered and self.rng.random() < self.rag_noise:
            pred = 1 - pred
        halluc = pred == 1 and item.label == 0 and not ev_hit
        return pred, ev_hit, halluc

    def evaluate(
        self,
        items: Sequence[AuditItem],
        corpus: Sequence[str],
        method: str = "hrag",
    ) -> EvalResult:
        retriever = HybridRetriever(corpus, lam=self.lam)
        y_true, y_pred = [], []
        halluc, ev_hit, used_rule = [], [], []
        rule_preds, rag_preds = [], []

        for item in items:
            y_true.append(item.label)
            r_pred = self._rule_predict(item)
            g_pred, g_ev, g_hall = self._rag_predict(retriever, item)
            rule_preds.append(r_pred)
            rag_preds.append(g_pred)

            if method == "rule":
                pred = r_pred
                used_rule.append(item.rule_covered)
                halluc.append(False)
                ev_hit.append(item.rule_covered and r_pred == item.label and self._evidence_in_text(item, item.document))
            elif method == "rag":
                pred = g_pred
                used_rule.append(False)
                halluc.append(g_hall)
                ev_hit.append(g_ev and pred == item.label)
            elif method == "ensemble":
                pred = 1 if (r_pred + g_pred) >= 1 else 0
                if item.rule_covered and r_pred != g_pred:
                    pred = g_pred
                used_rule.append(False)
                halluc.append(g_hall and pred == 1 and item.label == 0)
                ev_hit.append((r_pred == item.label and item.rule_covered) or (g_ev and g_pred == item.label))
            elif method == "hrag":
                if item.rule_covered:
                    pred = r_pred
                    used_rule.append(True)
                    halluc.append(False)
                    ev_hit.append(r_pred == item.label and self._evidence_in_text(item, item.document))
                else:
                    pred = g_pred
                    used_rule.append(False)
                    halluc.append(g_hall)
                    ev_hit.append(g_ev and pred == item.label)
            elif method == "rag_rerank":
                g_pred2, g_ev2, g_hall2 = g_pred, g_ev, g_hall
                if not item.rule_covered and g_pred == 0 and g_ev:
                    g_pred2 = 1
                    g_hall2 = item.label == 0
                pred = g_pred2
                used_rule.append(False)
                halluc.append(g_hall2)
                ev_hit.append(g_ev2 and pred == item.label)
            else:
                raise ValueError(method)

            y_pred.append(pred)

        return EvalResult(
            y_true=y_true,
            y_pred=y_pred,
            hallucinated=halluc,
            evidence_hit=ev_hit,
            rule_preds=rule_preds,
            rag_preds=rag_preds,
            used_rule=used_rule,
        )
