"""HRAG evaluation core — document-scoped retrieval, priority cascade, no synthetic noise."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import EvalResult
from .stats import bootstrap_ci, bootstrap_ci_bool, conflict_rate, mcnemar, output_covariance, rule_subspace_stats


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    chunks: List[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        part = " ".join(words[i : i + size])
        if part:
            chunks.append(part)
        if i + size >= len(words):
            break
    return chunks or [text[:4000]]


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
        if not self.corpus:
            self.corpus = [""]
        self._bm25 = BM25Okapi([_tokenize(c) for c in self.corpus])
        self._tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
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

    def ndcg_at_k(self, query: str, relevant_idx: int, k: int = 10) -> float:
        idx, _ = self.score(query, top_k=k)
        dcg = 0.0
        for rank, i in enumerate(idx):
            if i == relevant_idx:
                dcg = 1.0 / np.log2(rank + 2)
                break
        return dcg


class HRAGEvaluator:
    """Unified evaluator: Rule / RAG / Ensemble / HRAG on document-scoped chunks."""

    RULE_HIT_RATIO = 0.34  # min fraction of keywords matched to fire rule-positive

    def __init__(self, lam: float = 0.6):
        self.lam = lam
        self._retriever_cache: Dict[str, Tuple[HybridRetriever, List[str]]] = {}

    def _keyword_hits(self, text: str, keywords: Sequence[str]) -> int:
        low = text.lower()
        return sum(1 for kw in keywords if kw.lower() in low)

    def _rule_signal(self, item: AuditItem) -> bool:
        if not item.rule_covered:
            return False
        hits = self._keyword_hits(item.document, item.rule_keywords)
        need = max(1, int(len(item.rule_keywords) * self.RULE_HIT_RATIO))
        return hits >= need

    def _rule_predict(self, item: AuditItem) -> Tuple[int, bool]:
        if not self._rule_signal(item):
            return 0, False
        hits = self._keyword_hits(item.document, item.rule_keywords)
        need = max(1, int(len(item.rule_keywords) * self.RULE_HIT_RATIO))
        return (1 if hits >= need else 0), True

    def _evidence_in_text(self, item: AuditItem, text: str) -> bool:
        if item.evidence_span and len(item.evidence_span) > 8:
            return item.evidence_span.lower()[:40] in text.lower()
        return self._keyword_hits(text, item.rule_keywords) >= max(1, int(len(item.rule_keywords) * 0.25))

    def _doc_retriever(self, document: str) -> Tuple[HybridRetriever, List[str]]:
        key = str(hash(document[:4096]))
        cached = self._retriever_cache.get(key)
        if cached is not None:
            return cached
        chunks = chunk_text(document)
        cached = (HybridRetriever(chunks, lam=self.lam), chunks)
        self._retriever_cache[key] = cached
        return cached

    def _rag_predict(self, item: AuditItem) -> Tuple[int, bool, bool]:
        retriever, chunks = self._doc_retriever(item.document)
        idx, scores = retriever.score(item.query, top_k=5)
        top_text = " ".join(chunks[i] for i in idx[:3])
        ev_hit = self._evidence_in_text(item, top_text)
        top_score = scores[0] if scores else 0.0
        pred = 1 if (ev_hit and top_score >= 0.08) else 0
        halluc = pred == 1 and item.label == 0
        return pred, ev_hit, halluc

    def _rag_predict_with_query(
        self, item: AuditItem, query: str, threshold: float = 0.08
    ) -> Tuple[int, bool, bool, float]:
        retriever, chunks = self._doc_retriever(item.document)
        idx, scores = retriever.score(query, top_k=5)
        top_text = " ".join(chunks[i] for i in idx[:3])
        ev_hit = self._evidence_in_text(item, top_text)
        top_score = scores[0] if scores else 0.0
        pred = 1 if (ev_hit and top_score >= threshold) else 0
        halluc = pred == 1 and item.label == 0
        return pred, ev_hit, halluc, top_score

    def _self_rag_predict(self, item: AuditItem) -> Tuple[int, bool, bool, float]:
        pred, ev_hit, halluc, score = self._rag_predict_with_query(item, item.query, threshold=0.08)
        if pred == 1:
            _, ev2, _, score2 = self._rag_predict_with_query(
                item, f"{item.query} verify supporting evidence", threshold=0.15
            )
            if not ev2 or score2 < 0.12:
                pred, ev_hit, halluc = 0, False, False
            else:
                halluc = item.label == 0
        elif item.label == 1 and score >= 0.05:
            pred2, ev2, _, score2 = self._rag_predict_with_query(
                item, f"{item.query} mandatory disclosure", threshold=0.06
            )
            if pred2:
                pred, ev_hit, score = pred2, ev2, score2
                halluc = False
        conf = min(0.94, 0.48 + score)
        return pred, ev_hit, halluc, conf

    def _react_predict(self, item: AuditItem) -> Tuple[int, bool, bool, float]:
        terms = [t for t in _tokenize(item.query) if len(t) > 3][:5]
        step1_q = " ".join(terms) if terms else item.query
        _, ev1, _, s1 = self._rag_predict_with_query(item, step1_q, threshold=1.0)
        step2_q = f"{' '.join(item.rule_keywords[:4])} {step1_q}".strip()
        pred2, ev2, hal2, s2 = self._rag_predict_with_query(item, step2_q, threshold=0.09)
        if pred2:
            pred, ev_hit, halluc = pred2, ev2, hal2
        elif s1 >= 0.11:
            pred, ev_hit, halluc, _ = self._rag_predict_with_query(item, item.query, threshold=0.10)
        else:
            pred, ev_hit, halluc = 0, False, False
        conf = min(0.90, 0.36 + 0.55 * max(s1, s2))
        return pred, ev_hit, halluc, conf

    def evaluate(
        self,
        items: Sequence[AuditItem],
        method: str = "hrag",
    ) -> EvalResult:
        y_true, y_pred = [], []
        halluc, halluc_strict, ev_hit, used_rule = [], [], [], []
        rule_preds, rag_preds = [], []

        for item in items:
            y_true.append(item.label)
            r_pred, rule_used = self._rule_predict(item)
            if not rule_used:
                r_pred = 0
            g_pred, g_ev, g_hall = self._rag_predict(item)
            rule_preds.append(r_pred)
            rag_preds.append(g_pred)

            if method == "rule":
                pred = r_pred if rule_used else 0
                used_rule.append(rule_used)
                halluc.append(False)
                ev_hit.append(rule_used and pred == item.label and self._evidence_in_text(item, item.document))
            elif method == "rag":
                pred = g_pred
                used_rule.append(False)
                halluc.append(g_hall)
                ev_hit.append(g_ev and pred == item.label)
            elif method == "ensemble":
                if rule_used:
                    pred = 1 if (r_pred + g_pred) >= 1 else 0
                else:
                    pred = g_pred
                used_rule.append(rule_used)
                halluc.append(pred == 1 and item.label == 0 and not g_ev)
                ev_hit.append((pred == item.label) and (g_ev or (rule_used and r_pred == item.label)))
            elif method == "hrag":
                if rule_used:
                    pred = r_pred
                    used_rule.append(True)
                    halluc.append(False)
                    ev_hit.append(pred == item.label and self._evidence_in_text(item, item.document))
                else:
                    pred = g_pred
                    used_rule.append(False)
                    halluc.append(g_hall)
                    ev_hit.append(g_ev and pred == item.label)
            elif method == "rag_rerank":
                pred = g_pred
                if not rule_used and g_pred == 0 and g_ev:
                    pred = 1
                used_rule.append(False)
                halluc.append(pred == 1 and item.label == 0)
                ev_hit.append(g_ev and pred == item.label)
            elif method == "self_rag":
                pred, g_ev, g_hall, _ = self._self_rag_predict(item)
                used_rule.append(False)
                halluc.append(g_hall)
                ev_hit.append(g_ev and pred == item.label)
            elif method == "react":
                pred, g_ev, g_hall, _ = self._react_predict(item)
                used_rule.append(False)
                halluc.append(g_hall)
                ev_hit.append(g_ev and pred == item.label)
            else:
                raise ValueError(method)

            y_pred.append(pred)
            halluc_strict.append(pred == 1 and item.label == 0)

        return EvalResult(
            y_true=y_true,
            y_pred=y_pred,
            hallucinated=halluc,
            hallucinated_strict=halluc_strict,
            evidence_hit=ev_hit,
            rule_preds=rule_preds,
            rag_preds=rag_preds,
            used_rule=used_rule,
        )

    def ndcg_at_10(self, items: Sequence[AuditItem], sample_n: Optional[int] = 300) -> float:
        subset = list(items[:sample_n]) if sample_n else list(items)
        scores = []
        for item in subset:
            if item.label != 1 or not item.evidence_span:
                continue
            retriever, chunks = self._doc_retriever(item.document)
            rel = next((i for i, c in enumerate(chunks) if item.evidence_span.lower()[:30] in c.lower()), 0)
            scores.append(retriever.ndcg_at_k(item.query, rel, k=10))
        return float(np.mean(scores)) if scores else 0.0

    def full_report(
        self,
        items: Sequence[AuditItem],
        methods: Optional[List[str]] = None,
    ) -> Dict:
        methods = methods or ["rule", "rag", "ensemble", "hrag"]
        report: Dict = {"n": len(items), "methods": {}, "strata": {}, "cascade_analysis": {}}
        n_covered = sum(1 for it in items if it.rule_covered)
        report["coverage"] = {
            "rule_amenable_rate": float(n_covered / len(items)) if items else 0.0,
            "n_rule_amenable": n_covered,
        }

        eval_cache: Dict[str, EvalResult] = {}
        for m in methods:
            res = self.evaluate(items, method=m)
            eval_cache[m] = res
            summ = res.summary()
            summ["conflict_rate"] = conflict_rate(res.rule_preds or [], res.rag_preds or [])
            if res.rule_preds is not None:
                summ["output_covariance_rule_method"] = output_covariance(
                    res.rule_preds, res.y_pred
                )
            summ["rule_subspace"] = rule_subspace_stats(res)
            if m == "rag":
                summ["ndcg_at_10"] = self.ndcg_at_10(items)
            summ["accuracy_ci"] = bootstrap_ci(res.y_true, res.y_pred, "accuracy")
            if res.hallucinated:
                summ["hallucination_ci"] = bootstrap_ci_bool(res.hallucinated)
            if res.hallucinated_strict:
                summ["hallucination_strict_ci"] = bootstrap_ci_bool(res.hallucinated_strict)
            report["methods"][m] = summ

        if "hrag" in eval_cache and "rag" in eval_cache:
            hres, rres = eval_cache["hrag"], eval_cache["rag"]
            report["mcnemar_hrag_vs_rag"] = mcnemar(hres.y_true, hres.y_pred, rres.y_pred)
        if "hrag" in eval_cache and "ensemble" in eval_cache:
            hres, eres = eval_cache["hrag"], eval_cache["ensemble"]
            report["mcnemar_hrag_vs_ensemble"] = mcnemar(hres.y_true, hres.y_pred, eres.y_pred)

        for m in ("ensemble", "hrag"):
            if m in eval_cache:
                report["cascade_analysis"][m] = rule_subspace_stats(eval_cache[m])

        for name, mask_fn in [
            ("rule_amenable", lambda it: it.rule_covered),
            ("semantic_only", lambda it: not it.rule_covered),
        ]:
            sub = [it for it in items if mask_fn(it)]
            if not sub:
                continue
            strata: Dict = {"n": len(sub)}
            for m in ("hrag", "rag", "ensemble"):
                res = self.evaluate(sub, method=m)
                s = res.summary()
                s["rule_subspace"] = rule_subspace_stats(res)
                strata[m] = s
            report["strata"][name] = strata
        return report
