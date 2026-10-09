"""
Замер эффекта лемматизации на eval-наборе (ТЗ §15.2).
Запуск из skeleton/: python scripts/tune_lemmatize.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pymorphy3
from sentence_transformers import SentenceTransformer

from sticker_engine.catalog import EmojiCatalog
from sticker_engine.types import StickerConfig

EVAL_PATH = ROOT.parent / "fixtures" / "eval_set.json"
ASSETS = ROOT / "assets"

_morph = pymorphy3.MorphAnalyzer()
_word_re = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


def lemmatize(text: str) -> str:
    return " ".join(_morph.parse(w)[0].normal_form for w in _word_re.findall(text))


def _ok(expected, got: str) -> bool:
    if isinstance(expected, list):
        return got in expected
    return got == expected


def run(use_lemmas: bool, min_score: float, margin: float):
    cfg = StickerConfig(assets_dir=ASSETS / "stickers",
                        catalog_path=ASSETS / "emoji_catalog.json")
    catalog = EmojiCatalog.load(cfg)
    model = SentenceTransformer(cfg.model_name)

    catalog_texts = [e.ru for e in catalog.entries]
    if use_lemmas:
        catalog_texts = [lemmatize(t) for t in catalog_texts]
    catalog_emb = model.encode(catalog_texts, normalize_embeddings=True)

    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    phrases = [d["phrase"] for d in data]
    if use_lemmas:
        phrases = [lemmatize(p) for p in phrases]
    phrase_emb = model.encode(phrases, normalize_embeddings=True)

    sims = phrase_emb @ catalog_emb.T
    tp = fp = fn = 0

    for item, row in zip(data, sims):
        top = np.argsort(row)[::-1][:2]
        score = float(row[top[0]])
        entry = catalog.entries[top[0]]
        gap = score - float(row[top[1]]) if len(top) > 1 else 1.0

        got = None
        if (score >= min_score
                and entry.category not in cfg.blocked_categories
                and gap >= margin):
            got = entry.emoji

        expected = item.get("expect")
        if got and expected and _ok(expected, got):
            tp += 1
        elif got:
            fp += 1
        elif expected:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


if __name__ == "__main__":
    print(f"{'режим':>12} {'min_score':>10} {'margin':>7} {'precision':>10} {'recall':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for ms, mg in [(0.45, 0.03), (0.45, 0.05), (0.45, 0.08), (0.45, 0.12), (0.50, 0.08), (0.45, 0.15), (0.45, 0.18), (0.45, 0.22)]:
        for use_lemmas in (False, True):
            p, r, tp, fp, fn = run(use_lemmas, ms, mg)
            label = "лемматизац." if use_lemmas else "как есть"
            print(f"{label:>12} {ms:>10.2f} {mg:>7.2f} {p:>10.3f} {r:>8.3f} {tp:>4} {fp:>4} {fn:>4}")