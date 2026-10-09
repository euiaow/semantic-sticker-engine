"""
Подбор min_score по eval_set.json (ТЗ §13).
Запуск: python scripts/tune_threshold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent  # skeleton/
sys.path.insert(0, str(ROOT))

from sticker_engine.catalog import EmojiCatalog
from sticker_engine.matcher import EmbeddingMatcher
from sticker_engine.types import StickerConfig

EVAL_PATH = ROOT.parent / "fixtures" / "eval_set.json"   # <- .parent здесь обязателен
ASSETS = ROOT / "assets"


def _ok(expected, got: str) -> bool:
    if isinstance(expected, list):
        return got in expected
    return got == expected


def evaluate(min_score: float, margin: float = 0.03):
    cfg = StickerConfig(
        assets_dir=ASSETS / "stickers",
        catalog_path=ASSETS / "emoji_catalog.json",
        min_score=min_score,
        min_margin_over_runner_up=margin,
    )
    catalog = EmojiCatalog.load(cfg)
    matcher = EmbeddingMatcher(catalog, cfg)

    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = matcher.match_batch([d["phrase"] for d in data])

    tp = fp = fn = 0
    misses = []
    for item, res in zip(data, results):
        expected = item.get("expect")
        got = res[0] if res else None
        if got and expected and _ok(expected, got):
            tp += 1
        elif got:
            fp += 1
            misses.append((item["phrase"], expected, got))
        elif expected:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn, misses


if __name__ == "__main__":
    print(f"{'min_score':>10} {'precision':>10} {'recall':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for ms in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        p, r, tp, fp, fn, _ = evaluate(ms)
        print(f"{ms:>10.2f} {p:>10.3f} {r:>8.3f} {tp:>4} {fp:>4} {fn:>4}")

    print(f"\n{'min_score':>10} {'margin':>8} {'precision':>10} {'recall':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for ms in [0.40, 0.45, 0.50]:
        for mg in [0.03, 0.05, 0.08, 0.12, 0.15, 0.20]:
            p, r, tp, fp, fn, _ = evaluate(ms, mg)
            print(f"{ms:>10.2f} {mg:>8.2f} {p:>10.3f} {r:>8.3f} {tp:>4} {fp:>4} {fn:>4}")


    p, r, tp, fp, fn, misses = evaluate(0.45)
    negatives = [m for m in misses if m[1] is None]

    print("\n--- какие эмодзи чаще всего ложно срабатывают на негативах ---")
    counter = Counter(got for _, _, got in negatives)

    cfg = StickerConfig(assets_dir=ASSETS / "stickers",
                        catalog_path=ASSETS / "emoji_catalog.json")
    catalog = EmojiCatalog.load(cfg)
    by_emoji = {e.emoji: e for e in catalog.entries}

    for emoji, count in counter.most_common(10):
        entry = by_emoji.get(emoji)
        ru = entry.ru if entry else "???"
        print(f"  {emoji} x{count}  [{entry.category if entry else '?'}]  {ru}")