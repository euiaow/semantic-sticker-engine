"""
Приёмка §5 / §13 — качество подбора на размеченном eval-наборе.

Требует собранных эмбеддингов каталога и локально закэшированной модели,
поэтому помечен requires_model и пропускается на голом чекауте.

Eval-набор — fixtures/eval_set.json, формат (ТЗ §13.1):
    [{"phrase": "мы заработали первый миллион за три месяца", "expect": "💰"},
     {"phrase": "мы попали ровно в свою целевую аудиторию", "expect": ["🎯", "👥"]},
     {"phrase": "в общем, я хочу сказать следующее",          "expect": null}]

`expect` — строка, список допустимых эмодзи или null (стикер не нужен).
Заготовка на 29 фраз лежит в fixtures/eval_set.sample.json.
Исполнитель обязан довести набор до >= 120 фраз, из них >= 40 негативных.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FIXTURES, requires_model

MIN_EVAL_SIZE = 120
MIN_NEGATIVES = 40
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.35

EVAL_PATH = FIXTURES / "eval_set.json"


def _load_eval():
    if not EVAL_PATH.exists():
        pytest.skip("fixtures/eval_set.json не создан (см. ТЗ §13)")
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def test_eval_set_size():
    data = _load_eval()
    assert len(data) >= MIN_EVAL_SIZE
    negatives = [d for d in data if not d.get("expect")]
    assert len(negatives) >= MIN_NEGATIVES


@requires_model
def test_precision_and_recall():
    from sticker_engine.catalog import EmojiCatalog
    from sticker_engine.matcher import EmbeddingMatcher
    from sticker_engine.types import StickerConfig

    data = _load_eval()
    cfg = StickerConfig(
        assets_dir=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "stickers",
        catalog_path=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "emoji_catalog.json",
    )
    matcher = EmbeddingMatcher(EmojiCatalog.load(cfg), cfg)
    results = matcher.match_batch([d["phrase"] for d in data])

    def _ok(expected, got: str) -> bool:
        if isinstance(expected, list):
            return got in expected
        return got == expected

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
    report = (f"precision={precision:.3f} recall={recall:.3f} "
              f"tp={tp} fp={fp} fn={fn}\nпримеры ошибок: {misses[:10]}")

    assert precision >= TARGET_PRECISION, report
    assert recall >= TARGET_RECALL, report


@requires_model
def test_negatives_produce_nothing():
    """Вводные конструкции не должны давать стикер ни при каком раскладе."""
    from sticker_engine.catalog import EmojiCatalog
    from sticker_engine.matcher import EmbeddingMatcher
    from sticker_engine.types import StickerConfig

    hard_negatives = [
        "в общем, я хочу сказать следующее",
        "то есть получается вот такая история",
        "мы с вами сегодня поговорим об этом",
        "как бы это правильно сформулировать",
        "и вот здесь есть один нюанс",
        "давайте по порядку разберём",
    ]
    cfg = StickerConfig(
        assets_dir=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "stickers",
        catalog_path=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "emoji_catalog.json",
    )
    matcher = EmbeddingMatcher(EmojiCatalog.load(cfg), cfg)
    got = matcher.match_batch(hard_negatives)
    fired = [(p, g) for p, g in zip(hard_negatives, got) if g]
    assert not fired, f"ложные срабатывания на связках: {fired}"


@requires_model
def test_blocked_category_never_returned():
    from sticker_engine.catalog import EmojiCatalog
    from sticker_engine.matcher import EmbeddingMatcher
    from sticker_engine.types import StickerConfig

    cfg = StickerConfig(
        assets_dir=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "stickers",
        catalog_path=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "emoji_catalog.json",
    )
    matcher = EmbeddingMatcher(EmojiCatalog.load(cfg), cfg)
    phrases = ["пришлось убить время в аэропорту",
               "это была настоящая война за клиента",
               "я молился чтобы сделка прошла"]
    for res in matcher.match_batch(phrases):
        if res:
            assert res[1] not in cfg.blocked_categories, res


@requires_model
def test_model_loads_once(monkeypatch):
    """Повторное создание матчера не должно перегружать веса модели."""
    import time

    from sticker_engine.catalog import EmojiCatalog
    from sticker_engine.matcher import EmbeddingMatcher
    from sticker_engine.types import StickerConfig

    cfg = StickerConfig(
        assets_dir=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "stickers",
        catalog_path=Path(__file__).resolve().parent.parent / "skeleton" / "assets" / "emoji_catalog.json",
    )
    cat = EmojiCatalog.load(cfg)
    EmbeddingMatcher(cat, cfg).match("первый миллион")
    t0 = time.perf_counter()
    EmbeddingMatcher(cat, cfg).match("первый миллион")
    assert time.perf_counter() - t0 < 1.0, "модель перезагружается на каждый матчер"
