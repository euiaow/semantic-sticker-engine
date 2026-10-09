"""Приёмка §4 — каталог эмодзи и ассеты."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sticker_engine.catalog import ALLOWED_CATEGORIES, EmojiCatalog, load_catalog

MIN_CATALOG_SIZE = 150


def _catalog_path(cfg) -> Path:
    return Path(cfg.catalog_path)


def test_catalog_exists(cfg):
    assert _catalog_path(cfg).exists(), "assets/emoji_catalog.json отсутствует"


def test_catalog_size(cfg):
    entries = load_catalog(_catalog_path(cfg))
    assert len(entries) >= MIN_CATALOG_SIZE, f"нужно >= {MIN_CATALOG_SIZE} эмодзи, есть {len(entries)}"


def test_catalog_is_valid(cfg):
    cat = EmojiCatalog.load(cfg)
    problems = cat.validate()
    assert problems == [], "\n".join(problems[:20])


def test_categories_are_known(cfg):
    for e in load_catalog(_catalog_path(cfg)):
        assert e.category in ALLOWED_CATEGORIES, f"{e.emoji}: неизвестная категория {e.category}"


def test_emoji_unique(cfg):
    entries = load_catalog(_catalog_path(cfg))
    seen = [e.emoji for e in entries]
    dupes = {x for x in seen if seen.count(x) > 1}
    assert not dupes, f"дубли: {dupes}"


def test_ru_descriptions_are_meaningful(cfg):
    for e in load_catalog(_catalog_path(cfg)):
        words = [w for w in e.ru.replace(",", " ").split() if w]
        assert len(words) >= 3, f"{e.emoji}: описание слишком бедное — {e.ru!r}"


def test_blocked_categories_present_in_catalog(cfg):
    """
    Опасные эмодзи должны быть В каталоге и фильтроваться конфигом, а не отсутствовать
    (ТЗ §4.2): минимум по одной записи на каждую запрещённую категорию.
    """
    cats = {e.category for e in load_catalog(_catalog_path(cfg))}
    missing = set(cfg.blocked_categories) - cats
    assert not missing, f"в каталоге нет записей для запрещённых категорий: {sorted(missing)}"


def test_attribution_file(cfg):
    attribution = Path(cfg.assets_dir) / "ATTRIBUTION.md"
    assert attribution.exists(), "нет ATTRIBUTION.md (Twemoji — CC-BY 4.0)"
    assert "CC-BY" in attribution.read_text(encoding="utf-8").upper()


def test_embeddings_meta_matches_model(cfg):
    meta = Path(cfg.assets_dir).parent / "catalog_embeddings.meta.json"
    if not meta.exists():
        pytest.skip("эмбеддинги каталога не собраны")
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["model_name"] == cfg.model_name
    assert int(data["count"]) >= MIN_CATALOG_SIZE


def test_missing_catalog_is_survived(tmp_path):
    """Нет каталога — пустой каталог и WARNING, не исключение."""
    from sticker_engine.types import StickerConfig

    cfg = StickerConfig(assets_dir=tmp_path, catalog_path=tmp_path / "nope.json")
    cat = EmojiCatalog.load(cfg)
    assert cat.entries == []
