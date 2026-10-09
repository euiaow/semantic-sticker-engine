"""
Каталог эмодзи: загрузка assets/emoji_catalog.json, валидация, пути к PNG,
предрасчитанные эмбеддинги (assets/catalog_embeddings.npy).
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence
import json
import numpy as np
from PIL import Image
from .types import CatalogEntry, StickerConfig

logger = logging.getLogger("sticker_engine.catalog")

__all__ = ["EmojiCatalog", "ALLOWED_CATEGORIES", "load_catalog"]

# Закрытый список категорий. Расширять можно, но согласованно: имена из
# StickerConfig.blocked_categories должны существовать здесь.
ALLOWED_CATEGORIES: tuple[str, ...] = (
    "money", "growth", "hype", "emotion_positive", "emotion_negative", "surprise",
    "think", "time", "work", "tech", "food", "travel", "sport", "nature", "people",
    "communication", "warning", "success", "failure", "idea", "learning", "family",
    "weapons", "violence", "religion", "flags", "adult", "medical", "politics",
)


class EmojiCatalog:
    """
    Загруженный каталог + матрица эмбеддингов описаний.

    Матрица берётся из assets/catalog_embeddings.npy, если он есть и его мета
    совпадает с config.model_name. Иначе считается на лету (медленно) с WARNING.
    """

    entries: List[CatalogEntry]
    embeddings: Optional[np.ndarray]   # (N, D), L2-нормированные, порядок == entries

    def __init__(self, entries: Sequence[CatalogEntry], assets_dir: Path,
                 embeddings: Optional[np.ndarray] = None) -> None:
        self.entries = entries
        self.embeddings = embeddings
        self.assets_dir = assets_dir

    @classmethod
    def load(cls, config: StickerConfig) -> "EmojiCatalog":
        root = Path(__file__).resolve().parent.parent   # skeleton/
        assets_dir = config.assets_dir or (root / "assets" / "stickers")
        catalog_path = config.catalog_path or (root / "assets" / "emoji_catalog.json")

        try:
            entries = load_catalog(catalog_path)
        except Exception as e:
            logger.warning("не удалось загрузить каталог из %s: %s", catalog_path, e)
            entries = []

        embeddings = None
        embeddings_dir = Path(assets_dir).parent
        npy_path = embeddings_dir / "catalog_embeddings.npy"
        meta_path = embeddings_dir / "catalog_embeddings.meta.json"

        if npy_path.exists() and meta_path.exists() and entries:
            try:
                current_hash = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                if meta.get("model_name") == config.model_name and meta.get("catalog_sha256") == current_hash:
                    arr = np.load(npy_path)
                    if arr.shape[0] == len(entries):
                        embeddings = arr
                    else:
                        logger.warning("эмбеддинги не совпадают по размеру с каталогом (%d vs %d)",
                                        arr.shape[0], len(entries))
                else:
                    logger.warning("эмбеддинги устарели (модель или каталог изменились) — пересчитываю")
            except Exception as e:
                logger.warning("не удалось прочитать эмбеддинги каталога: %s", e)

        if embeddings is None and entries:
            logger.warning("эмбеддинги каталога не найдены/устарели — считаю на лету (медленно)")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.model_name, cache_folder=config.model_cache_dir)
            texts = [e.ru for e in entries]
            embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

        return cls(entries=entries, assets_dir=assets_dir, embeddings=embeddings)


    def asset_path(self, entry: CatalogEntry) -> Path:
        """Абсолютный путь к PNG записи."""
        return self.assets_dir / entry.file

    def validate(self) -> List[str]:
        problems: List[str] = []

        for entry in self.entries:
            png_path = self.asset_path(entry)
            try:
                img = Image.open(png_path)
                width, height = img.size
                if width != height:
                    problems.append(f"{entry.emoji}: PNG не квадратный ({width}x{height})")
                if width < 256:
                    problems.append(f"{entry.emoji}: PNG меньше 256px ({width}x{height})")
                if "A" not in img.mode:
                    problems.append(f"{entry.emoji}: PNG без альфа-канала (mode={img.mode})")
            except Exception as e:
                problems.append(f"{entry.emoji}: PNG не читается ({e})")

            if entry.category not in ALLOWED_CATEGORIES:
                problems.append(f"{entry.emoji}: неизвестная категория {entry.category}")

            if len(entry.ru.replace(",", " ").split()) < 3:
                problems.append(f"{entry.emoji}: описание ru слишком бедное — {entry.ru!r}")

        seen = set()
        for entry in self.entries:
            if entry.emoji in seen:
                problems.append(f"{entry.emoji}: дубликат emoji в каталоге")
            else:
                seen.add(entry.emoji)

        return problems


def load_catalog(path: Path) -> List[CatalogEntry]:
    """Прочитать и распарсить emoji_catalog.json. Бросает ValueError на битом JSON."""
    raw_text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"битый JSON в {path}: {e}") from e

    entries = []
    for item in data:
        entries.append(CatalogEntry(
            emoji=item["emoji"],
            file=item["file"],
            category=item["category"],
            ru=item["ru"],
            en=item.get("en", ""),
        ))
    return entries
