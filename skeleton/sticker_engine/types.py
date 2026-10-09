"""
Контракт данных фичи «стикеры по словам».

ВАЖНО: имена классов, имена полей и их порядок менять НЕЛЬЗЯ — по ним идёт
интеграция в основной пайплайн заказчика. Добавлять новые поля со значением
по умолчанию можно, удалять/переименовывать существующие — нет.

Все координаты — пиксели ВЫХОДНОГО кадра, начало в левом верхнем углу, ось Y вниз.
Всё время — секунды от начала клипа, float.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "WordCue",
    "Rect",
    "ForbiddenZone",
    "EXCLUSIVE_ZONE_KINDS",
    "StickerEvent",
    "PlacedSticker",
    "StickerConfig",
    "Matcher",
    "CatalogEntry",
]


# --------------------------------------------------------------------------- #
#  Вход                                                                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WordCue:
    """Одно распознанное слово с таймингами (клип-локальными)."""
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class Rect:
    """Прямоугольник в пикселях выходного кадра."""
    x: int
    y: int
    w: int
    h: int

    def intersects(self, other: "Rect", padding: int = 0) -> bool:
        """True, если прямоугольники пересекаются (self раздут на padding с каждой стороны)."""
        ax0, ay0 = self.x - padding, self.y - padding
        ax1, ay1 = self.x + self.w + padding, self.y + self.h + padding
        bx0, by0 = other.x, other.y
        bx1, by1 = other.x + other.w, other.y + other.h
        return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@dataclass(frozen=True)
class ForbiddenZone:
    """
    Занятая область кадра в окне времени [t_start, t_end).

    kind:
        'face'      — лицо говорящего (двигается, обычно 2 сэмпла/сек)
        'subtitle'  — блок субтитров и его стек
        'header'    — заголовок сверху
        'broll'     — окно b-roll вставки  -> стикеры В ЭТО ОКНО НЕ СТАВЯТСЯ ВООБЩЕ
        'endscreen' — конечная заставка    -> стикеры В ЭТО ОКНО НЕ СТАВЯТСЯ ВООБЩЕ
        'custom'    — прочее
    """
    t_start: float
    t_end: float
    rect: Rect
    kind: str = "custom"


# Зоны, при пересечении с которыми событие удаляется целиком (см. ТЗ §7.1).
EXCLUSIVE_ZONE_KINDS: frozenset[str] = frozenset({"broll", "endscreen"})


# --------------------------------------------------------------------------- #
#  Каталог                                                                     #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CatalogEntry:
    """Одна запись каталога эмодзи (assets/emoji_catalog.json)."""
    emoji: str
    file: str          # имя PNG внутри assets/stickers/
    category: str
    ru: str            # описание на русском — основное поле для матчинга
    en: str = ""


# --------------------------------------------------------------------------- #
#  Выход                                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StickerEvent:
    """Что показать и когда. Результат работы матчера + планировщика."""
    t_anchor: float
    t_start: float
    t_end: float
    emoji: str
    asset_path: str
    trigger_word: str
    trigger_word_index: int
    phrase: str
    score: float
    category: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class PlacedSticker:
    """Событие + позиция на кадре. Результат работы решателя позиции."""
    event: StickerEvent
    x: int
    y: int
    size: int          # сторона квадрата в пикселях при scale == 1.0

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.size, self.size)


# --------------------------------------------------------------------------- #
#  Конфиг                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class StickerConfig:
    """
    Все настройки фичи. Значения по умолчанию — согласованные с заказчиком дефолты,
    кроме `min_score`: его исполнитель подбирает эмпирически (см. ТЗ §13).
    """
    # --- плотность ---
    max_per_clip: int = 10
    density_per_min: float = 12.0
    min_gap_sec: float = 2.5
    max_concurrent: int = 1

    # --- качество подбора ---
    min_score: float = 0.45
    min_margin_over_runner_up: float = 0.03
    phrase_window: int = 5
    same_emoji_cooldown_sec: float = 15.0
    same_emoji_max_per_clip: int = 2
    edge_padding_head_sec: float = 1.0
    edge_padding_tail_sec: float = 1.5
    blocked_categories: Tuple[str, ...] = (
        "weapons", "violence", "religion", "flags", "adult", "medical", "politics",
    )

    # --- геометрия ---
    size_frac: float = 0.13
    margin_frac: float = 0.04
    zone_padding_frac: float = 0.015

    # --- анимация (секунды) ---
    pop_in_sec: float = 0.15
    hold_sec: float = 1.20
    fade_out_sec: float = 0.20
    pop_overshoot: float = 1.18

    # --- инфраструктура ---
    assets_dir: Optional[Path] = None
    catalog_path: Optional[Path] = None
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: Optional[Path] = None
    seed: int = 0

    @property
    def total_duration(self) -> float:
        """Полная длительность показа одного стикера."""
        return self.pop_in_sec + self.hold_sec + self.fade_out_sec

    def sticker_size(self, output_w: int) -> int:
        return round(self.size_frac * output_w)

    def frame_margin(self, output_w: int) -> int:
        return round(self.margin_frac * output_w)

    def zone_padding(self, output_w: int) -> int:
        return round(self.zone_padding_frac * output_w)

    @classmethod
    def from_json(cls, path: Path) -> "StickerConfig":
        """Загрузка конфига из JSON (для CLI --config). Неизвестные ключи -> WARNING, не падаем."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Seam для подмены матчера в тестах                                           #
# --------------------------------------------------------------------------- #

class Matcher(Protocol):
    """
    Подбор эмодзи по фразе.

    Возвращает (emoji, category, score) либо None, если уверенного соответствия нет.
    score — 0..1. Реализация по умолчанию — sticker_engine.matcher.EmbeddingMatcher.

    ОБА метода обязательны. Планировщик вызывает только match_batch (пофразовые
    вызовы модели слишком медленные); match удобен в отладке и тестах.
    """

    def match(self, phrase: str) -> Optional[Tuple[str, str, float]]:
        ...

    def match_batch(self, phrases: Sequence[str]) -> List[Optional[Tuple[str, str, float]]]:
        ...
