"""
sticker_engine — эмодзи-стикеры, привязанные к произнесённым словам.

Публичный API (то, что вызывает пайплайн заказчика):

    from sticker_engine import plan_stickers, StickerRenderer, StickerConfig

    placed = plan_stickers(
        word_cues,
        duration_sec=42.0,
        output_w=1080,
        output_h=1920,
        forbidden_zones=zones,
        config=StickerConfig(),
    )
    renderer = StickerRenderer(placed, 1080, 1920, config)
    for frame_idx in range(n_frames):
        for rgba, x, y in renderer.get_overlays(frame_idx / fps):
            ...  # альфа-блендинг делает вызывающая сторона

Гарантии:
  * plan_stickers детерминирован: одинаковый вход -> одинаковый выход;
  * ни одна функция модуля не бросает исключение наружу при проблемах с
    ассетами/моделью — вместо этого WARNING в лог и пустой результат;
  * никакого сетевого доступа в рантайме.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .types import (
    CatalogEntry,
    ForbiddenZone,
    Matcher,
    PlacedSticker,
    Rect,
    StickerConfig,
    StickerEvent,
    WordCue,
)

__all__ = [
    "plan_stickers",
    "StickerRenderer",
    "StickerConfig",
    "WordCue",
    "Rect",
    "ForbiddenZone",
    "StickerEvent",
    "PlacedSticker",
    "CatalogEntry",
    "Matcher",
    "__version__",
]

__version__ = "0.1.0"

logger = logging.getLogger("sticker_engine")
logger.addHandler(logging.NullHandler())


def plan_stickers(
    word_cues: Sequence[WordCue],
    *,
    duration_sec: float,
    output_w: int,
    output_h: int,
    forbidden_zones: Sequence[ForbiddenZone] = (),
    config: Optional[StickerConfig] = None,
    matcher: Optional[Matcher] = None,
) -> List[PlacedSticker]:
    """
    Полный план стикеров для одного клипа: подбор -> отбор моментов -> размещение.

    Вызывается ОДИН РАЗ на клип. Тяжёлые операции (загрузка модели, чтение
    каталога) допустимы здесь и запрещены в StickerRenderer.get_overlays.

    Порядок конвейера фиксирован (ТЗ §6):
        build_phrases -> matcher.match_batch -> plan_events
        -> drop_events_in_exclusive_zones -> place_events

    Если matcher передан извне, ML-модель грузить ЗАПРЕЩЕНО (не уложиться в
    лимит 2 c из §11), каталог грузить НУЖНО — из него берётся asset_path.

    Args:
        word_cues: пословный транскрипт клипа, тайминги клип-локальные, по возрастанию.
        duration_sec: длительность клипа.
        output_w, output_h: размер выходного кадра.
        forbidden_zones: занятые области кадра (лицо/субтитры/хедер/b-roll/эндскрин).
        config: настройки; None -> StickerConfig().
        matcher: подмена матчера (тесты); None -> EmbeddingMatcher по конфигу.

    Returns:
        Список PlacedSticker, отсортированный по event.t_start. Пустой список —
        штатный, допустимый результат (см. ТЗ §5.4).

    Никогда не бросает исключений: любая внутренняя ошибка -> WARNING + [].
    """
    if config is None:
        config = StickerConfig()
    try:
        from .catalog import EmojiCatalog
        from .planner import drop_events_in_exclusive_zones, plan_events
        from .placer import place_events

        catalog = EmojiCatalog.load(config)

        if matcher is None:
            from .matcher import EmbeddingMatcher
            matcher = EmbeddingMatcher(catalog, config)

        def asset_path_for(emoji: str):
            for entry in catalog.entries:
                if entry.emoji == emoji:
                    return str(catalog.asset_path(entry))
            return None

        events = plan_events(word_cues, duration_sec=duration_sec, matcher=matcher,
                             config=config, asset_path_for=asset_path_for)
        events = drop_events_in_exclusive_zones(events, forbidden_zones)
        return place_events(events, output_w=output_w, output_h=output_h,
                            forbidden_zones=forbidden_zones, config=config)
    except Exception as e:
        logger.warning("plan_stickers упал: %s — возвращаю пустой план", e)
        return []


from .renderer import StickerRenderer  # noqa: E402  (после plan_stickers ради читаемости)
