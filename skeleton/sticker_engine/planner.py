"""
Планировщик моментов: из кандидатов матчера — итоговый набор StickerEvent
с учётом плотности, интервалов, повторов и краёв клипа (ТЗ §6).

Чистая логика, без numpy/PIL/ffmpeg. Тестируется со стаб-матчером.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

from .types import ForbiddenZone, Matcher, StickerConfig, StickerEvent, WordCue
from .types import EXCLUSIVE_ZONE_KINDS

logger = logging.getLogger("sticker_engine.planner")

__all__ = ["plan_events", "max_events_for_duration"]


def max_events_for_duration(duration_sec: float, config: StickerConfig) -> int:
    """min(config.max_per_clip, ceil(duration_sec / 60 * config.density_per_min))."""
    return min(config.max_per_clip, math.ceil(duration_sec / 60 * config.density_per_min))


def plan_events(
    word_cues: Sequence[WordCue],
    *,
    duration_sec: float,
    matcher: Matcher,
    config: StickerConfig,
    asset_path_for: Optional[object] = None,
) -> List[StickerEvent]:
    """
    Отобрать моменты для стикеров.

    Алгоритм (порядок важен, проверяется тестами):
      1. Построить кандидатов функцией sticker_engine.matcher.build_phrases
         (это функция модуля, а НЕ метод протокола Matcher) и прогнать матчер
         ОДНИМ вызовом matcher.match_batch(...) — пофразовые вызовы модели медленные.
      2. ПРОДУБЛИРОВАТЬ фильтры матчера на его результатах (ТЗ §5.3): выбросить всё,
         где score < config.min_score или category in config.blocked_categories.
         Матчер может быть подменён извне, доверять ему нельзя.
      2b. Выкинуть кандидатов в первых config.edge_padding_head_sec и последних
         config.edge_padding_tail_sec секундах клипа.
      3. Отсортировать по score убыв. (при равенстве — по t_anchor возр., для детерминизма).
      4. Жадно набирать, проверяя на каждом шаге:
            - интервал до уже принятых >= config.min_gap_sec (по t_start);
            - одновременно активных <= config.max_concurrent;
            - тот же эмодзи не ближе config.same_emoji_cooldown_sec и не чаще
              config.same_emoji_max_per_clip раз;
         пока не набрано max_events_for_duration().
      5. Вернуть отсортированным по t_start.

    t_start == t_anchor == начало слова-триггера.
    t_end == t_start + config.total_duration.

    Args:
        asset_path_for: callable (emoji: str) -> str | None, отдаёт путь к PNG.
            В проде — метод каталога; в тестах — стаб. Если вернул None
            (эмодзи нет в каталоге) — событие отбрасывается с DEBUG-логом.
    """
    from .matcher import build_phrases  # локальный импорт — см. пояснение ниже

    candidates = build_phrases(word_cues, config)
    if not candidates:
        return []

    phrases = [c.phrase for c in candidates]
    results = matcher.match_batch(phrases)

    raw_events: List[StickerEvent] = []
    for cand, res in zip(candidates, results):
        if res is None:
            continue
        emoji, category, score = res

        # §5.3 — дублируем фильтры матчера, доверять ему нельзя
        if score < config.min_score or category in config.blocked_categories:
            continue

        # §6.2b — края клипа
        if cand.t_start < config.edge_padding_head_sec:
            continue
        if cand.t_start > duration_sec - config.edge_padding_tail_sec:
            continue

        asset_path = None
        if asset_path_for is not None:
            try:
                asset_path = asset_path_for(emoji)
            except Exception as e:
                logger.debug("asset_path_for упал на %s: %s", emoji, e)
        if not asset_path:
            logger.debug("эмодзи %s нет в каталоге — событие отброшено", emoji)
            continue

        raw_events.append(StickerEvent(
            t_anchor=cand.t_start,
            t_start=cand.t_start,
            t_end=cand.t_start + config.total_duration,
            emoji=emoji,
            asset_path=asset_path,
            trigger_word=cand.word,
            trigger_word_index=cand.word_index,
            phrase=cand.phrase,
            score=score,
            category=category,
        ))

    # §6.3 — сортировка по убыв. score, при равенстве по возр. t_anchor
    raw_events.sort(key=lambda e: (-e.score, e.t_anchor))

    max_count = max_events_for_duration(duration_sec, config)
    accepted: List[StickerEvent] = []

    for e in raw_events:
        if len(accepted) >= max_count:
            break

        if any(abs(e.t_start - a.t_start) < config.min_gap_sec for a in accepted):
            continue

        overlapping = sum(1 for a in accepted if a.t_start < e.t_end and e.t_start < a.t_end)
        if overlapping >= config.max_concurrent:
            continue

        same_emoji = [a for a in accepted if a.emoji == e.emoji]
        if len(same_emoji) >= config.same_emoji_max_per_clip:
            continue
        if any(abs(e.t_start - a.t_start) < config.same_emoji_cooldown_sec for a in same_emoji):
            continue

        accepted.append(e)

    accepted.sort(key=lambda e: e.t_start)
    return accepted


def drop_events_in_exclusive_zones(
    events: Sequence[StickerEvent],
    forbidden_zones: Sequence[ForbiddenZone],
) -> List[StickerEvent]:
    """
    Удалить события, чьё окно пересекается с зоной kind in EXCLUSIVE_ZONE_KINDS
    ('broll', 'endscreen'). Ни переноса во времени, ни сдвига — только удаление.
    """
    exclusive = [z for z in forbidden_zones if z.kind in EXCLUSIVE_ZONE_KINDS]
    result = []
    for e in events:
        if any(z.t_start < e.t_end and z.t_end > e.t_start for z in exclusive):
            continue
        result.append(e)
    return result

