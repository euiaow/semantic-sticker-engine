"""
Решатель позиции: куда положить стикер, чтобы не задеть лицо, субтитры,
хедер и не вылезти за кадр (ТЗ §7).

Чистая геометрия, без numpy-картинок. Полностью тестируется.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

from .types import ForbiddenZone, PlacedSticker, Rect, StickerConfig, StickerEvent

logger = logging.getLogger("sticker_engine.placer")

__all__ = ["place_events", "candidate_grid", "score_position", "active_zones"]

# Сетка кандидатов позиций. Плотнее — лучше качество, дороже CPU.
GRID_COLS = 6
GRID_ROWS = 10

# --- веса мягких предпочтений (крутятся заказчиком, не хардкодить в цикле) ---
# Каждое слагаемое нормируется в [0, 1] ДО взвешивания, иначе веса несравнимы.
W_FACE_DISTANCE = 1.00      # дальше от лица — лучше; расстояние / диагональ кадра
W_UPPER_HALF = 0.25         # верхняя половина кадра предпочтительнее
W_CENTER_X = 0.25           # ближе к центральной трети по X — лучше

# Если зон kind == 'face' в окне события нет, слагаемое расстояния до лица = FACE_TERM_NO_FACE
# (все позиции равноценны по этому критерию).
FACE_TERM_NO_FACE = 1.0


def active_zones(zones: Sequence[ForbiddenZone],
                 t_start: float, t_end: float) -> List[ForbiddenZone]:
    """Зоны, активные хотя бы частично в окне [t_start, t_end)."""
    return [z for z in zones if z.t_start < t_end and z.t_end > t_start]


def candidate_grid(output_w: int, output_h: int, size: int,
                   config: StickerConfig) -> List[Rect]:
    """
    Сетка позиций-кандидатов GRID_COLS x GRID_ROWS внутри кадра с учётом
    config.margin_frac. Возвращает прямоугольники размера size x size.
    Порядок детерминирован (слева-направо, сверху-вниз).
    """
    margin = config.frame_margin(output_w)
    avail_w = max(0, output_w - 2 * margin - size)
    avail_h = max(0, output_h - 2 * margin - size)
    step_x = avail_w / (GRID_COLS - 1) if GRID_COLS > 1 else 0
    step_y = avail_h / (GRID_ROWS - 1) if GRID_ROWS > 1 else 0

    grid = []
    for row in range(GRID_ROWS):
        y = round(margin + row * step_y)
        for col in range(GRID_COLS):
            x = round(margin + col * step_x)
            grid.append(Rect(x, y, size, size))
    return grid


def score_position(candidate: Rect, zones: Sequence[ForbiddenZone],
                   output_w: int, output_h: int) -> float:
    """
    Оценка допустимой позиции (чем больше — тем лучше). Использует W_* константы
    модуля; каждое слагаемое нормировано в [0, 1]. Про жёсткие запреты здесь не
    думаем — их отфильтровал вызывающий (в т.ч. запрет на клетку предыдущего стикера).

    Если зон kind == 'face' среди `zones` нет, слагаемое расстояния до лица
    равно FACE_TERM_NO_FACE.
    """
    cx = candidate.x + candidate.w / 2
    cy = candidate.y + candidate.h / 2

    face_zones = [z for z in zones if z.kind == "face"]
    if face_zones:
        diagonal = math.hypot(output_w, output_h)
        face_term = min(
            min(1.0, math.hypot(cx - (z.rect.x + z.rect.w / 2),
                                 cy - (z.rect.y + z.rect.h / 2)) / diagonal)
            for z in face_zones
        )
    else:
        face_term = FACE_TERM_NO_FACE

    upper_term = max(0.0, min(1.0, 1.0 - cy / output_h))
    center_term = max(0.0, min(1.0, 1.0 - abs(cx - output_w / 2) / (output_w / 2)))

    return (W_FACE_DISTANCE * face_term
            + W_UPPER_HALF * upper_term
            + W_CENTER_X * center_term)


def place_events(
    events: Sequence[StickerEvent],
    *,
    output_w: int,
    output_h: int,
    forbidden_zones: Sequence[ForbiddenZone] = (),
    config: Optional[StickerConfig] = None,
) -> List[PlacedSticker]:
    """
    Разместить события на кадре.

    Для каждого события (в порядке t_start):
      1. size = config.sticker_size(output_w);
      2. кандидаты = candidate_grid(...);
      3. отфильтровать те, что пересекаются (с паддингом config.zone_padding_frac)
         с любой зоной, активной в окне события;
      4. исключить клетку, в которой стоял ПРЕДЫДУЩИЙ размещённый стикер этого клипа
         (жёсткое правило ТЗ §7.1). Если после исключения кандидатов не осталось —
         запрет снимается (лучше повтор позиции, чем потерянный стикер), лог DEBUG;
      5. выбрать максимум по score_position; при равенстве — меньший индекс клетки
         (детерминизм);
      6. если допустимых кандидатов нет — СОБЫТИЕ УДАЛЯЕТСЯ (лог DEBUG),
         никаких «подвинем» и «уменьшим».

    Returns:
        Отсортированный по t_start список PlacedSticker. Длина может быть
        меньше длины events.
    """
    if config is None:
        config = StickerConfig()

    size = config.sticker_size(output_w)
    pad = config.zone_padding(output_w)
    grid = candidate_grid(output_w, output_h, size, config)

    placed = []
    prev_cell = None

    for event in sorted(events, key=lambda e: e.t_start):
        zones = active_zones(forbidden_zones, event.t_start, event.t_end)

        valid = [i for i, cand in enumerate(grid)
                 if not any(cand.intersects(z.rect, pad) for z in zones)]

        if prev_cell is not None:
            without_prev = [i for i in valid if i != prev_cell]
            if without_prev:
                valid = without_prev
            else:
                logger.debug("нет альтернативы клетке предыдущего стикера")

        if not valid:
            logger.debug("для события %s не нашлось места", event.trigger_word)
            continue

        best = max(valid, key=lambda i: (score_position(grid[i], zones, output_w, output_h), -i))
        chosen = grid[best]
        placed.append(PlacedSticker(event=event, x=chosen.x, y=chosen.y, size=size))
        prev_cell = best

    return placed
