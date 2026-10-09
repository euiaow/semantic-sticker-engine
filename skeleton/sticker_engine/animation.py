"""
Анимация стикера: pop-in -> hold -> fade-out.

Обе функции ЧИСТЫЕ: никаких кадров, PIL, numpy. Тестируются напрямую (см. ТЗ §8).
`elapsed` = time_sec - event.t_start.
"""

from __future__ import annotations

from .types import StickerConfig

__all__ = ["scale_at", "alpha_at", "is_active"]


C1, C3 = 1.70158, 2.70158
RAW_PEAK = 1.0999

def scale_at(elapsed: float, config: StickerConfig) -> float:
    """
    Масштаб спрайта в момент `elapsed`.

    Фазы:
      [0, pop_in_sec)                 0 -> pop_overshoot -> 1.0 (ease-out-back)
      [pop_in_sec, +hold_sec)         1.0
      [..., +fade_out_sec)            1.0 -> 0.92
      вне окна                        0.0

    Требования (проверяются тестами):
      * пик внутри pop-in равен РОВНО config.pop_overshoot (+- 0.02) — кривая
        нормируется, см. обязательную формулу в ТЗ §8;
      * scale_at(0) == 0.0, к концу pop-in значение == 1.0 +- 0.01;
      * на pop-in ровно ОДИН локальный максимум (без дребезга);
      * функция НЕПРЕРЫВНА: при шаге 0.001 c соседние значения отличаются
        не более чем на 0.05. Ограничения «не больше X за кадр при 30 fps»
        НЕТ и быть не может: 0 -> 1 за 0.15 c это ~4 кадра;
      * масштабирование происходит ОТ ЦЕНТРА спрайта (это учитывает рендерер,
        но именно эта функция задаёт значение).
    """
    total = config.total_duration
    if elapsed <= 0 or elapsed >= total:
        return 0.0
    if elapsed < config.pop_in_sec:
        p = elapsed / config.pop_in_sec
        raw = 1 + C3 * (p - 1) ** 3 + C1 * (p - 1) ** 2
        if raw <= 1.0:
            return raw
        k = (config.pop_overshoot - 1.0) / (RAW_PEAK - 1.0)
        return 1.0 + (raw - 1.0) * k
    if elapsed < config.pop_in_sec + config.hold_sec:
        return 1.0
    t = min(1.0, (elapsed - config.pop_in_sec - config.hold_sec) / config.fade_out_sec)
    return 1.0 - 0.08 * t


def alpha_at(elapsed: float, config: StickerConfig) -> float:
    """
    Непрозрачность 0.0..1.0 в момент `elapsed`.

      * 0.0 при elapsed <= 0 и при elapsed >= config.total_duration;
      * 0 -> 1 линейно за первые 40% pop_in_sec;
      * 1.0 на hold;
      * 1 -> 0 (ease-in) на fade_out_sec;
      * непрерывна: при шаге 0.001 c соседние значения отличаются не более чем на 0.05.
    """
    total = config.total_duration
    if elapsed <= 0 or elapsed >= total:
        return 0.0
    if elapsed < config.pop_in_sec:
        ramp = config.pop_in_sec * 0.4
        return 1.0 if elapsed >= ramp else elapsed / ramp
    if elapsed < config.pop_in_sec + config.hold_sec:
        return 1.0
    t = min(1.0, (elapsed - config.pop_in_sec - config.hold_sec) / config.fade_out_sec)
    return 1.0 - t



def is_active(elapsed: float, config: StickerConfig) -> bool:
    """True, если стикер виден в этот момент (эквивалентно alpha_at(...) > 0)."""
    return 0.0 < elapsed < config.total_duration
