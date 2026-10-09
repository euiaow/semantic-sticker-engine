"""
Покадровый рендерер стикеров (ТЗ §9).

Вызывается до 3000 раз на клип — весь тяжёлый ресайз делается в __init__
и кэшируется. get_overlays() должен быть словарным лукапом.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .types import PlacedSticker, StickerConfig

logger = logging.getLogger("sticker_engine.renderer")

__all__ = ["StickerRenderer", "SCALE_QUANT"]

# Шаг квантования для кэша спрайтов — И масштаба, И альфы.
# Ключ кэша: (индекс события, quant(scale), quant(alpha)).
SCALE_QUANT = 0.02

from PIL import Image

from .animation import alpha_at, scale_at
from .types import PlacedSticker, StickerConfig

def _quant(v: float) -> float:
    return round(v / SCALE_QUANT) * SCALE_QUANT

class StickerRenderer:
    """
    Отдаёт RGBA-оверлеи для произвольного момента времени.

    Контракт (проверяется тестами):
      * get_overlays(t) -> [(rgba, x, y), ...], обычно 0 или 1 элемент;
      * rgba.dtype == np.uint8, rgba.ndim == 3, rgba.shape[2] == 4;
      * альфа уже домножена на animation.alpha_at (RGB НЕ премультиплицирован);
      * (x, y) — левый верхний угол уже отмасштабированного спрайта;
      * спрайт целиком внутри кадра: x >= 0, y >= 0, x+w <= output_w, y+h <= output_h;
      * центр спрайта не смещается при изменении масштаба;
      * вне окна события — стикер не отдаётся вовсе;
      * никакого дискового I/O внутри get_overlays.
    """

    def __init__(
        self,
        placed: Sequence[PlacedSticker],
        output_w: int,
        output_h: int,
        config: Optional[StickerConfig] = None,
    ) -> None:
        """
        Загружает PNG, предрендерит все нужные комбинации (масштаб, альфа) с шагом
        SCALE_QUANT по обоим параметрам, строит индекс «время -> активные события».
        После __init__ get_overlays не должен ничего аллоцировать и считать —
        только доставать готовый массив из словаря.

        Битый/отсутствующий PNG: WARNING, событие молча исключается. Не падаем.
        """
        self.placed = list(placed)
        self.config = config or StickerConfig()
        self._cache: Dict[Tuple[int, float, float], np.ndarray] = {}
        self._base: Dict[int, Optional[np.ndarray]] = {}

        for idx, p in enumerate(self.placed):
            try:
                img = Image.open(p.event.asset_path).convert("RGBA").resize((p.size, p.size), Image.LANCZOS)
                self._base[idx] = np.array(img)
            except Exception as e:
                logger.warning("не удалось загрузить ассет %s: %s", p.event.asset_path, e)
                self._base[idx] = None

    def get_overlays(self, time_sec: float) -> List[Tuple[np.ndarray, int, int]]:
        """Активные оверлеи на момент time_sec. Пустой список — норма."""
        result = []
        for idx, p in enumerate(self.placed):
            base = self._base[idx]
            if base is None:
                continue
            elapsed = time_sec - p.event.t_start
            alpha = alpha_at(elapsed, self.config)
            if alpha <= 0.0:
                continue
            scale = scale_at(elapsed, self.config)
            key = (idx, _quant(scale), _quant(alpha))

            if key not in self._cache:
                size = max(1, round(p.size * key[1]))
                img = Image.fromarray(base).resize((size, size), Image.LANCZOS)
                arr = np.array(img).astype(np.float32)
                arr[..., 3] *= key[2]
                self._cache[key] = arr.astype(np.uint8)

            rgba = self._cache[key]
            size = rgba.shape[0]
            cx, cy = p.x + p.size / 2, p.y + p.size / 2
            result.append((rgba, round(cx - size / 2), round(cy - size / 2)))
        return result

    # --- вспомогательное, для отладки и тестов --------------------------------

    @property

    def active_windows(self) -> List[Tuple[float, float]]:
        """Окна [t_start, t_end) всех успешно загруженных стикеров."""
        return [(p.event.t_start, p.event.t_end)
                for idx, p in enumerate(self.placed)
                if self._base.get(idx) is not None]