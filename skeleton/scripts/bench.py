"""Замеры под ТЗ §11. Запуск из skeleton/: python scripts/bench.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sticker_engine import StickerRenderer, plan_stickers
from sticker_engine.cli import load_word_cues
from sticker_engine.types import StickerConfig

cues = load_word_cues(ROOT.parent / "fixtures" / "word_cues.sample.json")
cfg = StickerConfig()

t0 = time.perf_counter()
placed = plan_stickers(cues, duration_sec=60.0, output_w=1080, output_h=1920, config=cfg)
t_first = time.perf_counter() - t0
print(f"plan_stickers (первый вызов, с загрузкой модели): {t_first:.2f} с")

t0 = time.perf_counter()
placed = plan_stickers(cues, duration_sec=60.0, output_w=1080, output_h=1920, config=cfg)
print(f"plan_stickers (модель в кэше):                    {time.perf_counter() - t0:.3f} с")
print(f"стикеров в плане: {len(placed)}")

t0 = time.perf_counter()
r = StickerRenderer(placed, 1080, 1920, cfg)
print(f"StickerRenderer.__init__:                         {time.perf_counter() - t0:.3f} с")

if placed:
    t = placed[0].event.t_start + cfg.pop_in_sec + 0.05
    r.get_overlays(t)
    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        r.get_overlays(t + (i % 20) * 0.001)
    print(f"get_overlays (медиана по {n} вызовам):            {(time.perf_counter()-t0)/n*1000:.4f} мс")