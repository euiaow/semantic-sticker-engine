"""Приёмка §9 — покадровый рендерер оверлеев."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sticker_engine.placer import place_events
from sticker_engine.renderer import StickerRenderer

W, H = 1080, 1920


@pytest.fixture
def renderer(make_event, cfg):
    events = [make_event(2.0), make_event(6.0, emoji="💰", word="миллион", category="money")]
    placed = place_events(events, output_w=W, output_h=H, config=cfg)
    return StickerRenderer(placed, W, H, cfg), placed


def test_empty_outside_windows(renderer, cfg):
    r, placed = renderer
    assert r.get_overlays(0.0) == []
    assert r.get_overlays(placed[-1].event.t_end + 0.5) == []
    assert r.get_overlays(4.5) == []      # между двумя событиями


def test_overlay_format(renderer, cfg):
    r, placed = renderer
    t = placed[0].event.t_start + cfg.pop_in_sec + 0.1
    out = r.get_overlays(t)
    assert len(out) == 1
    rgba, x, y = out[0]
    assert isinstance(rgba, np.ndarray)
    assert rgba.dtype == np.uint8
    assert rgba.ndim == 3 and rgba.shape[2] == 4
    assert isinstance(x, int) and isinstance(y, int)


def test_overlay_inside_frame(renderer):
    r, placed = renderer
    ev = placed[0].event
    t = ev.t_start
    while t < ev.t_end:
        for rgba, x, y in r.get_overlays(t):
            h, w = rgba.shape[:2]
            assert x >= 0 and y >= 0
            assert x + w <= W and y + h <= H
        t += 1 / 30


def test_size_at_hold_matches_plan(renderer, cfg):
    r, placed = renderer
    p = placed[0]
    t = p.event.t_start + cfg.pop_in_sec + cfg.hold_sec / 2
    rgba, x, y = r.get_overlays(t)[0]
    h, w = rgba.shape[:2]
    assert abs(w - p.size) <= 2 and abs(h - p.size) <= 2


def test_center_is_stable_across_scales(renderer, cfg):
    """Масштаб растёт от центра: центр спрайта не должен ездить."""
    r, placed = renderer
    p = placed[0]
    centers = []
    t = p.event.t_start + 0.01
    while t < p.event.t_start + cfg.pop_in_sec + cfg.hold_sec:
        out = r.get_overlays(t)
        if out:
            rgba, x, y = out[0]
            h, w = rgba.shape[:2]
            centers.append((x + w / 2, y + h / 2))
        t += 1 / 60
    assert centers
    cx = [c[0] for c in centers]
    cy = [c[1] for c in centers]
    assert max(cx) - min(cx) <= 2.0, f"центр по X уехал на {max(cx) - min(cx):.1f}px"
    assert max(cy) - min(cy) <= 2.0, f"центр по Y уехал на {max(cy) - min(cy):.1f}px"


def test_alpha_follows_animation(renderer, cfg):
    from sticker_engine.animation import alpha_at

    r, placed = renderer
    p = placed[0]
    for frac in (0.05, 0.5, 0.9):
        elapsed = frac * cfg.total_duration
        out = r.get_overlays(p.event.t_start + elapsed)
        if not out:
            continue
        rgba = out[0][0]
        peak = float(rgba[:, :, 3].max()) / 255.0
        assert peak == pytest.approx(alpha_at(elapsed, cfg), abs=0.06)


def test_max_alpha_reaches_full_on_hold(renderer, cfg):
    r, placed = renderer
    t = placed[0].event.t_start + cfg.pop_in_sec + cfg.hold_sec / 2
    rgba = r.get_overlays(t)[0][0]
    assert rgba[:, :, 3].max() >= 250


def test_no_alpha_halo_after_resize(renderer, cfg):
    """Углы квадратного спрайта с круглым рисунком обязаны остаться прозрачными."""
    r, placed = renderer
    t = placed[0].event.t_start + cfg.pop_in_sec + 0.1
    rgba = r.get_overlays(t)[0][0]
    assert rgba[0, 0, 3] < 10
    assert rgba[0, -1, 3] < 10
    assert rgba[-1, 0, 3] < 10
    assert rgba[-1, -1, 3] < 10


def test_get_overlays_is_fast(renderer, cfg):
    r, placed = renderer
    t = placed[0].event.t_start + cfg.pop_in_sec + 0.05
    r.get_overlays(t)                       # прогрев
    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        r.get_overlays(t + (i % 20) * 0.001)
    per_call_ms = (time.perf_counter() - t0) / n * 1000
    assert per_call_ms < 0.3, f"{per_call_ms:.3f} мс/вызов — слишком медленно (ТЗ §11)"


def test_missing_asset_is_survived(make_event, cfg):
    """Битый путь к PNG: WARNING и пропуск, а не исключение."""
    from sticker_engine.types import PlacedSticker, StickerEvent

    bad = StickerEvent(
        t_anchor=2.0, t_start=2.0, t_end=2.0 + cfg.total_duration,
        emoji="💰", asset_path="/nope/does-not-exist.png",
        trigger_word="миллион", trigger_word_index=0, phrase="...", score=0.9,
        category="money",
    )
    r = StickerRenderer([PlacedSticker(bad, 100, 100, 140)], W, H, cfg)
    assert r.get_overlays(2.5) == []


def test_empty_plan(cfg):
    r = StickerRenderer([], W, H, cfg)
    assert r.get_overlays(1.0) == []
    assert r.active_windows == []
