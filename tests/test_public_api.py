"""Приёмка §3.4 / §11 — публичный API, детерминизм, устойчивость к ошибкам."""

from __future__ import annotations

import logging

import pytest

from sticker_engine import StickerRenderer, plan_stickers
from sticker_engine.types import ForbiddenZone, PlacedSticker, Rect, StickerConfig

W, H = 1080, 1920


def _plan(cues, cfg, zones=(), matcher=None, duration=None):
    return plan_stickers(
        cues,
        duration_sec=duration if duration is not None else cues[-1].end + 1.0,
        output_w=W, output_h=H,
        forbidden_zones=zones,
        config=cfg,
        matcher=matcher,
    )


def test_returns_placed_stickers(sample_cues, cfg, stub_matcher):
    placed = _plan(sample_cues, cfg, matcher=stub_matcher)
    assert isinstance(placed, list)
    assert all(isinstance(p, PlacedSticker) for p in placed)
    assert placed, "стаб-матчер обязан дать хотя бы один стикер"


def test_end_to_end_deterministic(sample_cues, cfg, sample_zones):
    from conftest import StubMatcher

    a = _plan(sample_cues, cfg, sample_zones, StubMatcher())
    b = _plan(sample_cues, cfg, sample_zones, StubMatcher())
    key = lambda ps: [(p.event.t_start, p.event.emoji, p.x, p.y, p.size) for p in ps]
    assert key(a) == key(b)


def test_zones_are_respected_end_to_end(sample_cues, cfg, sample_zones):
    from conftest import StubMatcher
    from sticker_engine.placer import active_zones

    placed = _plan(sample_cues, cfg, sample_zones, StubMatcher())
    pad = cfg.zone_padding(W)
    for p in placed:
        for z in active_zones(sample_zones, p.event.t_start, p.event.t_end):
            ax0, ay0 = p.x - pad, p.y - pad
            ax1, ay1 = p.x + p.size + pad, p.y + p.size + pad
            bx0, by0, bx1, by1 = z.rect.x, z.rect.y, z.rect.x + z.rect.w, z.rect.y + z.rect.h
            assert not (ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1)


def test_broll_window_has_no_stickers(sample_cues, cfg):
    from conftest import StubMatcher

    zones = [ForbiddenZone(0.0, 999.0, Rect(0, 0, W, H), "broll")]
    assert _plan(sample_cues, cfg, zones, StubMatcher()) == []


def test_never_raises_on_garbage(cfg, caplog):
    """Библиотека не имеет права уронить вызывающий пайплайн (ТЗ §11)."""
    from sticker_engine.types import WordCue

    broken = [
        WordCue(start=float("nan"), end=1.0, word="что-то"),
        WordCue(start=5.0, end=2.0, word=""),          # end < start
        WordCue(start=-3.0, end=-1.0, word="миллион"),
    ]
    with caplog.at_level(logging.WARNING):
        out = plan_stickers(broken, duration_sec=10.0, output_w=W, output_h=H, config=cfg)
    assert isinstance(out, list)


def test_zero_duration(cfg, sample_cues, stub_matcher):
    assert _plan(sample_cues, cfg, matcher=stub_matcher, duration=0.0) == []


def test_renderer_accepts_plan_output(sample_cues, cfg, stub_matcher):
    placed = _plan(sample_cues, cfg, matcher=stub_matcher)
    r = StickerRenderer(placed, W, H, cfg)
    t = placed[0].event.t_start + cfg.pop_in_sec + 0.05
    assert len(r.get_overlays(t)) == 1


def test_config_defaults_unchanged():
    """Дефолты согласованы с заказчиком. Менять их — только через ТЗ."""
    c = StickerConfig()
    assert c.max_per_clip == 10
    assert c.min_gap_sec == 2.5
    assert c.max_concurrent == 1
    assert c.size_frac == pytest.approx(0.13)
    assert c.pop_in_sec == pytest.approx(0.15)
    assert c.hold_sec == pytest.approx(1.20)
    assert c.fade_out_sec == pytest.approx(0.20)
    assert c.total_duration == pytest.approx(1.55)
    assert "weapons" in c.blocked_categories


def test_plan_is_fast(sample_cues, cfg, stub_matcher):
    import time

    t0 = time.perf_counter()
    _plan(sample_cues, cfg, matcher=stub_matcher)
    assert time.perf_counter() - t0 < 2.0
