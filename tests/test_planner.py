"""Приёмка §6 — планировщик моментов. Матчер подменён стабом, модель не грузится."""

from __future__ import annotations

import math

import pytest

from sticker_engine.planner import (
    drop_events_in_exclusive_zones,
    max_events_for_duration,
    plan_events,
)
from sticker_engine.types import ForbiddenZone, Rect, StickerConfig


def _plan(cues, matcher, cfg, asset_path, duration=None):
    return plan_events(
        cues,
        duration_sec=duration if duration is not None else (cues[-1].end + 1.0),
        matcher=matcher,
        config=cfg,
        asset_path_for=asset_path,
    )


def test_max_events_formula(cfg):
    assert max_events_for_duration(60.0, cfg) == min(cfg.max_per_clip, 12)
    assert max_events_for_duration(30.0, cfg) == math.ceil(30 / 60 * cfg.density_per_min)
    assert max_events_for_duration(1.0, cfg) >= 1


def test_events_sorted_by_time(sample_cues, stub_matcher, cfg, stub_asset_path):
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path)
    starts = [e.t_start for e in events]
    assert starts == sorted(starts)


def test_min_gap_respected(sample_cues, stub_matcher, cfg, stub_asset_path):
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path)
    for a, b in zip(events, events[1:]):
        assert b.t_start - a.t_start >= cfg.min_gap_sec - 1e-6


def test_max_concurrent_respected(sample_cues, stub_matcher, cfg, stub_asset_path):
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path)
    for i, a in enumerate(events):
        overlapping = [b for b in events[i + 1:] if b.t_start < a.t_end]
        assert len(overlapping) < cfg.max_concurrent


def test_count_within_density_budget(sample_cues, stub_matcher, cfg, stub_asset_path):
    duration = sample_cues[-1].end + 1.0
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path, duration)
    assert len(events) <= max_events_for_duration(duration, cfg)
    assert len(events) <= cfg.max_per_clip


def test_timing_contract(sample_cues, stub_matcher, cfg, stub_asset_path):
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path)
    assert events, "стаб-матчер обязан найти хоть что-то на тестовом транскрипте"
    for e in events:
        assert e.t_start == pytest.approx(e.t_anchor)
        assert e.t_end == pytest.approx(e.t_start + cfg.total_duration)
        assert 0.0 <= e.score <= 1.0
        assert e.asset_path
        assert e.trigger_word
        assert 0 <= e.trigger_word_index < len(sample_cues)


def test_trigger_word_matches_transcript(sample_cues, stub_matcher, cfg, stub_asset_path):
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path)
    for e in events:
        cue = sample_cues[e.trigger_word_index]
        assert e.t_anchor == pytest.approx(cue.start, abs=1e-6)


def test_edges_are_kept_clear(sample_cues, stub_matcher, cfg, stub_asset_path):
    duration = sample_cues[-1].end + 1.0
    events = _plan(sample_cues, stub_matcher, cfg, stub_asset_path, duration)
    for e in events:
        assert e.t_start >= cfg.edge_padding_head_sec - 1e-6
        assert e.t_start <= duration - cfg.edge_padding_tail_sec + 1e-6


def test_blocked_category_never_emitted(sample_cues, cfg, stub_asset_path):
    """🔪 (weapons) имеет максимальный score в стабе и обязан быть отфильтрован."""
    from conftest import StubMatcher

    matcher = StubMatcher(blocked=set(cfg.blocked_categories))
    events = _plan(sample_cues, matcher, cfg, stub_asset_path)
    assert all(e.emoji != "🔪" for e in events)
    assert all(e.category not in cfg.blocked_categories for e in events)


def test_same_emoji_cooldown(sample_cues, cfg, stub_asset_path):
    from conftest import StubMatcher

    # всё матчится в один и тот же эмодзи
    matcher = StubMatcher(rules=[("", "🔥", "hype", 0.9)])
    events = _plan(sample_cues, matcher, cfg, stub_asset_path)
    assert len(events) <= cfg.same_emoji_max_per_clip
    for a, b in zip(events, events[1:]):
        assert b.t_start - a.t_start >= cfg.same_emoji_cooldown_sec - 1e-6


def test_low_score_yields_nothing(sample_cues, cfg, stub_asset_path):
    from conftest import StubMatcher

    matcher = StubMatcher(rules=[("миллион", "💰", "money", 0.10)])
    assert _plan(sample_cues, matcher, cfg, stub_asset_path) == []


def test_empty_transcript_is_safe(stub_matcher, cfg, stub_asset_path):
    assert plan_events([], duration_sec=30.0, matcher=stub_matcher,
                       config=cfg, asset_path_for=stub_asset_path) == []


def test_deterministic(sample_cues, cfg, stub_asset_path):
    from conftest import StubMatcher

    a = _plan(sample_cues, StubMatcher(), cfg, stub_asset_path)
    b = _plan(sample_cues, StubMatcher(), cfg, stub_asset_path)
    assert [(e.t_start, e.emoji, e.trigger_word_index) for e in a] == \
           [(e.t_start, e.emoji, e.trigger_word_index) for e in b]


def test_matcher_called_in_batch(sample_cues, cfg, stub_asset_path):
    """Матчер не должен вызываться пословно по одной фразе за раз без нужды."""
    from conftest import StubMatcher

    m = StubMatcher()
    _plan(sample_cues, m, cfg, stub_asset_path)
    assert m.calls <= len(sample_cues), "слишком много вызовов матчера"


def test_exclusive_zones_drop_events(make_event):
    events = [make_event(2.0), make_event(8.0), make_event(14.0)]
    zones = [
        ForbiddenZone(t_start=7.5, t_end=12.0, rect=Rect(0, 0, 1080, 1920), kind="broll"),
        ForbiddenZone(t_start=13.0, t_end=20.0, rect=Rect(0, 0, 1080, 1920), kind="endscreen"),
    ]
    kept = drop_events_in_exclusive_zones(events, zones)
    assert [e.t_start for e in kept] == [2.0]


def test_face_zone_does_not_drop_events(make_event):
    """Лицо — не эксклюзивная зона: событие остаётся, его подвинет placer."""
    events = [make_event(5.0)]
    zones = [ForbiddenZone(t_start=0.0, t_end=30.0,
                           rect=Rect(300, 500, 480, 480), kind="face")]
    assert len(drop_events_in_exclusive_zones(events, zones)) == 1


def test_custom_config_changes_budget(sample_cues, stub_matcher, stub_asset_path):
    tight = StickerConfig(max_per_clip=2, min_gap_sec=1.0, min_score=0.3)
    events = _plan(sample_cues, stub_matcher, tight, stub_asset_path)
    assert len(events) <= 2
