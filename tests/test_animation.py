"""Приёмка §8 — анимация pop-in / hold / fade-out. Чистая математика."""

from __future__ import annotations

import pytest

from sticker_engine.animation import alpha_at, is_active, scale_at


def test_alpha_is_zero_outside_window(cfg):
    assert alpha_at(-0.01, cfg) == 0.0
    assert alpha_at(0.0, cfg) == 0.0
    assert alpha_at(cfg.total_duration, cfg) == 0.0
    assert alpha_at(cfg.total_duration + 1.0, cfg) == 0.0


def test_alpha_full_during_hold(cfg):
    mid_hold = cfg.pop_in_sec + cfg.hold_sec / 2
    assert alpha_at(mid_hold, cfg) == pytest.approx(1.0, abs=1e-3)


def test_alpha_monotonic_down_during_fade(cfg):
    t0 = cfg.pop_in_sec + cfg.hold_sec
    samples = [alpha_at(t0 + i * cfg.fade_out_sec / 10, cfg) for i in range(11)]
    assert all(b <= a + 1e-9 for a, b in zip(samples, samples[1:])), samples
    assert samples[0] == pytest.approx(1.0, abs=1e-3)
    assert samples[-1] == pytest.approx(0.0, abs=1e-3)


def test_alpha_in_range(cfg):
    for i in range(0, 200):
        t = i * cfg.total_duration / 199
        assert 0.0 <= alpha_at(t, cfg) <= 1.0


def test_scale_overshoots_then_settles(cfg):
    samples = [scale_at(i * cfg.pop_in_sec / 200, cfg) for i in range(201)]
    assert max(samples) > 1.0, "pop-in обязан переваливать за 1.0 (пружина)"
    assert max(samples) == pytest.approx(cfg.pop_overshoot, abs=0.02), \
        "пик обязан быть нормирован под config.pop_overshoot (ТЗ §8)"
    assert scale_at(cfg.pop_in_sec, cfg) == pytest.approx(1.0, abs=0.01)


def test_scale_stable_during_hold(cfg):
    t0 = cfg.pop_in_sec
    for i in range(20):
        t = t0 + i * cfg.hold_sec / 19
        assert scale_at(t, cfg) == pytest.approx(1.0, abs=1e-3)


def test_scale_is_continuous(cfg):
    """
    Непрерывность, а не «медленность»: 0 -> 1 за 0.15 c это ~4 кадра при 30 fps,
    четверть шкалы на кадр там неизбежна и нормальна. Проверяем отсутствие
    РАЗРЫВОВ на мелком шаге, особенно на границах фаз.
    """
    step = 0.001
    prev = scale_at(0.0, cfg)
    t = step
    while t < cfg.total_duration:
        cur = scale_at(t, cfg)
        assert abs(cur - prev) <= 0.05, f"разрыв масштаба на t={t:.4f}: {prev} -> {cur}"
        prev = cur
        t += step


def test_alpha_is_continuous(cfg):
    step = 0.001
    prev = alpha_at(step, cfg)
    t = 2 * step
    while t < cfg.total_duration - step:
        cur = alpha_at(t, cfg)
        assert abs(cur - prev) <= 0.05, f"разрыв альфы на t={t:.4f}: {prev} -> {cur}"
        prev = cur
        t += step


def test_pop_in_has_single_maximum(cfg):
    """На pop-in ровно один локальный максимум — овершут, без дребезга."""
    samples = [scale_at(i * cfg.pop_in_sec / 200, cfg) for i in range(201)]
    turns = sum(1 for a, b, c in zip(samples, samples[1:], samples[2:])
                if b > a + 1e-9 and b >= c + 1e-9)
    assert turns <= 1, f"на pop-in {turns} локальных максимумов, ожидался 1"


def test_is_active_matches_alpha(cfg):
    for i in range(0, 100):
        t = -0.2 + i * (cfg.total_duration + 0.4) / 99
        assert is_active(t, cfg) == (alpha_at(t, cfg) > 0.0)
