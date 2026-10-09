"""Приёмка §7 — размещение с запретными зонами."""

from __future__ import annotations

from sticker_engine.placer import active_zones, candidate_grid, place_events
from sticker_engine.types import ForbiddenZone, Rect

W, H = 1080, 1920


def _rects_intersect(a: Rect, b: Rect, pad: int = 0) -> bool:
    ax0, ay0, ax1, ay1 = a.x - pad, a.y - pad, a.x + a.w + pad, a.y + a.h + pad
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def test_size_from_config(make_event, cfg):
    placed = place_events([make_event(3.0)], output_w=W, output_h=H, config=cfg)
    assert placed
    assert placed[0].size == round(cfg.size_frac * W)


def test_inside_frame_with_margin(make_event, cfg):
    placed = place_events([make_event(3.0)], output_w=W, output_h=H, config=cfg)
    m = cfg.frame_margin(W)
    p = placed[0]
    assert p.x >= m and p.y >= m
    assert p.x + p.size <= W - m
    assert p.y + p.size <= H - m


def test_never_overlaps_active_zone(make_event, cfg):
    zones = [
        ForbiddenZone(0.0, 30.0, Rect(240, 600, 600, 700), "face"),
        ForbiddenZone(0.0, 30.0, Rect(0, 1400, 1080, 400), "subtitle"),
        ForbiddenZone(0.0, 3.0, Rect(0, 0, 1080, 320), "header"),
    ]
    events = [make_event(t) for t in (2.0, 6.0, 10.0, 14.0)]
    placed = place_events(events, output_w=W, output_h=H,
                          forbidden_zones=zones, config=cfg)
    pad = cfg.zone_padding(W)
    for p in placed:
        for z in active_zones(zones, p.event.t_start, p.event.t_end):
            assert not _rects_intersect(p.rect, z.rect, pad), \
                f"стикер на t={p.event.t_start} пересёк зону {z.kind}"


def test_event_dropped_when_no_room(make_event, cfg):
    """Кадр занят целиком — событие удаляется, не сжимается и не двигается."""
    zones = [ForbiddenZone(0.0, 30.0, Rect(0, 0, W, H), "face")]
    placed = place_events([make_event(5.0)], output_w=W, output_h=H,
                          forbidden_zones=zones, config=cfg)
    assert placed == []


def test_inactive_zone_does_not_block(make_event, cfg):
    """Зона, окно которой не пересекается с событием, не мешает."""
    ev = make_event(10.0)
    zones = [ForbiddenZone(0.0, 5.0, Rect(0, 0, W, H), "face")]
    placed = place_events([ev], output_w=W, output_h=H,
                          forbidden_zones=zones, config=cfg)
    assert len(placed) == 1


def test_active_zones_boundaries(cfg):
    zones = [
        ForbiddenZone(0.0, 5.0, Rect(0, 0, 10, 10), "face"),
        ForbiddenZone(5.0, 9.0, Rect(0, 0, 10, 10), "face"),
        ForbiddenZone(9.5, 12.0, Rect(0, 0, 10, 10), "face"),
    ]
    got = active_zones(zones, 4.9, 6.4)
    assert len(got) == 2
    assert active_zones(zones, 9.0, 9.4) == []


def test_consecutive_stickers_not_in_same_cell(make_event, cfg):
    """Жёсткое правило §7.1: подряд идущие стикеры не стоят в одной клетке сетки."""
    events = [make_event(t) for t in (2.0, 5.0, 8.0, 11.0, 14.0)]
    placed = place_events(events, output_w=W, output_h=H, config=cfg)
    positions = [(p.x, p.y) for p in placed]
    for a, b in zip(positions, positions[1:]):
        assert a != b, "два стикера подряд в одной и той же позиции"


def test_prefers_upper_half_when_free(make_event, cfg):
    """Низ занят субтитрами на весь клип — стикеры уходят наверх."""
    zones = [ForbiddenZone(0.0, 30.0, Rect(0, int(H * 0.62), W, int(H * 0.38)), "subtitle")]
    events = [make_event(t) for t in (2.0, 6.0, 10.0)]
    placed = place_events(events, output_w=W, output_h=H,
                          forbidden_zones=zones, config=cfg)
    assert placed
    for p in placed:
        assert p.y + p.size <= H * 0.62


def test_deterministic(make_event, cfg):
    events = [make_event(t) for t in (2.0, 6.0, 10.0)]
    zones = [ForbiddenZone(0.0, 30.0, Rect(240, 600, 600, 700), "face")]
    a = place_events(events, output_w=W, output_h=H, forbidden_zones=zones, config=cfg)
    b = place_events(events, output_w=W, output_h=H, forbidden_zones=zones, config=cfg)
    assert [(p.x, p.y, p.size) for p in a] == [(p.x, p.y, p.size) for p in b]


def test_works_for_non_vertical_frame(make_event, cfg):
    placed = place_events([make_event(3.0)], output_w=1080, output_h=1080, config=cfg)
    assert placed
    p = placed[0]
    assert p.x + p.size <= 1080 and p.y + p.size <= 1080


def test_candidate_grid_shape(cfg):
    size = cfg.sticker_size(W)
    grid = candidate_grid(W, H, size, cfg)
    assert len(grid) >= 20
    m = cfg.frame_margin(W)
    for r in grid:
        assert r.w == size and r.h == size
        assert r.x >= m and r.y >= m
        assert r.x + r.w <= W - m and r.y + r.h <= H - m


def test_output_sorted_by_time(make_event, cfg):
    events = [make_event(t) for t in (10.0, 2.0, 6.0)]
    placed = place_events(events, output_w=W, output_h=H, config=cfg)
    assert [p.event.t_start for p in placed] == sorted(p.event.t_start for p in placed)
