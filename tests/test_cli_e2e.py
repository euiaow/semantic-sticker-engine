"""Приёмка §10 — CLI-демо end-to-end. Требует ffmpeg."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import FFPROBE, FIXTURES, SKELETON, requires_ffmpeg


def _probe(path, stream, fields):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
         f"stream={','.join(fields)}", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out)["streams"]
    return streams[0] if streams else {}


def _duration(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _frame_at(path, t):
    """
    Кадр по НОМЕРУ, а не по миллисекундам: позиционирование по POS_MSEC
    неточное и может отдать разные кадры для входа и выхода.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"не удалось прочитать кадр {idx} (t={t})"
    return frame


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "sticker_engine.cli", *args],
        capture_output=True, text=True, cwd=str(SKELETON),
    )


def test_dry_run_produces_plan(tmp_path):
    """--dry-run работает без видео и без ffmpeg (ТЗ §10)."""
    plan = tmp_path / "plan.json"
    r = _run_cli([
        "--cues", str(FIXTURES / "word_cues.sample.json"),
        "--zones", str(FIXTURES / "zones.sample.json"),
        "--plan-json", str(plan), "--dry-run",
    ])
    assert r.returncode == 0, r.stderr
    assert plan.exists()
    data = json.loads(plan.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    required = {"t_start", "t_end", "t_anchor", "emoji", "asset_path", "trigger_word",
                "trigger_word_index", "phrase", "score", "category", "x", "y", "size"}
    for item in data:
        assert required <= set(item), f"в плане не хватает ключей: {required - set(item)}"


@requires_ffmpeg
def test_burn_produces_valid_video(tmp_path, sample_video):
    out = tmp_path / "out.mp4"
    r = _run_cli([
        "--video", str(sample_video),
        "--cues", str(FIXTURES / "word_cues.sample.json"),
        "--zones", str(FIXTURES / "zones.sample.json"),
        "--out", str(out),
    ])
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 10_000

    v_in = _probe(sample_video, "v:0", ["width", "height", "r_frame_rate"])
    v_out = _probe(out, "v:0", ["width", "height", "r_frame_rate"])
    assert (v_out["width"], v_out["height"]) == (v_in["width"], v_in["height"])
    assert v_out["r_frame_rate"] == v_in["r_frame_rate"]
    assert abs(_duration(out) - _duration(sample_video)) < 0.15


@requires_ffmpeg
def test_audio_preserved(tmp_path, sample_video):
    out = tmp_path / "out.mp4"
    r = _run_cli([
        "--video", str(sample_video),
        "--cues", str(FIXTURES / "word_cues.sample.json"),
        "--out", str(out),
    ])
    assert r.returncode == 0, r.stderr
    assert _probe(out, "a:0", ["codec_type"]).get("codec_type") == "audio"


@requires_ffmpeg
def test_stickers_actually_burned(tmp_path, sample_video):
    """Кадр в окне стикера обязан отличаться от исходного; вне окна — совпадать."""
    out = tmp_path / "out.mp4"
    plan = tmp_path / "plan.json"
    r = _run_cli([
        "--video", str(sample_video),
        "--cues", str(FIXTURES / "word_cues.sample.json"),
        "--out", str(out), "--plan-json", str(plan),
    ])
    assert r.returncode == 0, r.stderr
    events = json.loads(plan.read_text(encoding="utf-8"))
    assert events, ("план пуст на тестовом транскрипте — либо min_score задран, "
                    "либо в каталоге нет эмодзи из образца (ТЗ §4.2)")

    ev = events[0]
    x, y, s = ev["x"], ev["y"], ev["size"]
    t_mid = (ev["t_start"] + ev["t_end"]) / 2
    t_clean = max(0.05, ev["t_start"] - 1.0)

    def _roi_diff(t):
        a = _frame_at(sample_video, t).astype(np.int16)[y:y + s, x:x + s]
        b = _frame_at(out, t).astype(np.int16)[y:y + s, x:x + s]
        return float(np.abs(a - b).mean())

    # Сравниваем ТОЛЬКО область стикера: на весь кадр 140x140 даёт вклад <1%,
    # который тонет в шуме перекодирования.
    d_sticker = _roi_diff(t_mid)
    d_clean = _roi_diff(t_clean)
    assert d_sticker > 8.0, f"в окне стикера область не изменилась (diff={d_sticker:.2f})"
    assert d_clean < d_sticker / 3, (
        f"вне окна стикера та же область изменена слишком сильно "
        f"(clean={d_clean:.2f}, sticker={d_sticker:.2f})")
