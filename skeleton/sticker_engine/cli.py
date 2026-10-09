"""
CLI-демо (ТЗ §10). Только демо/приёмочный стенд — в проде не используется.

Запускать из папки skeleton/ (там лежит пакет sticker_engine):

    cd skeleton
    python -m sticker_engine.cli \
        --cues  ../fixtures/word_cues.sample.json \
        --video ../fixtures/sample.mp4 \
        --zones ../fixtures/zones.sample.json \
        --out   ../out.mp4 \
        --plan-json ../plan.json

    python -m sticker_engine.cli --cues ... --dry-run     # только план, без ffmpeg

Аргументы:
    --cues      ОБЯЗАТЕЛЬНЫЙ
    --video     обязателен, кроме --dry-run
    --out       обязателен, кроме --dry-run
    --zones     опционально
    --plan-json опционально
    --duration  опционально, см. правило ниже
    --size WxH  опционально
    --config    опционально, JSON со StickerConfig
    --dry-run   только план

duration_sec берётся по правилу (ТЗ §10):
    1) --duration, если задан;
    2) иначе фактическая длительность --video (ffprobe);
    3) иначе word_cues[-1].end + config.edge_padding_tail_sec.

Размер кадра: --size, иначе из --video, иначе 1080x1920.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .types import ForbiddenZone, PlacedSticker, Rect, StickerConfig, WordCue

logger = logging.getLogger("sticker_engine.cli")

__all__ = ["main", "load_word_cues", "load_zones", "burn_stickers", "plan_to_json"]


def load_word_cues(path: Path) -> List[WordCue]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cues = []
    for item in data:
        word = item.get("word", item.get("text", ""))
        cues.append(WordCue(start=float(item["start"]), end=float(item["end"]), word=word))
    return cues


def load_zones(path: Optional[Path]) -> List[ForbiddenZone]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = []
    for item in data:
        r = item["rect"]
        zones.append(ForbiddenZone(
            t_start=float(item["t_start"]),
            t_end=float(item["t_end"]),
            rect=Rect(x=int(r["x"]), y=int(r["y"]), w=int(r["w"]), h=int(r["h"])),
            kind=item.get("kind", "custom"),
        ))
    return zones


def plan_to_json(placed: Sequence[PlacedSticker], path: Path) -> None:
    out = []
    for p in placed:
        e = p.event
        out.append({
            "t_start": e.t_start, "t_end": e.t_end, "t_anchor": e.t_anchor,
            "emoji": e.emoji, "asset_path": e.asset_path,
            "trigger_word": e.trigger_word, "trigger_word_index": e.trigger_word_index,
            "phrase": e.phrase, "score": e.score, "category": e.category,
            "x": p.x, "y": p.y, "size": p.size,
        })
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _ffprobe_value(video: Path, entries: str, stream: str = "v:0") -> Optional[str]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream,
             "-show_entries", entries, "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception as e:
        logger.warning("ffprobe не сработал на %s: %s", video, e)
        return None


def _probe_duration(video: Path) -> Optional[float]:
    val = _ffprobe_value(video, "format=duration", stream="v:0")
    if val is None:
        try:
            val = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(video)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _probe_size(video: Path) -> Optional[tuple[int, int]]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception as e:
        logger.warning("не удалось определить размер кадра: %s", e)
        return None


def burn_stickers(
    video_in: Path,
    video_out: Path,
    placed: Sequence[PlacedSticker],
    config: Optional[StickerConfig] = None,
) -> None:
    import cv2
    import numpy as np

    from .renderer import StickerRenderer

    if config is None:
        config = StickerConfig()

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    renderer = StickerRenderer(placed, width, height, config)

    tmp_dir = Path(tempfile.mkdtemp(prefix="sticker_engine_"))
    silent = tmp_dir / "video_only.mp4"

    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            for rgba, x, y in renderer.get_overlays(t):
                h, w = rgba.shape[:2]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(width, x + w), min(height, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
                alpha = sub[..., 3:4].astype(np.float32) / 255.0
                sprite_bgr = sub[..., :3][..., ::-1].astype(np.float32)
                region = frame[y0:y1, x0:x1].astype(np.float32)
                frame[y0:y1, x0:x1] = (region * (1 - alpha) + sprite_bgr * alpha).astype(np.uint8)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    cmd = ["ffmpeg", "-y", "-i", str(silent), "-i", str(video_in),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-shortest", str(video_out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg упал: {result.stderr[-2000:]}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from . import plan_stickers

    parser = argparse.ArgumentParser(prog="sticker_engine.cli")
    parser.add_argument("--cues", required=True)
    parser.add_argument("--video")
    parser.add_argument("--out")
    parser.add_argument("--zones")
    parser.add_argument("--plan-json")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--size")
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.dry_run and (not args.video or not args.out):
        parser.error("--video и --out обязательны, кроме режима --dry-run")

    config = StickerConfig.from_json(Path(args.config)) if args.config else StickerConfig()

    cues = load_word_cues(Path(args.cues))
    zones = load_zones(Path(args.zones) if args.zones else None)

    video = Path(args.video) if args.video else None

    if args.size:
        w_str, h_str = args.size.lower().split("x")
        output_w, output_h = int(w_str), int(h_str)
    elif video is not None and (probed := _probe_size(video)):
        output_w, output_h = probed
    else:
        output_w, output_h = 1080, 1920

    if args.duration is not None:
        duration = args.duration
    elif video is not None and (probed_d := _probe_duration(video)) is not None:
        duration = probed_d
    elif cues:
        duration = cues[-1].end + config.edge_padding_tail_sec
    else:
        duration = 0.0

    placed = plan_stickers(cues, duration_sec=duration, output_w=output_w,
                           output_h=output_h, forbidden_zones=zones, config=config)
    print(f"запланировано стикеров: {len(placed)}")

    if args.plan_json:
        plan_to_json(placed, Path(args.plan_json))
        print(f"план записан в {args.plan_json}")

    if args.dry_run:
        return 0

    try:
        burn_stickers(video, Path(args.out), placed, config)
    except Exception as e:
        print(f"ошибка рендера: {e}", file=sys.stderr)
        return 1

    print(f"готово: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

