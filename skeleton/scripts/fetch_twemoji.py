"""
Сборка PNG-ассетов из Twemoji (ТЗ §4.1).

    python scripts/fetch_twemoji.py --catalog assets/emoji_catalog.json

Для каждой записи каталога:
  1. вычислить codepoint-имя (напр. "1f4b0", составные — через дефис,
     variation selector FE0F по правилам Twemoji);
  2. взять SVG из локального клона twemoji (--twemoji-dir) либо из скачанного
     релизного архива — НЕ ходить в сеть на каждый запуск;
  3. отрендерить в PNG 512x512 RGBA (cairosvg / rsvg-convert), без апскейла из 72x72;
  4. положить в assets/stickers/<name>.png;
  5. записать/обновить assets/stickers/ATTRIBUTION.md (Twemoji, CC-BY 4.0).

Скрипт идемпотентен: повторный запуск не перекачивает и не перерисовывает готовое.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cairosvg

ZWJ = "\u200d"
VS16 = "\ufe0f"

def emoji_to_codepoint(emoji: str) -> str:
    """Twemoji-style codepoint-имя без расширения, напр. '1f4b0' или '1f468-200d-1f4bb'."""
    if ZWJ not in emoji:
        emoji = emoji.replace(VS16, "")
    return "-".join(f"{ord(ch):x}" for ch in emoji)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="путь к emoji_catalog.json")
    parser.add_argument("--twemoji-dir", required=True, help="путь к локальному клону/архиву twemoji")
    parser.add_argument("--out-dir", default="assets/stickers", help="куда класть PNG")
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog)
    svg_dir = Path(args.twemoji_dir) / "assets" / "svg"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads(catalog_path.read_text(encoding="utf-8"))

    ok, skipped_existing, missing_svg = 0, 0, []

    for entry in entries:
        codepoint = emoji_to_codepoint(entry["emoji"])
        svg_path = svg_dir / f"{codepoint}.svg"
        png_path = out_dir / f"{codepoint}.png"

        if png_path.exists():
            skipped_existing += 1
            continue

        if not svg_path.exists():
            missing_svg.append((entry["emoji"], codepoint))
            continue

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=512,
            output_height=512,
        )
        ok += 1

    attribution_path = out_dir / "ATTRIBUTION.md"
    if not attribution_path.exists():
        attribution_path.write_text(ATTRIBUTION_TEXT, encoding="utf-8")

    print(f"rendered: {ok}, already existed: {skipped_existing}, missing svg: {len(missing_svg)}")
    for emoji, codepoint in missing_svg:
        print(f"  MISSING: {emoji} -> {codepoint}.svg")

    return 1 if missing_svg else 0



if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
