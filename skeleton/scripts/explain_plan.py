"""Показать все фразы-кандидаты и что матчер по ним решил."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sticker_engine.catalog import EmojiCatalog
from sticker_engine.cli import load_word_cues
from sticker_engine.matcher import EmbeddingMatcher, build_phrases
from sticker_engine.types import StickerConfig

cues = load_word_cues(Path(sys.argv[1]))
cfg = StickerConfig()
cands = build_phrases(cues, cfg)
matcher = EmbeddingMatcher(EmojiCatalog.load(cfg), cfg)
results = matcher.match_batch([c.phrase for c in cands])

for c, r in zip(cands, results):
    verdict = f"{r[0]}  {r[2]:.3f}  [{r[1]}]" if r else "—"
    print(f"{c.t_start:6.2f}  {c.word:18} {verdict:24} | {c.phrase}")