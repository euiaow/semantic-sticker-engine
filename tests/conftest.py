"""
Общие фикстуры приёмочных тестов.

Тесты запускаются из корня пакета:  pytest tests/
Скелет реализации ищется в ./skeleton (или в установленном пакете sticker_engine).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKELETON = ROOT / "skeleton"
FIXTURES = ROOT / "fixtures"

if SKELETON.exists() and str(SKELETON) not in sys.path:
    sys.path.insert(0, str(SKELETON))

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)

requires_model = pytest.mark.skipif(
    not (ROOT / "skeleton" / "assets" / "catalog_embeddings.npy").exists(),
    reason="эмбеддинги каталога не собраны (scripts/build_catalog_embeddings.py)",
)
# NB: наличие локального кэша весов модели этим маркером не проверяется —
# следи сам, чтобы scripts/prefetch_model.py был отработан. Сеть в рантайме
# запрещена (ТЗ §5.1), и молчаливый поход в интернет здесь не поймается.


# --------------------------------------------------------------------------- #
#  Данные                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_cues():
    """Пословный транскрипт из fixtures/word_cues.sample.json."""
    from sticker_engine.types import WordCue

    data = json.loads((FIXTURES / "word_cues.sample.json").read_text(encoding="utf-8"))
    return [WordCue(start=float(d["start"]), end=float(d["end"]), word=d["word"])
            for d in data]


@pytest.fixture
def sample_zones():
    """Запретные зоны из fixtures/zones.sample.json."""
    from sticker_engine.types import ForbiddenZone, Rect

    data = json.loads((FIXTURES / "zones.sample.json").read_text(encoding="utf-8"))
    out = []
    for d in data:
        r = d["rect"]
        out.append(ForbiddenZone(
            t_start=float(d["t_start"]), t_end=float(d["t_end"]),
            rect=Rect(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])),
            kind=str(d.get("kind", "custom")),
        ))
    return out


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """
    Тестовое видео 1080x1920, 32 c, 30 fps, со звуком.
    Берётся из fixtures/sample.mp4, а если его нет — генерируется ffmpeg'ом
    (длительность обязана покрывать транскрипт, который идёт до 30.12 c).
    """
    src = FIXTURES / "sample.mp4"
    if src.exists():
        return src
    if FFMPEG is None:
        pytest.skip("нет fixtures/sample.mp4 и нет ffmpeg для генерации")
    out = tmp_path / "sample.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "gradients=size=1080x1920:rate=30:duration=32:speed=0.05",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=32",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out),
    ], check=True)
    return out


@pytest.fixture
def cfg():
    """Дефолтный конфиг с ассетами скелета."""
    from sticker_engine.types import StickerConfig

    return StickerConfig(
        assets_dir=SKELETON / "assets" / "stickers",
        catalog_path=SKELETON / "assets" / "emoji_catalog.json",
    )


# --------------------------------------------------------------------------- #
#  Стаб-матчер: тесты планировщика не должны тянуть ML-модель                   #
# --------------------------------------------------------------------------- #

class StubMatcher:
    """
    Детерминированный матчер по подстрокам. Реализует types.Matcher.

    rules: список (подстрока, эмодзи, категория, score).
    """

    DEFAULT_RULES = [
        ("миллион", "💰", "money", 0.92),
        ("деньг", "💰", "money", 0.88),
        ("рост", "📈", "growth", 0.81),
        ("страшно", "😱", "emotion_negative", 0.79),
        ("идея", "💡", "idea", 0.76),
        ("время", "⏰", "time", 0.71),
        ("огонь", "🔥", "hype", 0.95),
        ("нож", "🔪", "weapons", 0.99),
    ]

    def __init__(self, rules=None, blocked=()):
        self.rules = list(rules if rules is not None else self.DEFAULT_RULES)
        self.blocked = set(blocked)
        self.calls = 0

    def match(self, phrase: str):
        self.calls += 1
        low = (phrase or "").lower()
        best = None
        for sub, emoji, cat, score in self.rules:
            if sub in low and (best is None or score > best[2]):
                best = (emoji, cat, score)
        if best is None or best[1] in self.blocked:
            return None
        return best

    def match_batch(self, phrases):
        return [self.match(p) for p in phrases]


@pytest.fixture
def stub_matcher():
    return StubMatcher()


@pytest.fixture
def stub_asset_path(tmp_path):
    """
    Callable emoji -> путь к PNG. Создаёт реальные 512x512 RGBA PNG на лету,
    чтобы рендерер мог их открыть.
    """
    from PIL import Image

    created: dict[str, str] = {}

    def _path(emoji: str):
        if emoji not in created:
            p = tmp_path / f"{ord(emoji[0]):x}.png"
            img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            # непрозрачный круг в центре — есть что искать пиксельным тестам
            from PIL import ImageDraw
            ImageDraw.Draw(img).ellipse((40, 40, 471, 471), fill=(255, 64, 0, 255))
            img.save(p)
            created[emoji] = str(p)
        return created[emoji]

    return _path


# --------------------------------------------------------------------------- #
#  Хелперы построения объектов                                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def make_event(stub_asset_path, cfg):
    """Фабрика StickerEvent с валидными таймингами."""
    from sticker_engine.types import StickerEvent

    def _make(t_start: float, emoji: str = "🔥", score: float = 0.9,
              word: str = "огонь", index: int = 0, category: str = "hype"):
        return StickerEvent(
            t_anchor=t_start,
            t_start=t_start,
            t_end=t_start + cfg.total_duration,
            emoji=emoji,
            asset_path=stub_asset_path(emoji),
            trigger_word=word,
            trigger_word_index=index,
            phrase=f"... {word} ...",
            score=score,
            category=category,
        )

    return _make
