"""
Генератор тестовых фикстур (запускается заказчиком, исполнителю трогать не нужно).

    python fixtures/_gen_fixtures.py

Создаёт:
    word_cues.sample.json  — синтетический пословный транскрипт 32 c
    zones.sample.json      — запретные зоны для кадра 1080x1920
    sample.mp4             — тестовое видео 1080x1920 / 30 fps / 32 c со звуком
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Фразы устной русской речи. Между фразами пауза 0.5 c — планировщик режет по ней.
PHRASES = [
    "и вот здесь начинается самое интересное честно говоря",
    "мы заработали первый миллион за три месяца работы",
    "тогда мне было по-настоящему страшно взять эту ответственность",
    "но идея оказалась рабочей и команда её вытянула",
    "дальше пошёл кратный рост выручки каждый квартал",
    "в общем я хочу сказать вам следующее",
    "деньги приходят туда где есть система а не героизм",
    "время это единственный ресурс который нельзя докупить",
    "и на этом мы сегодня закончим спасибо что досмотрели",
]

WORD_DUR = 0.32          # длительность слова
WORD_GAP = 0.04          # микропауза внутри фразы
PHRASE_GAP = 0.50        # пауза между фразами (> 0.35 -> граница фразы)
DURATION = 32.0
FPS = 30


def build_cues() -> list[dict]:
    cues: list[dict] = []
    t = 0.6
    for phrase in PHRASES:
        for word in phrase.split():
            cues.append({"start": round(t, 3),
                         "end": round(t + WORD_DUR, 3),
                         "word": word})
            t += WORD_DUR + WORD_GAP
        t += PHRASE_GAP
    return cues


def build_zones() -> list[dict]:
    """Хедер 0-3 c, субтитры весь клип, лицо — 2 сэмпла/сек с дрейфом, b-roll 12.0-14.5 c."""
    zones: list[dict] = [
        {"t_start": 0.0, "t_end": 3.0, "kind": "header",
         "rect": {"x": 0, "y": 120, "w": 1080, "h": 260}},
        {"t_start": 0.0, "t_end": DURATION, "kind": "subtitle",
         "rect": {"x": 60, "y": 1360, "w": 960, "h": 380}},
        {"t_start": 12.0, "t_end": 14.5, "kind": "broll",
         "rect": {"x": 0, "y": 0, "w": 1080, "h": 1920}},
        {"t_start": 29.0, "t_end": DURATION, "kind": "endscreen",
         "rect": {"x": 0, "y": 0, "w": 1080, "h": 1920}},
        {"t_start": 0.0, "t_end": DURATION, "kind": "custom",
         "rect": {"x": 0, "y": 1840, "w": 1080, "h": 80}},
    ]
    step = 0.5
    t = 0.0
    i = 0
    while t < DURATION:
        # лицо медленно ездит по горизонтали в верхней половине кадра
        x = 300 + int(120 * ((i % 8) - 4) / 4)
        zones.append({
            "t_start": round(t, 3), "t_end": round(t + step, 3), "kind": "face",
            "rect": {"x": x, "y": 520, "w": 460, "h": 520},
        })
        t += step
        i += 1
    return zones


def build_video(out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg не найден — sample.mp4 не создан")
        return
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gradients=size=1080x1920:rate={FPS}:duration={DURATION}:speed=0.05",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={DURATION}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-shortest", str(out),
    ], check=True)


def main() -> None:
    cues = build_cues()
    assert cues[-1]["end"] < DURATION, f"транскрипт длиннее клипа: {cues[-1]['end']}"
    (HERE / "word_cues.sample.json").write_text(
        json.dumps(cues, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "zones.sample.json").write_text(
        json.dumps(build_zones(), ensure_ascii=False, indent=1), encoding="utf-8")
    build_video(HERE / "sample.mp4")
    print(f"слов: {len(cues)}, конец транскрипта: {cues[-1]['end']} c")


if __name__ == "__main__":
    main()
