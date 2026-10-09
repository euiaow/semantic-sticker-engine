"""Конвертер whisper json -> word_cues.json (формат ТЗ)."""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

data = json.loads(src.read_text(encoding="utf-8"))
cues = []
for seg in data["segments"]:
    for w in seg.get("words", []):
        cues.append({
            "start": float(w["start"]),
            "end": float(w["end"]),
            "word": w["word"].strip(),
        })

dst.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(cues)} слов -> {dst}")