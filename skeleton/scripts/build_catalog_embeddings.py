"""
Одноразовый предрасчёт эмбеддингов каталога (ТЗ §4.3).

    python scripts/build_catalog_embeddings.py [--catalog assets/emoji_catalog.json]

Пишет:
    assets/catalog_embeddings.npy        — (N, D) float32, L2-нормированные,
                                           порядок строк == порядок записей каталога
    assets/catalog_embeddings.meta.json  — {"model_name", "dim", "count", "catalog_sha256"}

Рантайм эти эмбеддинги только читает и никогда не пересчитывает при совпадении меты.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def catalog_sha256(catalog_path: Path) -> str:
    """Хэш содержимого файла каталога — по нему рантайм поймёт, что каталог изменился."""
    return hashlib.sha256(catalog_path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="assets/emoji_catalog.json")
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog)
    out_dir = Path(args.out_dir)
    entries = json.loads(catalog_path.read_text(encoding="utf-8"))

    texts = [entry["ru"] for entry in entries]  # порядок совпадает с порядком entries

    print(f"loading model {args.model_name} ...")
    model = SentenceTransformer(args.model_name)

    print(f"encoding {len(texts)} descriptions ...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # L2-нормировка -> косинус = просто скалярное произведение
        show_progress_bar=True,
    ).astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "catalog_embeddings.npy", embeddings)

    meta = {
        "model_name": args.model_name,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "catalog_sha256": catalog_sha256(catalog_path),
    }
    (out_dir / "catalog_embeddings.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"saved {embeddings.shape} -> {out_dir / 'catalog_embeddings.npy'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
