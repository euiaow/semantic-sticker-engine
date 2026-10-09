"""
Предзагрузка ML-модели в локальный кэш (ТЗ §5.1).

    python scripts/prefetch_model.py [--model ...] [--cache-dir ...]

В рантайме сеть запрещена, поэтому модель обязана быть скачана заранее —
этим скриптом на этапе сборки образа.

Печатает итоговый путь кэша и суммарный размер — это попадает в README.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
