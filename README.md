# sticker_engine — «Стикеры по словам»

Python-библиотека, которая по пословному транскрипту видео решает, на каких словах показать эмодзи-стикер, каким именно и где на кадре его разместить, не задевая запретные зоны (лицо, субтитры, хедер, b-roll).

Реализация ТЗ `TZ_STICKERS.md`. Отчёт по подбору порогов — `REPORT.md`.

---

## Требования

- Python 3.10+
- ffmpeg и ffprobe в `PATH` (нужны для CLI-демо и `tests/test_cli_e2e.py`)
- Для пересборки PNG-ассетов дополнительно: нативная библиотека Cairo (см. «Пересборка ассетов»)

---

## Установка

```bash
cd skeleton
pip install -r requirements.txt
pip install pytest                      # тестовый инструмент, в requirements не входит
```

Для пересборки ассетов дополнительно:

```bash
pip install -r requirements-build.txt
```

---

## Предзагрузка модели

Рантайм не ходит в сеть (§11), модель должна лежать в локальном кэше HuggingFace:

```bash
cd skeleton
python scripts/prefetch_model.py
```

Модель: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~470 МБ, CPU).
Путь кэша задаётся переменной окружения `HF_HOME` или полем `StickerConfig.model_cache_dir`.

В проде рекомендуется выставить `HF_HUB_OFFLINE=1` — иначе `sentence-transformers` при каждой инициализации сверяет версии файлов по сети, даже когда они уже в кэше (см. §9 отчёта).

---

## Пересборка ассетов

Нужна только при изменении `assets/emoji_catalog.json`. В сдаче все ассеты уже собраны.

### PNG-стикеры

Требуется локальный клон Twemoji (SVG, не PNG 72×72 — апскейл запрещён §4.1):

```bash
git clone --depth 1 https://github.com/twitter/twemoji.git twemoji-src
```

На Windows дополнительно нужна нативная Cairo для `cairosvg`:

```powershell
winget install --id=tschoonj.GTKForWindows -e
```

Затем (из `skeleton/`, после установки — перезапустить терминал):

```bash
python scripts/fetch_twemoji.py --catalog assets/emoji_catalog.json --twemoji-dir twemoji-src
```

Скрипт идемпотентен — уже готовые PNG не перерисовываются.

### Эмбеддинги каталога

**Запускать после каждого изменения каталога:**

```bash
cd skeleton
python scripts/build_catalog_embeddings.py
```

Создаёт `assets/catalog_embeddings.npy` и `assets/catalog_embeddings.meta.json` (имя модели, размерность, число записей, `sha256` каталога).

Рантайм эмбеддинги не пересчитывает: при старте сверяет `model_name` и `catalog_sha256` из меты. При рассогласовании — WARNING в лог и пересчёт на лету (медленно, но не падает).

---

## Тесты

```bash
cd <корень пакета>          # уровень выше skeleton/
python -m pytest tests/ -v
```

**Ожидаемый результат: 76 passed, 2 failed, 0 skipped.**

Два теста падают **осознанно**, это не недоделка:

- `test_matcher_quality.py::test_precision_and_recall`
- `test_matcher_quality.py::test_negatives_produce_nothing`

Планка §13.2 (`precision ≥ 0.90 при recall ≥ 0.35`) на текущем eval-наборе недостижима ни при какой комбинации порогов. Согласно §13.2, исполнитель обязан остановиться и передать цифры заказчику до сдачи, а не подгонять молча. Полный разбор — 32 замеренные конфигурации, анализ причин, предложения — в **`REPORT.md`**.

Скипы возможны только при отсутствии ffmpeg в `PATH` (`tests/test_cli_e2e.py`).

---

## CLI-демо

Запускать из `skeleton/`. Это демо-стенд и приёмочный инструмент; прод-путь — `plan_stickers` + `StickerRenderer` внутри пайплайна заказчика (§14).

### Полный рендер

```bash
cd skeleton
python -m sticker_engine.cli \
  --video ../fixtures/sample.mp4 \
  --cues  ../fixtures/word_cues.sample.json \
  --zones ../fixtures/zones.sample.json \
  --out   ../out.mp4
```

### Только план, без видео и ffmpeg

```bash
python -m sticker_engine.cli \
  --cues ../fixtures/word_cues.sample.json \
  --plan-json ../plan.json \
  --dry-run
```

### Аргументы

| Аргумент | Обязательность | Назначение |
|---|---|---|
| `--cues` | всегда | пословный транскрипт JSON |
| `--video` | кроме `--dry-run` | исходное видео |
| `--out` | кроме `--dry-run` | путь выходного mp4 |
| `--zones` | опционально | запретные зоны JSON |
| `--plan-json` | опционально | дамп плана в JSON |
| `--duration` | опционально | длительность клипа в секундах |
| `--size WxH` | опционально | размер выходного кадра |
| `--config` | опционально | JSON со `StickerConfig` |
| `--dry-run` | опционально | только план, без рендера |

`duration_sec` определяется по правилу §10: `--duration`, иначе фактическая длительность `--video` через ffprobe, иначе `word_cues[-1].end + edge_padding_tail_sec`.

Размер кадра: `--size`, иначе из `--video`, иначе `1080x1920`.

---

## Формат входных данных

### word_cues.json

```json
[
  {"start": 0.12, "end": 0.44, "word": "мы"},
  {"start": 0.44, "end": 0.98, "word": "заработали"}
]
```

Допускается ключ `"text"` вместо `"word"`. Тайминги — секунды от начала клипа, float.

### zones.json

```json
[
  {"t_start": 0.0, "t_end": 3.0, "kind": "header",
   "rect": {"x": 0, "y": 120, "w": 1080, "h": 260}}
]
```

`kind`: `face` | `subtitle` | `header` | `broll` | `endscreen` | `custom`.
Зоны `broll` и `endscreen` — эксклюзивные: события в их окне удаляются целиком (§7.1).

---

## Программный API

```python
from sticker_engine import StickerRenderer, StickerConfig, plan_stickers

placed = plan_stickers(
    word_cues,
    duration_sec=42.0,
    output_w=1080,
    output_h=1920,
    forbidden_zones=zones,
    config=StickerConfig(),
)

renderer = StickerRenderer(placed, 1080, 1920, config)
for frame_idx in range(n_frames):
    for rgba, x, y in renderer.get_overlays(frame_idx / fps):
        ...  # альфа-блендинг делает вызывающая сторона
```

`plan_stickers` вызывается один раз на клип, `get_overlays` — на каждом кадре (до 3000 раз) и оптимизирован под это: предрендер спрайтов в `__init__`, кэш по квантованным (scale, alpha), словарный лукап без аллокаций.

Библиотека не бросает исключений наружу: любая внутренняя ошибка — WARNING в логгер `sticker_engine` и пустой результат.

---

## Структура сдачи

```
skeleton/
  sticker_engine/          реализованный модуль
  scripts/
    fetch_twemoji.py               сборка PNG из Twemoji SVG
    build_catalog_embeddings.py    предрасчёт эмбеддингов каталога
    prefetch_model.py              предзагрузка модели в кэш
    tune_threshold.py              подбор порогов (материал для REPORT.md)
    tune_lemmatize.py              замер эффекта лемматизации (§15.2)
    explain_plan.py                диагностика: что матчер решил по каждому слову
    bench.py                       замеры под §11
    whisper_to_cues.py             конвертер whisper json -> word_cues.json
  assets/
    emoji_catalog.json             каталог (200 записей)
    stickers/*.png                 PNG 512x512 + ATTRIBUTION.md
    catalog_embeddings.npy         предрасчитанные эмбеддинги
    catalog_embeddings.meta.json   мета для инвалидации кэша
tests/                     приёмочные тесты (не изменялись)
fixtures/
  eval_set.json            eval-набор, 160 фраз
REPORT.md                  отчёт по подбору порогов (§13)
README.md                  этот файл
```

---

## Известные ограничения

1. **Планка §13.2 не достигнута.** Подробности и предложения — в `REPORT.md`, §1 и §10.

2. **Пороги в конфиге — дефолтные** (`min_score = 0.45`, `min_margin_over_runner_up = 0.03`). Рекомендуемое значение margin — 0.12, но оно ломает `test_stickers_actually_burned` на синтетической фикстуре. Требует совместного решения, см. `REPORT.md`, §2.

3. **Демо-ролики.** Система проверена на фрагменте реального русского подкаста (разбор в `REPORT.md`, §8). Полный комплект из 3–5 роликов на разнообразном материале не подготовлен — не было исходного видеоматериала с пословным транскриптом. Для генерации транскрипта под свой материал приложен `scripts/whisper_to_cues.py`.

4. **Вопросы §15.1 и §15.3** (выбор модели, оценка сроков по разделам) остались без письменного ответа. Вопрос §15.2 о лемматизации закрыт замерами в `REPORT.md`, §5.
