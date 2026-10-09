"""
Матчер: фраза (окно контекста вокруг слова) -> эмодзи.

Только косинусная близость к предрасчитанным эмбеддингам каталога.
Никакого обучения, никакой сети в рантайме, никаких LLM (ТЗ §5).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import string
from .catalog import EmojiCatalog
from .types import StickerConfig, WordCue
from .stopwords import STOP_WORDS
import numpy as np

logger = logging.getLogger("sticker_engine.matcher")

__all__ = ["EmbeddingMatcher", "build_phrases", "PhraseCandidate", "split_into_phrases"]

# Порог паузы между словами, по которому режем транскрипт на фразы (сек).
PHRASE_PAUSE_THRESHOLD = 0.35

# Минимальная длина слова-кандидата в символах.
MIN_TRIGGER_WORD_LEN = 4


class PhraseCandidate:
    """Слово-кандидат вместе с окном контекста, по которому идёт матчинг."""

    def __init__(self, word_index: int, word: str, phrase: str,
                 t_start: float, t_end: float) -> None:
        self.word_index = word_index
        self.word = word
        self.phrase = phrase
        self.t_start = t_start
        self.t_end = t_end


def split_into_phrases(word_cues: Sequence[WordCue],
                       pause_threshold: float = PHRASE_PAUSE_THRESHOLD) -> List[List[int]]:
    """
    Разрезать транскрипт на фразы по паузам. Возвращает список фраз,
    каждая — список индексов в исходном word_cues.
    """
    if not word_cues:
        return []

    phrases: List[List[int]] = []
    current: List[int] = [0]

    for i in range(1, len(word_cues)):
        pause = word_cues[i].start - word_cues[i - 1].end
        if pause > pause_threshold:
            phrases.append(current)
            current = [i]
        else:
            current.append(i)

    phrases.append(current)
    return phrases


def build_phrases(word_cues: Sequence[WordCue],
                  config: StickerConfig) -> List[PhraseCandidate]:
    """
    Построить кандидатов: для каждого содержательного слова — окно
    +-config.phrase_window слов В ПРЕДЕЛАХ ЕГО ФРАЗЫ.

    Окно — ±config.phrase_window слов С КАЖДОЙ стороны, то есть максимум
    2 * phrase_window + 1 слов, склеенных пробелом.

    Отсеиваются (ТЗ §5.2):
      * слова короче MIN_TRIGGER_WORD_LEN;
      * стоп-слова (sticker_engine.stopwords);
      * числительные — то есть токены, целиком состоящие из цифр и пунктуации
        ("2024", "15%", "3,5"). Словесные числительные ("три", "первый")
        числительными НЕ считаются и кандидатами быть могут.
    """
    def is_number(word: str) -> bool:
        return (all(ch.isdigit() or ch in string.punctuation for ch in word)
                and any(ch.isdigit() for ch in word))

    phrases_ids = split_into_phrases(word_cues)
    candidates: List[PhraseCandidate] = []

    for phrase in phrases_ids:
        for pos, global_idx in enumerate(phrase):
            word_text = word_cues[global_idx].word

            if len(word_text) < MIN_TRIGGER_WORD_LEN:
                continue
            if word_text in STOP_WORDS:  # <- поправим имя после stopwords.py
                continue
            if is_number(word_text):
                continue

            lo = max(0, pos - config.phrase_window)
            hi = min(len(phrase) - 1, pos + config.phrase_window)
            window_global_indices = phrase[lo:hi + 1]

            phrase_text = " ".join(word_cues[i].word for i in window_global_indices)

            candidates.append(PhraseCandidate(
                word_index=global_idx,
                word=word_text,
                phrase=phrase_text,
                t_start=word_cues[global_idx].start,
                t_end=word_cues[global_idx].end,
            ))

    return candidates


                

_model_cache: dict[str, "SentenceTransformer"] = {}

def _get_model(model_name: str, cache_dir=None):
    if model_name not in _model_cache:
        logger.info("loading model %s ...", model_name)
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name, cache_folder=cache_dir)
    return _model_cache[model_name]

class EmbeddingMatcher:
    """
    Реализация протокола types.Matcher поверх sentence-transformers.

    Модель грузится ЛЕНИВО и кэшируется на уровне модуля: повторное создание
    EmbeddingMatcher в том же процессе не должно перегружать веса.
    """

    def __init__(self, catalog: EmojiCatalog, config: StickerConfig) -> None:
        self.catalog = catalog
        self.config = config
        self.model = _get_model(config.model_name, config.model_cache_dir)

    def match(self, phrase: str) -> Optional[Tuple[str, str, float]]:
        """
        Подобрать эмодзи для фразы.

        Returns:
            (emoji, category, score) или None если:
              * score < config.min_score;
              * категория в config.blocked_categories;
              * разрыв до второго кандидата < config.min_margin_over_runner_up;
              * модель/каталог недоступны.
        """
        return self.match_batch([phrase])[0]

    def match_batch(self, phrases: Sequence[str]) -> List[Optional[Tuple[str, str, float]]]:
        """
        Батчевый вариант — им и надо пользоваться в planner (одна фраза за вызов
        модели это медленно). Порядок результатов соответствует входу.
        """
        if not phrases:
            return []

        phrase_embeddings = self.model.encode(list(phrases), normalize_embeddings=True)
        similarities = phrase_embeddings @ self.catalog.embeddings.T  # (N_фраз, N_каталог)

        results = []
        for i in range(len(phrases)):
            top = np.argsort(similarities[i])[::-1][:2]
            best_idx = top[0]
            score = float(similarities[i][best_idx])
            entry = self.catalog.entries[best_idx]

            margin = score - float(similarities[i][top[1]]) if len(top) > 1 else 1.0

            if (score < self.config.min_score
                    or entry.category in self.config.blocked_categories
                    or margin < self.config.min_margin_over_runner_up):
                results.append(None)
            else:
                results.append((entry.emoji, entry.category, score))

        return results
