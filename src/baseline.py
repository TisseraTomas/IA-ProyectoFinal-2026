"""Baseline sin IA: coincidencia de palabras clave ponderada por IDF (sin embeddings)."""
from __future__ import annotations

import json
import math
import re

from src.config import KB_PATH
from src.guardrail import normalize

_STOP = set(
    "el la los las un una unos unas de del al a en y o u que q por pa para pq con sin se me mi mis te tu tus "
    "su sus es son ser soy sos hay ha han he como cual cuales cuanto cuantos cuanta cuantas cuando donde quien "
    "puedo puede pueden podes tengo tiene tienen tener esta estan eso esto ese esa lo le les si no ni ya mas "
    "muy tambien hola che x d".split()
)


def tokens(text: str) -> list[str]:
    """Normaliza (minúsculas, sin acentos), quita stopwords y 'stemmea' cortando a 5 letras."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [w[:5] for w in words if len(w) > 2 and w not in _STOP]


class KeywordBaseline:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or json.loads(KB_PATH.read_text(encoding="utf-8"))
        self.docs = [set(tokens(c["texto_indexable"])) for c in self.chunks]
        n = len(self.docs)
        df: dict[str, int] = {}
        for d in self.docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(n + 1) + 1.0  # palabras que no existen en el reglamento

    def _w(self, t: str) -> float:
        return self.idf.get(t, self.max_idf)

    def rank(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        """Devuelve [(chunk, distancia)], con distancia = 1 - (peso IDF de lo que coincide / peso total)."""
        q = set(tokens(query))
        if not q:
            return [(self.chunks[0], 1.0)]
        total = sum(self._w(t) for t in q)
        scored = [(c, 1.0 - sum(self._w(t) for t in q & d) / total) for c, d in zip(self.chunks, self.docs)]
        return sorted(scored, key=lambda x: x[1])[:k]

    def distance(self, query: str) -> float:
        return self.rank(query, 1)[0][1]