"""Chequeos de consistencia del dataset de evaluación (no requieren modelos)."""
import json
from collections import Counter

from src.config import HELDOUT_PATH, KB_PATH, TEST_QUERIES_PATH

DATA = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
ARTICULOS = {c["articulo"] for c in json.loads(KB_PATH.read_text(encoding="utf-8"))}
CASOS = [c for k in ("preguntas_reglamento", "fuera_de_tema", "prompt_injection") for c in DATA[k]]


def test_ids_unicos_y_totales():
    ids = [c["id"] for c in CASOS]
    assert len(ids) == len(set(ids))
    for k, n in DATA["metadata"]["totales"].items():
        assert len(DATA[k]) == n


def test_split_dev_test_estratificado():
    for k in ("preguntas_reglamento", "fuera_de_tema", "prompt_injection"):
        cnt = Counter(c["split"] for c in DATA[k])
        assert set(cnt) == {"dev", "test"}
        assert abs(cnt["dev"] - cnt["test"]) <= 1


def test_articulos_esperados_existen_en_la_base():
    for c in DATA["preguntas_reglamento"]:
        assert set(c["articulos_esperados"]) <= ARTICULOS, c["id"]
        assert (c["cobertura"] == "ninguna") == (not c["articulos_esperados"]), c["id"]


def test_heldout_consistente_y_sin_ids_repetidos():
    ho = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    casos = [c for k in ("preguntas_reglamento", "fuera_de_tema", "prompt_injection") for c in ho[k]]
    ids = [c["id"] for c in casos]
    assert len(ids) == len(set(ids)) and not set(ids) & {c["id"] for c in CASOS}
    assert all(c["split"] == "heldout" for c in casos)
    for c in ho["preguntas_reglamento"]:
        assert set(c["articulos_esperados"]) <= ARTICULOS, c["id"]