"""Garantías sobre la base de conocimiento (data/reglamento_chunks.json)."""
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "reglamento_chunks.json"


@pytest.fixture(scope="module")
def chunks():
    return json.loads(DATA.read_text(encoding="utf-8"))


def by_id(chunks):
    return {c["id"]: c for c in chunks}


def test_cantidad_y_unicidad(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert 80 <= len(chunks) <= 120


def test_metadata_obligatoria(chunks):
    for c in chunks:
        for k in ("id", "titulo_numero", "titulo", "articulo", "articulo_nombre", "texto_literal", "texto_indexable"):
            assert c[k] not in (None, ""), (c["id"], k)
        assert c["texto_indexable"].endswith(c["texto_literal"])


def test_cubre_titulos_I_a_X(chunks):
    assert {c["titulo_numero"] for c in chunks} == set(range(1, 11))


@pytest.mark.parametrize("chunk_id, fragmentos", [
    ("art_6.2", ["55% al 59%", "cuatro (4)"]),
    ("art_6.4", ["72 hs. hábiles"]),
    ("art_4.1__requisitos", ["33% de aplazos", "no aplica en las 6 materias finales", "más de 3 contribuciones"]),
    ("art_7.2.2.1__plazo_vigencia", ["25 meses", "dos años"]),
    ("art_7.2.2.1__perdida_regularidad", ["Tres aplazos"]),
    ("art_7.2.2.2__condiciones_promocion", ["nota igual o superior a 6", "nota igual o superior a 7"]),
    ("art_7.2.2.3", ["nota igual o superior a 7"]),
    ("art_7.4.1__requisitos_para_rendir", ["15 min", "60 minutos", "D.N.I."]),
    ("art_7.4.1__requisitos_inscripcion", ["24 hs hábiles"]),
    ("art_7.4.3__topes_maximos", ["20%", "10%", "5%"]),
    ("art_2.3__condiciones", ["50%", "cinco años"]),
    ("art_4.5__actuaciones", ["Trámites de egreso ❌", "Trámites administrativos ✅"]),
    ("art_4.6.2__situacion_y_requisitos", ["ninguna oferta académica"]),
    ("art_4.4.2__situacion_y_requisitos", ["a) Falta de reinscripción anual", "b) Falta de cumplimiento en tiempo y forma",
                                           "c) Falta de cumplimiento de alguna", "a y b) el estudiante debe realizar"]),
])
def test_datos_duros_textuales(chunks, chunk_id, fragmentos):
    texto = by_id(chunks)[chunk_id]["texto_literal"]
    for f in fragmentos:
        assert f in texto, f"{chunk_id}: falta {f!r}"


def test_tablas_de_actuaciones_completas(chunks):
    for c in chunks:
        if c["parte"] == "actuaciones":
            filas = c["texto_literal"].split("\n")[1:]
            assert len(filas) >= 17, c["id"]
            assert all(any(m in f for m in ("✅", "❌", "N/A", "condicionado")) for f in filas), c["id"]