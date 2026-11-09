"""Pruebas del guardrail. Las de reglas no requieren modelos; las de embeddings sí (ver requirements.txt)."""
import pytest

from src.baseline import KeywordBaseline
from src.config import TOP_K
from src.evaluate import calibrate, confusion, hit_at_k, load_cases, _row
from src.guardrail import detect_injection

CASES = load_cases()


# ---------------------------------------------------------------- capa 1 (offline)
def test_reglas_bloquean_todos_los_prompt_injection():
    sin_detectar = [c["id"] for c in CASES if c["grupo"] == "prompt_injection" and not detect_injection(c["texto"])]
    assert not sin_detectar, f"Ataques sin detectar: {sin_detectar}"


def test_reglas_no_bloquean_consultas_legitimas():
    falsos = [c["id"] for c in CASES if c["grupo"] != "prompt_injection" and detect_injection(c["texto"])]
    assert not falsos, f"Falsos positivos de las reglas: {falsos}"


# ---------------------------------------------------------------- capa 2 (requiere modelo)
@pytest.fixture(scope="module")
def guardrail(tmp_path_factory):
    from src.guardrail import IntentGuardrail
    from src.vector_store import build_vector_store

    db = build_vector_store(persist_directory=tmp_path_factory.mktemp("chroma"))
    return IntentGuardrail(db, k=TOP_K)


def test_guardrail_performance(guardrail):
    """Umbral calibrado en DEV; las métricas se exigen en TEST (sin fuga de información)."""
    rows = [_row(c, guardrail.score(c["texto"])) for c in CASES]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    t = calibrate(dev, use_rules=True)
    m = confusion(test, t, use_rules=True)
    print(f"\numbral={t:.3f} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
          f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    assert m["f1"] >= 0.85
    assert "prompt_injection" not in m["fp_por_grupo"], "Se coló un prompt injection"


def test_embeddings_recuperan_mejor_que_el_baseline(guardrail):
    """Justifica el uso de embeddings: Hit@k de artículos esperados >= baseline por palabras clave."""
    test = [c for c in CASES if c["split"] == "test"]
    base = KeywordBaseline()
    h_base = hit_at_k(test, lambda q: [h["articulo"] for h, _ in base.rank(q, TOP_K)])
    h_emb = hit_at_k(test, lambda q: [h["articulo"] for h in guardrail.retrieve(q)])
    print(f"\nHit@{TOP_K}: baseline={h_base:.0%} embeddings={h_emb:.0%}")
    assert h_emb >= h_base