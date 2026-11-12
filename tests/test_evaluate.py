"""Pruebas unitarias de las métricas y la calibración (no requieren modelos)."""
from src.evaluate import calibrate, confusion


def _rows():
    mk = lambda s, ok, g="x", rule=None: {"score": s, "permitido": ok, "grupo": g, "rule": rule, "split": "dev"}
    return [mk(0.2, True), mk(0.3, True), mk(0.4, True), mk(0.8, False, "fuera_de_tema"),
            mk(0.9, False, "fuera_de_tema"), mk(0.1, False, "prompt_injection", rule="r")]


def test_confusion_con_y_sin_reglas():
    sin = confusion(_rows(), 0.5, use_rules=False)
    assert (sin["tp"], sin["fn"], sin["fp"], sin["tn"]) == (3, 0, 1, 2)  # el injection cercano se cuela
    con = confusion(_rows(), 0.5, use_rules=True)
    assert (con["fp"], con["tn"], con["f1"]) == (0, 3, 1.0)


def test_calibrate_separa_las_clases():
    t = calibrate(_rows(), use_rules=True)
    assert 0.4 <= t < 0.8


def test_choose_alpha_prefiere_la_mezcla_cuando_los_embeddings_solos_se_solapan():
    from src.evaluate import choose_alpha, hybrid_rows

    def mk(i, emb, ok, g):
        return {"id": i, "score": emb, "permitido": ok, "grupo": g, "rule": None, "split": "dev"}

    emb = [mk("v1", 0.30, True, "pr"), mk("v2", 0.40, True, "pr"), mk("v3", 0.50, True, "pr"),
           mk("o1", 0.45, False, "fuera_de_tema"), mk("o2", 0.50, False, "fuera_de_tema"), mk("o3", 0.55, False, "fuera_de_tema")]
    kw = [{**r, "score": s} for r, s in zip(emb, [0.2, 0.3, 0.3, 0.9, 0.95, 0.9])]
    alpha, t, tabla = choose_alpha(emb, kw)
    assert alpha < 1.0                       # con embeddings solos no se pueden separar
    assert dict((a, f) for a, f, _ in tabla)[1.0] < 1.0
    assert dict((a, f) for a, f, _ in tabla)[alpha] == 1.0
    assert len(hybrid_rows(emb, kw, alpha)) == len(emb)


def test_kfold_bootstrap_y_rrf():
    import numpy as np
    from src.evaluate import bootstrap_f1, bootstrap_hits, kfold_predictions, rrf

    rows = []
    for i in range(20):
        ok = i % 2 == 0
        rows.append({"id": f"c{i}", "grupo": "preguntas_reglamento" if ok else "fuera_de_tema", "permitido": ok,
                     "score": 0.2 + 0.01 * i if ok else 0.8 + 0.01 * i, "rule": None, "split": "dev"})
    pred = kfold_predictions(rows, use_rules=False)
    assert len(pred) == len(rows) and pred == [r["permitido"] for r in rows]  # separables -> sin errores

    r = bootstrap_f1(rows, pred, pred)
    assert r["a"] == 1.0 and r["ic_a"][0] <= r["a"] <= r["ic_a"][1] and r["diff"] == 0.0

    h = bootstrap_hits([True] * 8 + [False] * 2, [True] * 6 + [False] * 4)
    assert h["aciertos_a"] == 8 and h["aciertos_b"] == 6 and h["ic_diff"][0] <= h["diff"] <= h["ic_diff"][1]

    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"] or rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["b", "a"]
    assert rrf([["x", "y"], ["z", "x"]])[0] == "x"  # el que aparece en ambos rankings queda primero