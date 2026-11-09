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