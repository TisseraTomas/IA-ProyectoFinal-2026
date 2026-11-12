"""Evaluación del guardrail: calibra (umbral y mezcla) en DEV y reporta métricas en TEST.

Uso:
    python -m src.evaluate                 # baseline + embeddings + híbrido (+ reglas)
    python -m src.evaluate --baseline-only # sólo el baseline por palabras clave (no requiere modelos)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.baseline import KeywordBaseline
from src.config import DISTANCE_THRESHOLD, HELDOUT_PATH, HYBRID_ALPHA, SEED, TEST_QUERIES_PATH, TOP_K
from src.guardrail import detect_injection

CLASES = {"preguntas_reglamento": "permitir", "fuera_de_tema": "rechazar_fuera_de_tema",
          "prompt_injection": "bloquear_injection"}
ALPHAS = (1.0, 0.75, 0.5, 0.25)  # 1.0 = sólo embeddings


def load_cases(path=TEST_QUERIES_PATH) -> list[dict]:
    data = json.loads(open(path, encoding="utf-8").read())
    cases = []
    for key in CLASES:
        for c in data[key]:
            cases.append({**c, "grupo": key, "permitido": c["esperado"] == "permitir"})
    return cases


def confusion(rows: list[dict], threshold: float, use_rules: bool) -> dict:
    """rows: casos con 'score' (distancia) y 'rule' (regla que disparó o None). Positivo = permitir."""
    tp = tn = fp = fn = 0
    fp_por_grupo: dict[str, int] = {}
    for r in rows:
        pred = r["score"] <= threshold and not (use_rules and r["rule"])
        if r["permitido"] and pred:
            tp += 1
        elif r["permitido"] and not pred:
            fn += 1
        elif not r["permitido"] and pred:
            fp += 1
            fp_por_grupo[r["grupo"]] = fp_por_grupo.get(r["grupo"], 0) + 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(rows),
            "precision": prec, "recall": rec, "f1": f1, "fp_por_grupo": fp_por_grupo}


def calibrate(rows: list[dict], use_rules: bool) -> float:
    """Umbral que maximiza F1 sobre `rows` (DEV). Ante empates toma el valor central."""
    scores = sorted({r["score"] for r in rows})
    cands = [scores[0] - 1e-6] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-6]
    f1s = [(confusion(rows, t, use_rules)["f1"], t) for t in cands]
    best = max(f for f, _ in f1s)
    ties = [t for f, t in f1s if abs(f - best) < 1e-12]
    return ties[len(ties) // 2]


def hybrid_rows(emb_rows: list[dict], kw_rows: list[dict], alpha: float) -> list[dict]:
    """Mezcla de distancias: alpha * embeddings + (1 - alpha) * palabras clave (mismo orden de casos)."""
    return [{**e, "score": alpha * e["score"] + (1 - alpha) * k["score"]} for e, k in zip(emb_rows, kw_rows)]


def choose_alpha(emb_rows: list[dict], kw_rows: list[dict], alphas=ALPHAS, use_rules: bool = True):
    """Elige alpha usando SOLO dev. Ante empates se queda con el mayor alpha (más cerca de embeddings puros).

    Devuelve (alpha, umbral, tabla) con tabla = [(alpha, f1_dev, umbral)].
    """
    tabla, best = [], None
    for a in sorted(alphas, reverse=True):
        rows = hybrid_rows(emb_rows, kw_rows, a)
        dev = [r for r in rows if r["split"] == "dev"]
        t = calibrate(dev, use_rules)
        f1 = confusion(dev, t, use_rules)["f1"]
        tabla.append((a, f1, t))
        if best is None or f1 > best[1] + 1e-12:
            best = (a, f1, t)
    return best[0], best[2], tabla


def hit_at_k(cases: list[dict], retrieve) -> float | None:
    """Fracción de consultas con cobertura donde algún artículo esperado aparece en el top-k."""
    evaluables = [c for c in cases if c["grupo"] == "preguntas_reglamento" and c["articulos_esperados"]]
    if not evaluables:
        return None
    ok = sum(any(a in set(retrieve(c["texto"])) for a in c["articulos_esperados"]) for c in evaluables)
    return ok / len(evaluables)


# --------------------------------------------------------------------------- estadística sobre TODO el conjunto
def kfold_predictions(rows: list[dict], use_rules: bool, folds: int = 5) -> list[bool]:
    """Predicciones fuera de muestra: cada fold se predice con el umbral calibrado en los demás folds.

    La asignación a folds es determinística (el i-ésimo caso de cada grupo va al fold i % folds).
    """
    contador: dict[str, int] = {}
    fold_de: dict[str, int] = {}
    for r in rows:
        i = contador.get(r["grupo"], 0)
        contador[r["grupo"]] = i + 1
        fold_de[r["id"]] = i % folds
    pred: dict[str, bool] = {}
    for f in range(folds):
        entrenar = [r for r in rows if fold_de[r["id"]] != f]
        t = calibrate(entrenar, use_rules)
        for r in rows:
            if fold_de[r["id"]] == f:
                pred[r["id"]] = r["score"] <= t and not (use_rules and r["rule"])
    return [pred[r["id"]] for r in rows]


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int((y & p).sum())
    fp = int(((~y) & p).sum())
    fn = int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap_f1(rows: list[dict], pred_a: list[bool], pred_b: list[bool], n: int = 2000, seed: int = SEED) -> dict:
    """F1 de dos sistemas con IC 95 % (bootstrap pareado: se remuestrean los mismos casos para ambos)."""
    y = np.array([r["permitido"] for r in rows], dtype=bool)
    a, b = np.array(pred_a, dtype=bool), np.array(pred_b, dtype=bool)
    rng = np.random.RandomState(seed)
    fa, fb, d = [], [], []
    for _ in range(n):
        idx = rng.randint(0, len(rows), len(rows))
        x, z = _f1(y[idx], a[idx]), _f1(y[idx], b[idx])
        fa.append(x), fb.append(z), d.append(x - z)
    ic = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"a": _f1(y, a), "ic_a": ic(fa), "b": _f1(y, b), "ic_b": ic(fb), "diff": _f1(y, a) - _f1(y, b), "ic_diff": ic(d)}


def hit_vector(cases: list[dict], retrieve) -> list[bool]:
    """Acierto (sí/no) por cada pregunta con artículos esperados; `retrieve` devuelve los artículos del top-k."""
    ev = [c for c in cases if c["grupo"] == "preguntas_reglamento" and c["articulos_esperados"]]
    return [any(a in set(retrieve(c["texto"])) for a in c["articulos_esperados"]) for c in ev]


def bootstrap_hits(hits_a: list[bool], hits_b: list[bool], n: int = 2000, seed: int = SEED) -> dict:
    """Tasa de acierto de dos sistemas e IC 95 % de la diferencia (bootstrap pareado)."""
    a, b = np.array(hits_a, dtype=float), np.array(hits_b, dtype=float)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        idx = rng.randint(0, len(a), len(a))
        d.append(a[idx].mean() - b[idx].mean())
    return {"n": len(a), "aciertos_a": int(a.sum()), "aciertos_b": int(b.sum()), "diff": float(a.mean() - b.mean()),
            "ic_diff": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))}


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: combina varios rankings de ids sin parámetros que ajustar."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + pos)
    return sorted(score, key=lambda c: -score[c])


def _row(c: dict, score: float) -> dict:
    return {**c, "score": score, "rule": detect_injection(c["texto"])}


def report(nombre: str, rows: list[dict], use_rules: bool, hit: float | None, threshold: float | None = None) -> dict:
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    t = calibrate(dev, use_rules) if threshold is None else threshold
    m = confusion(test, t, use_rules)
    inj = [r for r in test if r["grupo"] == "prompt_injection"]
    ood = [r for r in test if r["grupo"] == "fuera_de_tema"]
    inj_block = 1 - confusion(inj, t, use_rules)["fp"] / len(inj) if inj else float("nan")
    ood_rej = 1 - confusion(ood, t, use_rules)["fp"] / len(ood) if ood else float("nan")
    print(f"{nombre:<34} umbral={t:.3f} | F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
          f"Acc={m['accuracy']:.3f} | injection bloqueada={inj_block:.0%} fuera de tema rechazada={ood_rej:.0%}"
          + (f" | Hit@{TOP_K}={hit:.0%}" if hit is not None else ""))
    return {**m, "umbral": t}


def diagnostico(rows: list[dict], cache: dict, t: float) -> None:
    print("\nConsultas válidas rechazadas (falsos rechazos):")
    fr = [r for r in rows if r["permitido"] and r["score"] > t]
    for r in fr:
        print(f"  {r['id']} [{r.get('cobertura')}] d={r['score']:.3f}  {r['texto'][:70]}")
    if not fr:
        print("  (ninguna)")

    print("Consultas que NO deberían pasar y el sistema deja pasar (falsos positivos):")
    fp = sorted((r for r in rows if not r["permitido"] and r["score"] <= t and not r["rule"]),
                key=lambda r: r["score"])
    for r in fp:
        print(f"  {r['id']} [{r['split']}] d={r['score']:.3f} -> art. {cache[r['id']][0]['articulo']}  {r['texto'][:60]}")
    if not fp:
        print("  (ninguna)")

    validas = sorted((r for r in rows if r["permitido"]), key=lambda r: -r["score"])
    ajenas = [r["score"] for r in rows if r["grupo"] == "fuera_de_tema"]
    print(f"\nDistancia máx. entre consultas válidas: {validas[0]['score']:.3f} | "
          f"mín. entre fuera de tema: {min(ajenas):.3f} (si la primera supera a la segunda, hay solapamiento)")
    print("Las 6 consultas válidas más lejanas:")
    for r in validas[:6]:
        print(f"  {r['id']} [{r.get('cobertura')}] d={r['score']:.3f}  {r['texto'][:70]}")


def evaluar_heldout(nombre: str, score_fn, threshold: float, top_fn=None, path=HELDOUT_PATH) -> None:
    """Prueba el sistema ya calibrado con casos externos (held-out). El umbral NO se ajusta con estos casos."""
    if not path.exists():
        return
    cases = load_cases(path)
    print(f"\n=== Held-out ({len(cases)} casos externos) | {nombre} | umbral={threshold:.3f} ===")
    ok = 0
    for c in cases:
        rule = detect_injection(c["texto"])
        score = score_fn(c["texto"])
        pred = score <= threshold and not rule
        ok += pred == c["permitido"]
        if rule:
            decision = f"BLOQUEA (regla {rule})"
        elif score > threshold:
            decision = f"RECHAZA (d={score:.3f})"
        else:
            art = f" -> art. {top_fn(c['texto'])}" if top_fn else ""
            decision = f"PERMITE (d={score:.3f}){art}"
        marca = "ok " if pred == c["permitido"] else "ERR"
        print(f"  {marca} {c['id']} esperado={c['esperado'][:12]:<12} {decision:<46} {c['texto'][:55]}")
    print(f"  Aciertos: {ok}/{len(cases)}")


def estadistica_global(cases, base, base_rows, guard=None, emb_rows=None) -> None:
    """Resultados sobre TODO el conjunto (dev+test) con intervalos de confianza al 95 %."""
    print("\n=== Todo el conjunto (dev+test): intervalos de confianza 95 % (bootstrap pareado, 2000 remuestreos) ===")
    art = {c["id"]: c["articulo"] for c in base.chunks}
    hits_base = hit_vector(cases, lambda q: [h["articulo"] for h, _ in base.rank(q, TOP_K)])
    n = len(hits_base)
    print(f"Preguntas con artículos esperados: {n}")
    print(f"  Hit@{TOP_K} baseline palabras clave: {sum(hits_base)}/{n} = {sum(hits_base) / n:.0%}")
    p_base = kfold_predictions(base_rows, True)
    if guard is None:
        r = bootstrap_f1(base_rows, p_base, p_base)
        print(f"  F1 baseline + reglas (validación cruzada 5 folds): {r['a']:.3f} IC95 [{r['ic_a'][0]:.3f}, {r['ic_a'][1]:.3f}]")
        return

    full = len(base.chunks)

    def emb_rank(q):
        return [d.metadata["id"] for d, _ in guard.vector_db.similarity_search_with_score(q, k=full)]

    hits_emb = hit_vector(cases, lambda q: [art[c] for c in emb_rank(q)[:TOP_K]])
    hits_rrf = hit_vector(cases, lambda q: [art[c] for c in
                                            rrf([emb_rank(q), [c["id"] for c, _ in base.rank(q, full)]])[:TOP_K]])
    for nombre, h in (("embeddings", hits_emb), ("híbrido RRF (embeddings + palabras clave)", hits_rrf)):
        r = bootstrap_hits(h, hits_base)
        print(f"  Hit@{TOP_K} {nombre}: {sum(h)}/{n} = {sum(h) / n:.0%} | diferencia vs baseline {r['diff']:+.1%} "
              f"IC95 [{r['ic_diff'][0]:+.1%}, {r['ic_diff'][1]:+.1%}]")
    p_emb = kfold_predictions(emb_rows, True)
    r = bootstrap_f1(emb_rows, p_emb, p_base)
    print(f"  F1 (validación cruzada 5 folds, con reglas): embeddings {r['a']:.3f} IC95 [{r['ic_a'][0]:.3f}, {r['ic_a'][1]:.3f}]"
          f" | baseline {r['b']:.3f} IC95 [{r['ic_b'][0]:.3f}, {r['ic_b'][1]:.3f}]")
    print(f"  Diferencia de F1 (embeddings - baseline): {r['diff']:+.3f} IC95 [{r['ic_diff'][0]:+.3f}, {r['ic_diff'][1]:+.3f}]"
          "  (si el intervalo incluye 0, no hay diferencia demostrable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()

    cases = load_cases()
    n_dev = sum(c["split"] == "dev" for c in cases)
    print(f"Casos: {len(cases)} (dev={n_dev}, test={len(cases) - n_dev}). Umbral calibrado en dev, métricas en test.\n")

    test_cases = [c for c in cases if c["split"] == "test"]
    base = KeywordBaseline()
    base_rows = [_row(c, base.distance(c["texto"])) for c in cases]
    base_hit = hit_at_k(test_cases, lambda q: [h["articulo"] for h, _ in base.rank(q, TOP_K)])
    report("Baseline keywords", base_rows, False, base_hit)
    rb = report("Baseline keywords + reglas", base_rows, True, base_hit)
    if a.baseline_only:
        evaluar_heldout("Baseline + reglas", base.distance, rb["umbral"])
        estadistica_global(cases, base, base_rows)
        return

    from src.guardrail import IntentGuardrail
    from src.vector_store import build_vector_store

    guard = IntentGuardrail(build_vector_store(), k=TOP_K, alpha=1.0)  # sólo para recuperar
    cache = {c["id"]: guard.retrieve(c["texto"]) for c in cases}
    emb_rows = [_row(c, cache[c["id"]][0]["distance"]) for c in cases]
    emb_hit = hit_at_k(test_cases, lambda q: [h["articulo"] for h in guard.retrieve(q)])
    report("Embeddings (sólo semántica)", emb_rows, False, emb_hit)
    report("Embeddings + reglas", emb_rows, True, emb_hit)

    # Compuerta híbrida: mezcla embeddings + palabras clave; alpha y umbral se eligen SOLO con dev
    alpha, t_h, tabla = choose_alpha(emb_rows, base_rows)
    hyb_rows = hybrid_rows(emb_rows, base_rows, alpha)
    print("\nMezcla embeddings/palabras clave (alpha = peso de embeddings), elegida con dev:")
    for al, f1_dev, t in tabla:
        test_m = confusion([r for r in hybrid_rows(emb_rows, base_rows, al) if r["split"] == "test"], t, True)
        print(f"  alpha={al:.2f}  umbral={t:.3f}  F1 dev={f1_dev:.3f}  F1 test={test_m['f1']:.3f}"
              f"{'   <- elegido' if al == alpha else ''}")
    report(f"Híbrido + reglas (alpha={alpha:.2f})", hyb_rows, True, emb_hit, threshold=t_h)

    print(f"\nSugerido para src/config.py: HYBRID_ALPHA = {alpha}  DISTANCE_THRESHOLD = {t_h:.3f} "
          f"(actual: HYBRID_ALPHA = {HYBRID_ALPHA}, DISTANCE_THRESHOLD = {DISTANCE_THRESHOLD})")
    print(f"\n=== Diagnóstico del sistema sugerido (alpha={alpha:.2f}, umbral={t_h:.3f}) ===")
    diagnostico(hyb_rows, cache, t_h)
    estadistica_global(cases, base, base_rows, guard, emb_rows)

    # Casos externos con el sistema tal como quedó calibrado en dev (alpha y umbral ya elegidos)
    def score_sistema(q: str) -> float:
        d = guard.retrieve(q)[0]["distance"]
        return d if alpha >= 1.0 else alpha * d + (1 - alpha) * base.distance(q)

    evaluar_heldout(f"Embeddings + reglas (alpha={alpha:.2f})", score_sistema, t_h,
                    top_fn=lambda q: guard.retrieve(q)[0]["articulo"])

if __name__ == "__main__":
    main()