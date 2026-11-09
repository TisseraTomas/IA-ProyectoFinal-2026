"""Evaluación del guardrail: calibra (umbral y mezcla) en DEV y reporta métricas en TEST.

Uso:
    python -m src.evaluate                 # baseline + embeddings + híbrido (+ reglas)
    python -m src.evaluate --baseline-only # sólo el baseline por palabras clave (no requiere modelos)
"""
from __future__ import annotations

import argparse
import json

from src.baseline import KeywordBaseline
from src.config import DISTANCE_THRESHOLD, HYBRID_ALPHA, TEST_QUERIES_PATH, TOP_K
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
    report("Baseline keywords + reglas", base_rows, True, base_hit)
    if a.baseline_only:
        return

    from src.guardrail import IntentGuardrail
    from src.vector_store import build_vector_store

    guard = IntentGuardrail(build_vector_store(), k=TOP_K, alpha=1.0)  # sólo para recuperar
    cache = {c["id"]: guard.retrieve(c["texto"]) for c in cases}
    emb_rows = [_row(c, cache[c["id"]][0]["distance"]) for c in cases]
    emb_hit = hit_at_k(test_cases, lambda q: [h["articulo"] for h in guard.retrieve(q)])
    report("Embeddings (sólo semántica)", emb_rows, False, emb_hit)
    res = report("Embeddings + reglas", emb_rows, True, emb_hit)

    # Compuerta híbrida: mezcla embeddings + palabras clave; alpha y umbral se eligen SOLO con dev
    alpha, t_h, tabla = choose_alpha(emb_rows, base_rows)
    hyb_rows = hybrid_rows(emb_rows, base_rows, alpha)
    print(f"\nMezcla embeddings/palabras clave (alpha = peso de embeddings), elegida con dev:")
    for al, f1_dev, t in tabla:
        test_m = confusion([r for r in hybrid_rows(emb_rows, base_rows, al) if r["split"] == "test"], t, True)
        print(f"  alpha={al:.2f}  umbral={t:.3f}  F1 dev={f1_dev:.3f}  F1 test={test_m['f1']:.3f}"
              f"{'   <- elegido' if al == alpha else ''}")
    final = report(f"Híbrido + reglas (alpha={alpha:.2f})", hyb_rows, True, emb_hit, threshold=t_h)

    print(f"\nSugerido para src/config.py: HYBRID_ALPHA = {alpha}  DISTANCE_THRESHOLD = {t_h:.3f} "
          f"(actual: HYBRID_ALPHA = {HYBRID_ALPHA}, DISTANCE_THRESHOLD = {DISTANCE_THRESHOLD})")
    print(f"\n=== Diagnóstico del sistema sugerido (alpha={alpha:.2f}, umbral={t_h:.3f}) ===")
    diagnostico(hyb_rows, cache, t_h)


if __name__ == "__main__":
    main()