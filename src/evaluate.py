"""Evaluación del guardrail: calibra el umbral en DEV y reporta métricas en TEST.

Uso:
    python -m src.evaluate                 # embeddings + reglas + baseline
    python -m src.evaluate --baseline-only # sólo el baseline por palabras clave (no requiere modelos)
"""
from __future__ import annotations

import argparse
import json

from src.baseline import KeywordBaseline
from src.config import DISTANCE_THRESHOLD, TEST_QUERIES_PATH, TOP_K
from src.guardrail import detect_injection

CLASES = {"preguntas_reglamento": "permitir", "fuera_de_tema": "rechazar_fuera_de_tema",
          "prompt_injection": "bloquear_injection"}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()

    cases = load_cases()
    n_dev = sum(c["split"] == "dev" for c in cases)
    print(f"Casos: {len(cases)} (dev={n_dev}, test={len(cases) - n_dev}). Umbral calibrado en dev, métricas en test.\n")

    base = KeywordBaseline()
    base_rows = [_row(c, base.distance(c["texto"])) for c in cases]
    base_hit = hit_at_k([c for c in cases if c["split"] == "test"],
                        lambda q: [h["articulo"] for h, _ in base.rank(q, TOP_K)])
    report("Baseline keywords", base_rows, False, base_hit)
    report("Baseline keywords + reglas", base_rows, True, base_hit)
    if a.baseline_only:
        return

    from src.guardrail import IntentGuardrail
    from src.vector_store import build_vector_store

    guard = IntentGuardrail(build_vector_store(), k=TOP_K)
    cache = {c["id"]: guard.retrieve(c["texto"]) for c in cases}
    emb_rows = [_row(c, cache[c["id"]][0]["distance"]) for c in cases]
    emb_hit = hit_at_k([c for c in cases if c["split"] == "test"],
                       lambda q: [h["articulo"] for h in guard.retrieve(q)])
    report("Embeddings (sólo semántica)", emb_rows, False, emb_hit)
    res = report("Embeddings + reglas (sistema)", emb_rows, True, emb_hit)
    print(f"\nUmbral sugerido para src/config.py: DISTANCE_THRESHOLD = {res['umbral']:.3f} "
          f"(actual: {DISTANCE_THRESHOLD})")

    # Diagnóstico: consultas válidas que el sistema bloquearía, por cobertura del reglamento
    t = res["umbral"]
    print("\nConsultas válidas rechazadas (falsos rechazos) con el umbral sugerido:")
    for r in emb_rows:
        if r["permitido"] and r["score"] > t:
            print(f"  {r['id']} [{r.get('cobertura')}] d={r['score']:.3f}  {r['texto'][:70]}")


if __name__ == "__main__":
    main()