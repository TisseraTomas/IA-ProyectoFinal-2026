import json
from src.vector_store import build_vector_store
from src.guardrail import IntentGuardrail

def test_guardrail_performance():
    # 1. Cargar base y Guardrail
    vector_db = build_vector_store("data/faq_dataset.json", persist_directory="models/chroma_db")
    guardrail = IntentGuardrail(vector_db=vector_db, distance_threshold=8.5)

    # 2. Cargar test set
    with open("data/test_queries.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    tp, tn, fp, fn = 0, 0, 0, 0

    for case in test_cases:
        query = case["query"]
        expected = case["expected_allowed"]
        
        result = guardrail.evaluate_query(query)
        predicted = result["allowed"]

        # Conteo de la matriz de confusión
        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1  # Dejó pasar algo que no debía (Peligro de seguridad)
        elif expected and not predicted:
            fn += 1  # Bloqueó una pregunta legítima (Mala experiencia de usuario)

    total = len(test_cases)
    accuracy = (tp + tn) / total

    print(f"\n--- REPORTE DE EVALUACIÓN DEL GUARDRAIL ---")
    print(f"Total de casos evaluados: {total}")
    print(f"Accuracy (Exactitud): {accuracy * 100:.2f}%")
    print(f"Matriz -> TP: {tp}, TN: {tn}, FP (Fallas de seguridad): {fp}, FN (Falsos rechazos): {fn}")

    # Aserción para asegurar un estándar mínimo de calidad en el MVP (ej: Accuracy > 80%)
    assert accuracy >= 0.80, f"La precisión del guardrail es muy baja: {accuracy}"