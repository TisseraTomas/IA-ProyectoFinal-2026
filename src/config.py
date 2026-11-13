"""Parámetros globales, rutas y semillas del proyecto."""
import logging
import os
import random
from pathlib import Path

import numpy as np

# Desactiva la telemetría anónima de ChromaDB (evita los avisos "Failed to send telemetry event").
# Debe fijarse antes de importar chromadb; como todos los módulos importan este archivo primero, alcanza.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Con posthog >= 6, chromadb 0.5.0 loguea "Failed to send telemetry event" aunque la telemetría esté apagada.
# El error es inofensivo; se silencia el logger (la solución de fondo es fijar posthog<6 en requirements.txt).
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "reglamento_chunks.json"
TEST_QUERIES_PATH = ROOT / "data" / "test_queries.json"
HELDOUT_PATH = ROOT / "data" / "test_queries_heldout.json"  # casos externos: nunca se usan para calibrar
PERSIST_DIR = ROOT / "models" / "chroma_db"
LOG_PATH = ROOT / "logs" / "consultas.jsonl"  # log estructurado de la API (una línea JSON por consulta)

# Modelo de embeddings MULTILINGÜE (entrenado para paráfrasis en 50+ idiomas, incluido español).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cantidad de fragmentos del reglamento que se recuperan por consulta.
TOP_K = 3

# Umbral de la distancia COSENO (1 - similitud) del fragmento más cercano.
# Peso de la distancia de embeddings en la compuerta: score = ALPHA * d_embeddings + (1 - ALPHA) * d_palabras_clave.
# 1.0 = sólo embeddings. `python -m src.evaluate` sugiere el valor (se elige en dev).
HYBRID_ALPHA = 1.0

# Calibrado en el conjunto dev con `python -m src.evaluate`; recalibrar si cambia la base o el test set.
DISTANCE_THRESHOLD = 0.674


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla global (random, numpy y torch si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # permite correr las pruebas "offline" sin torch
        pass


set_seed()