"""Parámetros globales, rutas y semillas del proyecto."""
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "reglamento_chunks.json"
TEST_QUERIES_PATH = ROOT / "data" / "test_queries.json"
PERSIST_DIR = ROOT / "models" / "chroma_db"

# Modelo de embeddings MULTILINGÜE (entrenado para paráfrasis en 50+ idiomas, incluido español).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cantidad de fragmentos del reglamento que se recuperan por consulta.
TOP_K = 3

# Umbral de la distancia COSENO (1 - similitud) del fragmento más cercano.
# Valor inicial: recalibrar con `python -m src.evaluate` y pegar acá el valor sugerido.
DISTANCE_THRESHOLD = 0.60


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