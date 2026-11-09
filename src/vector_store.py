"""Creación de embeddings y almacenamiento vectorial (ChromaDB, distancia coseno)."""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import EMBEDDING_MODEL, KB_PATH, PERSIST_DIR

# Campos del chunk que viajan como metadata (y se devuelven en cada recuperación).
META_FIELDS = ("id", "titulo_numero", "titulo", "articulo", "articulo_nombre", "parte", "texto_literal")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Modelo multilingüe; los vectores se normalizan para que L2 y coseno sean equivalentes."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def load_chunks(json_path: str | Path = KB_PATH) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(json_path: str | Path = KB_PATH, persist_directory: str | Path = PERSIST_DIR) -> Chroma:
    """Reconstruye el índice desde cero (determinístico y sin duplicados entre ejecuciones).

    Se indexa `texto_indexable` (encabezado de título/artículo + texto literal) y el resto del
    chunk viaja como metadata, de modo que cada respuesta puede citar artículo y texto original.
    """
    chunks = load_chunks(json_path)
    persist = Path(persist_directory)
    if persist.exists():
        shutil.rmtree(persist)

    metadatas = [{k: ("" if c[k] is None else c[k]) for k in META_FIELDS} for c in chunks]
    return Chroma.from_texts(
        texts=[c["texto_indexable"] for c in chunks],
        embedding=get_embeddings(),
        metadatas=metadatas,
        ids=[c["id"] for c in chunks],
        persist_directory=str(persist),
        collection_metadata={"hnsw:space": "cosine"},
    )