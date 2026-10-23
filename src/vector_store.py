"""Creación de embeddings y almacenamiento vectorial."""
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import SEED

# Cargamos el modelo multilingüe liviano
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def build_vector_store(json_path: str, persist_directory: str = "./models/chroma_db"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadatas = []

    for item in data:
        # Indexamos la pregunta de referencia y sus variantes
        all_questions = [item["pregunta_referencia"]] + item.get("preguntas_alternativas", [])
        for q in all_questions:
            texts.append(q)
            metadatas.append({
                "id": item["id"],
                "categoria": item["categoria"],
                "respuesta_oficial": item["respuesta_oficial"]
            })

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    return vector_db