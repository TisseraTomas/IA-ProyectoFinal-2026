"""API REST del sistema (FastAPI): consulta del Reglamento del Alumno con guardrail y logging estructurado.

Ejecución:  uvicorn src.api:app --port 8000      (documentación interactiva en http://127.0.0.1:8000/docs)
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.config import DISTANCE_THRESHOLD, EMBEDDING_MODEL, HYBRID_ALPHA, KB_PATH, LOG_PATH
from src.rag_chain import FALLBACK

logger = logging.getLogger("iua.api")


# --------------------------------------------------------------------------- esquemas (validación con Pydantic)
class ConsultaRequest(BaseModel):
    pregunta: str = Field(..., min_length=3, max_length=500,
                          description="Consulta del estudiante sobre el Reglamento del Alumno (3 a 500 caracteres).",
                          examples=["hasta cuando vale la regularidad de una materia?"])

    @field_validator("pregunta")
    @classmethod
    def sin_espacios_sobrantes(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("la pregunta no puede estar vacía ni ser sólo espacios")
        return v


class Fragmento(BaseModel):
    id: str
    titulo: str
    articulo: str
    articulo_nombre: str
    parte: Optional[str] = None
    texto_literal: str
    distancia: float = Field(..., description="Distancia coseno al fragmento (menor = más cercano).")


class ConsultaResponse(BaseModel):
    request_id: str
    permitido: bool = Field(..., description="True si la consulta pertenece al dominio y no es un intento de manipulación.")
    motivo: str = Field(..., description="en_dominio | fuera_de_dominio | prompt_injection | sin_coincidencias")
    distancia: Optional[float] = None
    fragmentos: List[Fragmento] = []
    mensaje: Optional[str] = Field(None, description="Mensaje al usuario cuando la consulta no se responde.")


class ArticuloChunk(BaseModel):
    id: str
    titulo: str
    articulo: str
    articulo_nombre: str
    parte: Optional[str] = None
    texto_literal: str


# --------------------------------------------------------------------------- logging estructurado (JSON lines)
class _SoloMensaje(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configurar_logging(ruta: Path = LOG_PATH) -> logging.Logger:
    """Una línea JSON por evento, en consola y en archivo. Reconfigurable (los tests usan otra ruta)."""
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(ruta, encoding="utf-8", delay=True)):
        handler.setFormatter(_SoloMensaje())
        logger.addHandler(handler)
    return logger


def log_evento(evento: str, **campos) -> None:
    registro = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "evento": evento, **campos}
    logger.info(json.dumps(registro, ensure_ascii=False))


def _codigo_motivo(res: dict) -> str:
    if res.get("rule"):
        return "prompt_injection"
    if res["allowed"]:
        return "en_dominio"
    return "fuera_de_dominio" if str(res.get("reason", "")).startswith("Fuera") else "sin_coincidencias"


def _cargar_chunks() -> list:
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- aplicación
def create_app(guardrail=None, log_path: Path = LOG_PATH) -> FastAPI:
    """Fábrica de la app. Si no se pasa `guardrail`, se construye el índice (con modelos) al iniciar."""
    configurar_logging(log_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.guardrail is None:
            from src.guardrail import IntentGuardrail
            from src.vector_store import build_vector_store

            app.state.guardrail = IntentGuardrail(build_vector_store(), distance_threshold=DISTANCE_THRESHOLD,
                                                  alpha=HYBRID_ALPHA)
            log_evento("inicio", modelo=EMBEDDING_MODEL, umbral=DISTANCE_THRESHOLD, fragmentos=len(app.state.chunks))
        yield

    app = FastAPI(
        title="Consulta del Reglamento del Alumno (CRUC IUA)",
        version="1.0.0",
        description="Recupera los artículos del Reglamento (Títulos I-X) relevantes para una consulta informal. "
                    "Un guardrail de dos capas rechaza consultas fuera de dominio e intentos de prompt injection.",
        lifespan=lifespan,
    )
    app.state.guardrail = guardrail
    app.state.chunks = _cargar_chunks()

    @app.middleware("http")
    async def request_id_y_latencia(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        inicio = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - inicio) * 1000:.1f}"
        return response

    @app.get("/health", tags=["Estado"])
    def health():
        """Estado del servicio. 503 si el índice todavía no está cargado."""
        if app.state.guardrail is None:
            raise HTTPException(status_code=503, detail="El índice vectorial todavía no está cargado.")
        return {"status": "ok", "fragmentos": len(app.state.chunks), "modelo": EMBEDDING_MODEL,
                "umbral": DISTANCE_THRESHOLD}

    @app.post("/consulta", response_model=ConsultaResponse, tags=["Consulta"])
    def consulta(body: ConsultaRequest, request: Request):
        """Evalúa la consulta (reglas anti-injection + distancia semántica) y devuelve los artículos más cercanos.

        Una consulta rechazada **no es un error**: responde 200 con `permitido=false` y el motivo.
        Un cuerpo inválido (campo faltante, vacío o de más de 500 caracteres) responde 422.
        """
        if app.state.guardrail is None:
            raise HTTPException(status_code=503, detail="El índice vectorial todavía no está cargado.")
        inicio = time.perf_counter()
        try:
            res = app.state.guardrail.evaluate_query(body.pregunta)
        except Exception as exc:  # error inesperado del modelo/índice
            log_evento("error", request_id=request.state.request_id, error=type(exc).__name__, detalle=str(exc)[:200])
            raise HTTPException(status_code=500, detail="Error interno al evaluar la consulta.")

        motivo = _codigo_motivo(res)
        fragmentos = [Fragmento(id=h["id"], titulo=h["titulo"], articulo=h["articulo"],
                                articulo_nombre=h["articulo_nombre"], parte=h.get("parte") or None,
                                texto_literal=h["texto_literal"], distancia=h["distance"])
                      for h in (res.get("data") or [])]
        log_evento("consulta", request_id=request.state.request_id, pregunta=body.pregunta[:200],
                   permitido=bool(res["allowed"]), motivo=motivo, regla=res.get("rule"),
                   distancia=None if res.get("score") is None else round(float(res["score"]), 4),
                   articulos=[f.articulo for f in fragmentos],
                   latencia_ms=round((time.perf_counter() - inicio) * 1000, 1))
        return ConsultaResponse(request_id=request.state.request_id, permitido=bool(res["allowed"]), motivo=motivo,
                                distancia=res.get("score"), fragmentos=fragmentos,
                                mensaje=None if res["allowed"] else FALLBACK)

    @app.get("/articulos/{articulo}", response_model=List[ArticuloChunk], tags=["Reglamento"])
    def obtener_articulo(articulo: str):
        """Devuelve el/los fragmentos de un artículo (p. ej. `7.2.2.1`, `4.4.2`, `VIII`). 404 si no existe."""
        encontrados = [c for c in app.state.chunks if c["articulo"] == articulo]
        if not encontrados:
            raise HTTPException(status_code=404, detail=f"El artículo '{articulo}' no existe en los Títulos I-X.")
        return [ArticuloChunk(id=c["id"], titulo=c["titulo"], articulo=c["articulo"],
                              articulo_nombre=c["articulo_nombre"], parte=c["parte"], texto_literal=c["texto_literal"])
                for c in encontrados]

    return app


app = create_app()