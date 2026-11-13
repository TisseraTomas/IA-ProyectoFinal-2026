"""Guardrail de dos capas: (1) reglas anti prompt-injection, (2) distancia semántica al reglamento."""
from __future__ import annotations

import re
import unicodedata

from src.config import DISTANCE_THRESHOLD, HYBRID_ALPHA, TOP_K

# --------------------------------------------------------------------------- capa 1: reglas
# Se aplican sobre el texto en minúsculas y sin acentos.
_OVERRIDE = (
    r"\b(ignora|ignorame|ignoren|olvida|olvidate|olvidense|desestima|omite|saltea|salteate|saltate|"
    r"ignore|disregard|forget|bypass|override)\b.{0,60}"
    r"\b(instruccion\w*|regla\w*|reglamento|restriccion\w*|limite\w*|indicacion\w*|filtro\w*|directiva\w*|"
    r"prompt|documento\w*|rule\w*|instruction\w*|guideline\w*|anterior\w*|previa\w*|previous|prior|above)"
)
INJECTION_PATTERNS: dict[str, str] = {
    "ignorar_instrucciones": _OVERRIDE,
    "pedir_prompt_o_config": (
        r"\b(tu|tus|your)\s+(prompt|instruccion\w*|indicacion\w*|directiva\w*|pauta\w*|lineamiento\w*|"
        r"configuracion|config\w*|reglas|system|hidden|internal|secret)\b"
    ),
    "prompt_de_sistema": (
        r"\b(prompt|instruccion\w*|indicacion\w*|directiva\w*|configuracion)\s+(del?\s+sistema|internas?|ocultas?|iniciales|secretas?)"
        r"|\b(hidden|internal|secret)\s+(configuration|instructions?|prompt|rules)"
        r"|system[ _]?prompt|reglas_internas"
    ),
    # "DAN" es un jailbreak conocido pero también el verbo "dar" en español ("me dan equivalencias"):
    # sólo se marca con contexto de rol/modo.
    "modo_especial": (
        r"developer mode|modo (desarrollador|developer|dios|mantenimiento|admin|root|dan)|\bdan mode\b|jailbreak|"
        r"do anything now|\b(sos|eres|you are|you're|actua como|act as|ahora sos)\s+dan\b"
    ),
    "delimitadores_falsos": (
        r"\[/?(system|sistema|inst|admin)\]|</?(reglamento|system|sistema|instructions?)>|<\|.*?\||###\s*(system|instruction)"
        r"|^\s*(system|sistema|assistant|asistente|developer|admin|root)\s*:"
    ),
    "autoridad_falsa": (
        r"\b(soy|i am|i'm)\s+(el |la |un |una )?(administrador|admin|desarrollador|developer|creador|programador)\b"
        r"|(codigo|code) de (autorizacion|acceso)|admin override|\boverride\b"
        r"|\bel usuario (es|tiene)\s+(administrador|admin|root|permisos)"
        r"|\bpermisos (de|como) (admin|administrador|root)"
    ),
    "ia_sin_restricciones": (
        r"\b(bot|ia|ai|modelo|asistente|assistant)\s+(sin|without)\b|unrestricted|jailbroken|uncensored|sin censura"
    ),
    "olvidar_lo_anterior": (
        r"\b(olvidate|olvida|olvidense|ignora|ignorame)\s+(de\s+)?(todo|lo)\s+(lo\s+)?(que|anterior)"
        r"|\bcomo si no tuvieras\s+(ninguna\s+)?(politica|regla|restriccion|limite|filtro)\w*"
    ),
    "repetir_contexto": (
        r"\b(repeti|repite|copia|pega|escribi|transcribi)\w*\b.{0,40}"
        r"(lo que dice (arriba|antes)|texto (previo|anterior|inicial)|mensaje (anterior|inicial|del sistema)|todo lo anterior)"
    ),
    "credenciales_admin": r"\b(clave|contrasena|password|credenciales)\s+(del?\s+)?(administrador|admin|sistema|root)\b",
    "nueva_regla": r"\b(nuevas?|new)\s+(regla|instruccion|rule|instruction)s?\s*:",
    "codificacion": r"[a-z0-9+/]{24,}={0,2}|base64|decodifica|rot13",
    "secretos": (
        r"\b(muestra\w*|mostra\w*|imprim\w*|revela\w*|reveal|show|print|dame|pasame|decime)\b.{0,40}"
        r"(variables? de entorno|api[ _-]?key|claves?|tokens?|credenciales|password)"
    ),
}
_COMPILED = {name: re.compile(rx) for name, rx in INJECTION_PATTERNS.items()}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def detect_injection(user_query: str) -> str | None:
    """Devuelve el nombre de la primera regla que dispara (o None). No usa embeddings."""
    norm = normalize(user_query)
    # el patrón de codificación se evalúa sobre el texto original (distingue mayúsculas/minúsculas de base64)
    if re.search(r"[A-Za-z0-9+/]{24,}={0,2}", user_query):
        return "codificacion"
    for name, rx in _COMPILED.items():
        if name != "codificacion" and rx.search(norm):
            return name
    if "base64" in norm or "decodifica" in norm or "rot13" in norm:
        return "codificacion"
    return None


# --------------------------------------------------------------------------- capa 2: semántica
class IntentGuardrail:
    def __init__(self, vector_db, distance_threshold: float = DISTANCE_THRESHOLD, k: int = TOP_K,
                 alpha: float = HYBRID_ALPHA):
        self.vector_db = vector_db
        self.threshold = distance_threshold
        self.k = k
        self.alpha = alpha
        self.baseline = None
        if alpha < 1.0:  # compuerta híbrida: se combina con la distancia por palabras clave
            from src.baseline import KeywordBaseline  # import diferido (evita import circular)

            self.baseline = KeywordBaseline()

    def retrieve(self, user_query: str) -> list[dict]:
        """Top-k fragmentos con su distancia coseno (menor = más cercano)."""
        results = self.vector_db.similarity_search_with_score(user_query, k=self.k)
        return [{**doc.metadata, "distance": float(dist)} for doc, dist in results]

    def score(self, user_query: str, hits: list[dict] | None = None) -> float:
        """Puntaje de la compuerta (menor = más en dominio): embeddings, o mezcla con palabras clave."""
        hits = hits or self.retrieve(user_query)
        d = hits[0]["distance"]
        if self.baseline is None:
            return d
        return self.alpha * d + (1.0 - self.alpha) * self.baseline.distance(user_query)

    def evaluate_query(self, user_query: str) -> dict:
        rule = detect_injection(user_query)
        if rule:
            return {"allowed": False, "reason": f"Prompt injection ({rule})", "score": None, "data": None, "rule": rule}

        hits = self.retrieve(user_query)
        if not hits:
            return {"allowed": False, "reason": "Sin coincidencias", "score": None, "data": None, "rule": None}

        best = self.score(user_query, hits)
        if best > self.threshold:
            return {"allowed": False, "reason": "Fuera de dominio", "score": best, "data": None, "rule": None}
        return {"allowed": True, "reason": "En dominio", "score": best, "data": hits, "rule": None}