"""Generación de respuestas con un modelo de lenguaje."""
from langchain_community.llms import Ollama  # o la API elegida (Groq/OpenAI)

SYSTEM_PROMPT = """Sos un asistente virtual oficial de la Facultad de Ingeniería del CRUC IUA.
Tu única función es responder la pregunta del usuario utilizando EXCLUSIVAMENTE la siguiente información oficial expuesta.
Si la información proporcionada no responde directamente a la pregunta, indica que no dispones de esa información.
No inventes ni agregues regulaciones que no estén escritas aquí.

Información Oficial: {respuesta_oficial}
Pregunta del Usuario: {pregunta_usuario}
"""

def generate_response(user_query: str, guardrail_result: dict, llm_model):
    if not guardrail_result["allowed"]:
        return "Lo siento, solo puedo responder preguntas relacionadas con el Reglamento del Estudiante del CRUC IUA."

    respuesta_oficial = guardrail_result["data"]["respuesta_oficial"]
    prompt_final = SYSTEM_PROMPT.format(
        respuesta_oficial=respuesta_oficial,
        pregunta_usuario=user_query
    )

    # Generación controlada
    return llm_model.invoke(prompt_final)