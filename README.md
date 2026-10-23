# Project Charter - Sistema RAG de Consulta del Reglamento Estudiantil (CRUC IUA)

## 1. Título del Proyecto
**Asistente Virtual Inteligente con Control Semántico y RAG para la Consulta del Reglamento Estudiantil del CRUC IUA**

---

## 2. Definición del Problema y Justificación
Los estudiantes y la comunidad académica del CRUC IUA se enfrentan frecuentemente a dificultades para localizar de manera ágil, precisa y centralizada las normativas, condiciones académicas y derechos/deberes estipulados en el Reglamento del Estudiante. La consulta manual de documentos extensos genera demoras administrativas y desinformación.

Por otro lado, los asistentes conversacionales basados únicamente en modelos de lenguaje generativos (*Large Language Models*) son susceptibles a "alucinaciones" (invención de datos) y vulnerables a ataques de inyección de instrucciones (*jailbreaking*). 

Este proyecto resuelve ambas problemáticas mediante la implementación de una arquitectura **RAG (Retrieval-Augmented Generation)** respaldada por un **Guardrail de seguridad determinístico**, garantizando que el sistema responda únicamente con información verídica del reglamento y bloquee consultas fuera de dominio o intentos de manipulación.

---

## 3. Objetivos del Proyecto

* **Objetivo General:** Desarrollar un sistema conversacional seguro basado en IA que responda consultas sobre el Reglamento del Estudiante del CRUC IUA con cero margen de alucinación y alta precisión semántica.
* **Objetivos Específicos:**
  1. Estructurar la base de conocimiento normativa en un dataset sintético/FAQ validado en formato JSON.
  2. Implementar un espacio de búsqueda vectorial ($L_2$ / Coseno) utilizando *embeddings* multilingües y almacenamiento persistente en `ChromaDB`.
  3. Diseñar e integrar un **Guardrail de Intención** mediante umbrales de distancia calibrados para clasificar entradas como *In-Domain* o *Out-of-Domain/Jailbreak*.
  4. Encapsular la lógica en una arquitectura modular en Python, reproducible y preparada para su despliegue mediante API REST y UI interactiva.

---

## 4. Alcance del Proyecto (*Scope*)

### Incluido en el Alcance (In-Scope)
* Indexación completa de las disposiciones normativas (Títulos I al X del Reglamento Estudiantil del CRUC IUA).
* Modelo de *embeddings* denso (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
* Base de datos vectorial persistente con `ChromaDB`.
* Módulo de filtrado determinístico (*Guardrail*) por distancia euclidiana/coseno.
* Cadena de generación aumentada por recuperación (*RAG Chain*) con integración a LLM.
* Pruebas automatizadas de validación del *Guardrail* con casos reales, fuera de dominio e intentos de *jailbreak*.
* Arquitectura en Python modular (`src/`, `data/`, `models/`) con fijación estricta de dependencias (`requirements.txt`) y semillas determinísticas (`SEED`).

### Fuera del Alcance (Out-of-Scope)
* Gestión de trámites administrativos en tiempo real o integración directa con los servidores de SIU Guaraní.
* Modificación dinámica del reglamento en tiempo real sin proceso previo de re-indexación.

---

## 5. Arquitectura del Sistema

```text
[Consulta Usuario]
        │
        ▼
1. Módulo de Embeddings (Sentence-Transformers)
        │
        ▼
2. Vector Store (ChromaDB) ──► Búsqueda por Proximidad Semántica ($L_2$)
        │
        ▼
3. Módulo de Guardrail (Evaluación de Umbral $\tau$)
        │
        ├── Distancia > $\tau$ ──► [Respuesta de Fallback (Out-of-Domain / Jailbreak)]
        │
        └── Distancia ≤ $\tau$ ──► 4. Cadena RAG + LLM ──► [Respuesta Oficial Fundamentada]
```

---

## 6. Stack Tecnológico

| Componente | Tecnología Seleccionada | Justificación Téconica |
| :--- | :--- | :--- |
| **Lenguaje Base** | Python $\ge 3.10$ | Estándar de la industria para ecosistemas de IA/ML. |
| **Embeddings** | `Sentence-Transformers` (`MiniLM-L12`) | Modelo multilingüe optimizado para inferencia rápida y alta comprensión semántica en español. |
| **Vector Database** | `ChromaDB` | Base vectorial embebida, liviana y con soporte de persistencia local. |
| **Framework RAG** | `LangChain Community` | Abstracción estándar para orquestación de prompts, vector stores y modelos. |
| **Modelos de Lenguaje** | `Ollama` / `Groq` / `OpenAI` | Motor de generación contextualizado. |
| **Testing / Repro.** | `pytest` / `torch.manual_seed` | Aislamiento de código y reproducibilidad de resultados. |

---

## 7. Criterios de Éxito y Métricas de Evaluación

1. **Eficacia del Guardrail:** $\ge 95\%$ de precisión en el bloqueo de consultas fuera de dominio (*Out-of-Domain*) e intentos de engaño (*Jailbreak*).
2. **Fidelidad RAG (*Groundedness*):** $100\%$ de las respuestas permitidas deben estar fundamentadas en la información recuperada del dataset oficial.
3. **Reproducibilidad:** Capacidad de ejecutar el proyecto desde cero reproduciendo los mismos resultados vectoriales gracias a la fijación de semillas (`SEED = 42`).

---

## 8. Estructura del Repositorio

```text
├── data/
│   └── faq_dataset.json      # Dataset oficial del Reglamento estructurado
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración global y semillas determinísticas
│   ├── vector_store.py       # Ingesta, embeddings y almacenamiento en ChromaDB
│   ├── guardrail.py          # Lógica de filtrado por umbral de distancia
│   └── rag_chain.py          # Integración con LLM y generación restringida
├── models/
│   └── chroma_db/            # Base de datos vectorial persistida localmente
├── main.py                   # Script de prueba y validación del MVP
├── requirements.txt          # Dependencias y versiones fijadas
└── README.md                 # Project Charter y documentación
```