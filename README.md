# 🛡️ Sistema RAG + Guardrail con IA para Reglamento Estudiantil (IUA)

Sistema conversacional inteligente y seguro diseñado para responder consultas sobre el **Reglamento del Estudiante de la Facultad de Ingeniería del IUA**, implementando un mecanismo de filtrado de seguridad (*Guardrail*) basado en distancia vectorial para prevenir alucinaciones, consultas fuera de dominio e intentos de *Prompt Injection*.

---

## 1. Project Charter

### **1.1. Propósito y Objetivos**
*   **Objetivo Principal:** Proveer un asistente virtual automatizado y seguro que resuelva dudas normativas frecuentes de los estudiantes de ingeniería basándose estrictamente en el articulado oficial.
*   **Criterio de Éxito (*Promoción Directa*):** Lograr una arquitectura modular, control de versiones basado en commits atómicos, reproducibilidad total del entorno y un rendimiento del Guardrail con un **F1-Score > 0.85** evaluado sobre un conjunto de pruebas sintéticas.

### **1.2. Alcance (Scope)**
*   **In-Scope:** 
    *   Procesamiento e indexación vectorial del Reglamento del Estudiante (Títulos I-X).
    *   Motor de recuperación semántica (*Retriever*) con ChromaDB y `sentence-transformers` (`all-MiniLM-L12-v2`).
    *   Guardrail determinístico basado en umbrales de distancia $L_2$ para clasificar consultas válidas (*In-Domain*), fuera de tema (*Out-of-Domain*) e inyecciones de código (*Jailbreak*).
    *   Interfaz CLI y script de validación automatizada mediante pruebas unitarias.
*   **Out-of-Scope:**
    *   Despliegue en servidores cloud pagados (se ejecuta localmente o mediante contenedores livianos).
    *   Interfaz gráfica web compleja (se prioriza la robustez del backend y la lógica RAG).

---

## 2. Especificación de Datos y Línea Base (Baseline)

### **2.1. Data Card (Dataset de Conocimiento y Validación)**
*   **Base de Conocimiento (`data/faq_dataset.json`):** Contiene las normas oficializadas fragmentadas en unidades atómicas (asistencia, regímenes de exámenes, correlatividades, licencias) para optimizar el cálculo de *embeddings*.
*   **Conjunto de Validación (`data/test_queries.json`):** Dataset de prueba compuesto por **45 casos de evaluación rigurosa**:
    *   *Preguntas Válidas (In-Domain):* 20 consultas formuladas con lenguaje coloquial, abreviaciones y variaciones sintácticas del estudiante argentino.
    *   *Fuera de Tema (Out-of-Domain):* 15 consultas de cultura general, cocina, clima y otras áreas ajenas a la institución.
    *   *Ataques de Seguridad (Prompt Injection):* 10 vectores de ataque probando *roleplaying* (DAN), delimitadores falsos e intentos de extracción del *system prompt*.

### **2.2. Línea Base (Baseline Methodológico)**
Para justificar la complejidad del uso de **Embeddings + ChromaDB** frente a métodos tradicionales, se comparó el sistema contra una línea base basada en **búsqueda por palabras clave (*Keyword Matching / Regex*)**:
*   *Limitación del Baseline:* Fallaba ante sinónimos, errores de tipeo o preguntas informales (*"cuanto % de asistencia necesito pa no quedar libre??"*).
*   *Ventaja del enfoque actual:* El espacio vectorial semántico permite capturar la intención real de la consulta independientemente de la formulación superficial, logrando reducir drásticamente los falsos rechazos.

---

## 3. Arquitectura del Proyecto

```text
├── data/
│   ├── faq_dataset.json      # Base de conocimiento oficial trozada
│   └── test_queries.json     # Dataset de validación para métricas del Guardrail
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración y semillas de reproducibilidad
│   ├── vector_store.py       # Gestión y persistencia del índice vectorial (ChromaDB)
│   ├── guardrail.py          # Lógica de filtrado y umbral de distancia L2
│   └── rag_chain.py          # Conexión con el LLM generativo (Ollama / Local)
├── tests/
│   └── test_guardrail.py     # Suite de pruebas automatizadas y matriz de confusión
├── main.py                   # Script de ejecución principal del MVP
├── requirements.txt          # Dependencias fijadas con versiones estrictas
└── README.md                 # Documentación y Project Charter
```
## 🚀 4. Guía de Inicio Rápido (Quickstart & Reproducibilidad)

Seguí estos pasos para clonar, configurar y poner en marcha el proyecto en tu entorno local:

### **4.1. Requisitos Previos**
*   Python 3.10 o superior instalado.
*   Git instalado.

### **4.2. Instalación del Entorno**
1. Clona el repositorio y entra al directorio del proyecto:
```bash
   git clone [https://github.com/TisseraTomas/IA-ProyectoFinal-2026.git](https://github.com/TisseraTomas/IA-ProyectoFinal-2026.git)
   cd IA-ProyectoFinal-2026
```
Crea y activa un entorno virtual (venv):

```Bash
python -m venv venv
# En Linux / WSL:
source venv/bin/activate
# En Windows (PowerShell):
# venv\Scripts\Activate.ps1
```
2. Instala las dependencias del proyecto:

```Bash
pip install -r requirements.txt
```
### **4.3. Ejecución del MVP**
Para correr el flujo principal del asistente interactivo:

```Bash
python main.py
```
### **4.4. Ejecución de Pruebas Automatizadas (Métricas y Guardrail)**
Para correr la suite de validación y verificar la matriz de confusión sobre el Test Set:

```Bash
pytest -v
```
## 5. Validación y Métricas
El rendimiento del Guardrail se evalúa midiendo los siguientes indicadores sobre el conjunto de pruebas:

- Accuracy (Exactitud General): Porcentaje total de decisiones correctas.

- Precision & Recall: Capacidad de bloquear amenazas sin descartar consultas legítimas.

- F1-Score: Métrica global de balance de seguridad (Objetivo: > 0.85).
