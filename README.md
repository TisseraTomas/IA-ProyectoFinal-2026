# Project Charter: Sistema Inteligente de Diagnóstico Agrícola (DiagFolia Vision)

## 1. Información General
* **Nombre del Proyecto:** DiagFolia Vision — Diagnóstico Automatizado de Enfermedades Foliares mediante Visión Artificial
* **Categoría Tecnológica:** Computer Vision (Visión Artificial)
* **Autor / Equipo:** Tomás Agustín Tissera
* **Materia:** Inteligencia Artificial (Ingeniería en Informática - IUA)
* **Modalidad de Trabajo:** Individual (Vía de Escalado a Proyecto Funcional)

---

## 2. Definición del Problema y Modelado del Entorno

### Descripción Funcional
Los cultivadores urbanos, aficionados a la botánica y pequeños productores agrícolas suelen detectar las fitopatologías (plagas, hongos o deficiencias nutricionales) en sus etapas avanzadas, cuando el daño en los cultivos es severo o irreversible. Este proyecto busca proveer un sistema automatizado capaz de clasificar el estado de salud de una planta a partir de una fotografía de sus hojas, emitiendo un diagnóstico temprano junto con recomendaciones agronómicas.

### Modelado del Entorno (PEAS)
* **Performance (Rendimiento):**
  * F1-Score (Macro) ≥ 0,85 en clasificación multiclase.
  * Tiempo de respuesta de la inferencia < 1,5 segundos por imagen.
* **Environment (Entorno):**
  * Fotografías de hojas aisladas (formato JPEG/PNG) bajo condiciones controladas de iluminación y sobre fondo uniforme/natural.
* **Actuators (Actuadores):**
  * Respuesta en formato JSON estandarizado devuelta por la API REST.
  * Panel visual interactivo con barras de probabilidad y tarjetas de recomendación agronómica.
* **Sensors (Sensores):**
  * Enrutador de carga de archivos (HTTP multipart POST) o selector de archivos de la interfaz gráfica.

---

## 3. Origen y Naturaleza de los Datos

* **Dataset Base:** *PlantVillage Dataset* (público y estandarizado).
* **Volumen y Selección:**
  * Para optimizar los tiempos de entrenamiento y mantener rigor en la prueba, se trabajará con un subconjunto enfocado en **3 cultivos clave** (Tomate, Papa y Pimiento), sumando **15 clases distintas** (ej. *Tomate con Tizón Tardío*, *Papa Sana*, *Pimiento con Mancha Bacteriana*, etc.) con un estimado de **~18.000 imágenes**.
* **Preprocesamiento y Calidad:**
  * Resizing estandarizado a 224 x 224 píxeles.
  * Normalización de canales RGB (media y desviación estándar del dataset).
  * *Data Augmentation* en entrenamiento (rotaciones leves, espejado horizontal, variación de brillo/contraste) para evitar sobreajuste (*overfitting*).

---

## 4. Métricas de Éxito y Baseline

### Modelo Baseline (Sin IA avanzada / Red Convencional)
* **Arquitectura:** Red Neuronal Convolucional (CNN) simple construida desde cero (3 capas de `Conv2D` + `MaxPooling` + capas `Dense`), con inicialización aleatoria de pesos (sin Transfer Learning).
* **Objetivo del Baseline:** Establecer la métrica piso a superar (esperado F1-Score ≈ 0,60 - 0,70).

### Modelo Principal y Métrica Objetivo
* **Arquitectura:** Red Neuronal Convolucional con *Transfer Learning* y *Fine-Tuning* utilizando una arquitectura preentrenada liviana (**MobileNetV3** o **EfficientNet-B0**).
* **Métrica de Éxito Objetivo:** Alcanzar un **F1-Score Macro ≥ 0,85** sobre el conjunto de test sin sufrir *data leakage*.

---

## 5. Alcance del Proyecto

### Producto Mínimo Viable (MVP)
* Pipeline completo de datos con división determinística (train/val/test) fijando semilla aleatoria (`seed=42`).
* Entrenamiento, validación y exportación de artefactos del modelo en formato `.pt` (PyTorch) o `.h5` (TensorFlow).
* API REST funcional desarrollada en **FastAPI** que reciba una imagen en la ruta `/predict` mediante un payload válido, realice la inferencia y devuelva un JSON estructurado:
  ```json
  {
    "status": "success",
    "planta": "Tomate",
    "diagnostico": "Tizón Tardío (Phytophthora infestans)",
    "confianza": 0.942,
    "recomendacion": "Aislar la planta afectada, eliminar hojas infectadas y aplicar fungicida a base de cobre."
  }
  ```

### Escalado a Proyecto Funcional (Camino Promoción Directa)
Para dar cumplimiento a las exigencias de ingeniería de software del Camino 2 (Individual):
1. **Eje Arquitectura y Despliegue:** 
   * Validación estricta del esquema de entrada mediante **Pydantic**.
   * Contenerización completa de la solución utilizando **Docker**.
2. **Eje Interfaz y Usabilidad (GUI):** 
   * Desarrollo de una interfaz gráfica interactiva utilizando **Gradio** o **Streamlit** conectada a la API de inferencia.
3. **Visibilidad y Documentación:**
   * Publicación de la documentación, arquitectura del sistema y demostración interactiva en la página web **`usuario.github.io`**.

---

## 6. Límites del Alcance (Out of Scope)

Para acotar el esfuerzo al núcleo de Inteligencia Artificial e Inferencia:
* Autenticación, registro o gestión de roles de usuarios.
* Base de datos relacional persistente para historial de diagnósticos.
* Desarrollo de aplicaciones móviles nativas (Android/iOS).
* Inferencia o procesamiento en tiempo real desde transmisiones de video continuo.

---

## 7. Entorno de Desarrollo y Reproducibilidad

* **Lenguaje:** Python ≥ 3.10.
* **Aislamiento de Entorno:** Entorno virtual (`venv` / `conda`) y contenedor **Docker**.
* **Gestión de Dependencias:** Archivo `requirements.txt` con congelamiento explícito de versiones (*version pinning*).
* **Semilla Aleatoria:** Fijada explícitamente en `42` (`torch.manual_seed(42)`, `np.random.seed(42)`) para garantizar determinismo estricto.
