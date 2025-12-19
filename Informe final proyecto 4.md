# Informe Final de Proyecto: Desarrollo de un Tutor Inteligente de Algoritmos basado en LLMs
---

## 1. Resumen Ejecutivo

El presente proyecto detalla el proceso de diseño, entrenamiento e implementación de un **Asistente Pedagógico Virtual** especializado en Ciencias de la Computación. Utilizando técnicas de vanguardia en Inteligencia Artificial Generativa, específicamente el ajuste fino eficiente (Parameter-Efficient Fine-Tuning) mediante **QLoRA**, se logró transformar un modelo de lenguaje generalista en un tutor experto capaz de explicar conceptos algorítmicos complejos, generar analogías didácticas y depurar código utilizando un enfoque socrático.

El modelo final, optimizado para ejecutarse en hardware de consumo (GPUs con 6GB de VRAM), demuestra una capacidad notable para guiar el aprendizaje del estudiante sin entregar soluciones directas, fomentando el pensamiento crítico.

---

## 2. Objetivos del Proyecto

1.  **Especialización del Conocimiento:** Adaptar un LLM para dominar conceptos de estructuras de datos, recursividad y complejidad algorítmica.
2.  **Optimización de Recursos:** Lograr un entrenamiento efectivo en un entorno con recursos limitados (Google Colab, Tesla T4) utilizando cuantización de 4 bits.
3.  **Estilo Pedagógico:** Modificar el "tono" del modelo para que responda con explicaciones paso a paso y analogías, en lugar de solo generar código.
4.  **Portabilidad:** Exportar el modelo resultante a formato GGUF para su inferencia local offline.

---

## 3. Metodología Técnica

### 3.1. Selección del Modelo Base y Herramientas
Se seleccionó **Llama-3-8B-Instruct** debido a su alto rendimiento en benchmarks de razonamiento. Para viabilizar el entrenamiento en una GPU T4 (16GB VRAM), se utilizó la librería **Unsloth**, que permite una aceleración de hasta 2x en el entrenamiento y una reducción del 60% en el uso de memoria.

* **Frameworks:** PyTorch, Hugging Face (Transformers, TRL, PEFT).
* **Modelo:** `unsloth/llama-3-8b-Instruct-bnb-4bit`.

### 3.2. Configuración de LoRA (Low-Rank Adaptation)
En lugar de reentrenar los 8 billones de parámetros (lo cual sería computacionalmente inviable), se inyectaron adaptadores entrenables en las capas de atención del modelo.

* **Rango (r):** 16 (Balance entre capacidad de aprendizaje y eficiencia).
* **Alpha:** 16.
* **Módulos Objetivo:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Parámetros Entrenables:** 41,943,040 (Solo el **0.52%** del total del modelo).

### 3.3. Ingeniería de Datos
Se curó un dataset en formato **JSONL** siguiendo la estructura de instrucción tipo "Alpaca". El dataset incluyó:
* Explicaciones conceptuales con analogías (ej. "La recursividad son espejos enfrentados").
* Ejercicios de depuración de código.
* Instrucciones de pseudocódigo agnóstico al lenguaje.

---

## 4. Proceso de Entrenamiento

El ciclo de entrenamiento se ejecutó durante **120 pasos** (aproximadamente 10 épocas dado el tamaño del dataset), con un tamaño de lote efectivo de 8 (batch size 2 * gradient accumulation 4).

### 4.1. Análisis de Convergencia (Pérdida/Loss)
La curva de aprendizaje mostró un comportamiento excelente y estable:

* **Inicio (Step 1):** Loss ~2.73. El modelo tenía un conocimiento general pero no conocía el formato ni el estilo pedagógico específico.
* **Medio (Step 60):** Loss ~0.49. El modelo comenzó a asimilar la estructura de respuesta (Instrucción -> Input -> Tutor).
* **Final (Step 120):** Loss **0.0593**.
* **Interpretación:** La reducción drástica y sostenida de la función de pérdida indica que el modelo aprendió exitosamente a predecir las respuestas del dataset de entrenamiento sin sufrir divergencia. No se observaron picos anómalos (spikes), lo que sugiere un *Learning Rate* (1e-4) adecuado.
*(Referencia visual basada en los logs de entrenamiento: descenso monotónico de 2.7 a 0.06)*

---

## 5. Evaluación Cualitativa y Resultados

Una vez finalizado el entrenamiento, se realizaron pruebas de inferencia para validar la calidad de las respuestas.

**Prueba Realizada:**
> **Pregunta:** "No entiendo por qué mi función recursiva nunca termina"

**Respuesta Generada por el Tutor:**
> "Una función recursiva es como mirarte en dos espejos enfrentados: la imagen se repite hasta el infinito.
>
> **Paso a paso:**
> 1. Escribes el problema en términos de la solución más pequeña.
> 2. La llamada recursiva cambia el estado del problema.
> 3. Asegúrate de tener un **caso base** claro: una condición que detiene la recursividad.
>
> Si el caso base falta, la función se llama a sí misma sin avanzar realmente, como un espejo sin retroalimentación. [...]"

**Análisis del Resultado:**
La respuesta valida el éxito del proyecto:
1.  **Uso de Analogías:** El modelo aprendió a usar la metáfora de los "espejos" presente en el dataset.
2.  **Estructura:** Respetó el formato "Paso a paso" definido en los datos de entrenamiento.
3.  **Precisión Técnica:** Identificó correctamente la falta de un "caso base" como la causa del bucle infinito (Stack Overflow).

---

## 6. Implementación y Despliegue

Para garantizar la utilidad práctica del proyecto, el modelo fue sometido a un proceso de conversión y cuantización final.

* **Formato de Salida:** GGUF (GPT-Generated Unified Format).
* **Método de Cuantización:** `q4_k_m` (4-bit Medium).
* **Impacto:** El modelo original de ~16GB FP16 se redujo a un archivo de **~5.7GB**.
* **Viabilidad:** Este tamaño permite la ejecución local en tarjetas gráficas de gama media (ej. NVIDIA GTX 1060/1660 de 6GB) o incluso en CPU con suficiente memoria RAM, democratizando el acceso a la herramienta.

---

## 7. Conclusiones

El proyecto ha demostrado exitosamente que es posible crear herramientas educativas de alta calidad utilizando técnicas de **Fine-Tuning eficiente (PEFT)**. El "Tutor de Algoritmos" resultante no es simplemente un modelo que sabe programar, sino un modelo que sabe **enseñar**.

La combinación de **Unsloth** para la optimización del entrenamiento y **Llama-3** como base ha resultado en un sistema robusto, rápido y pedagógicamente coherente, listo para ser integrado en entornos de desarrollo (IDEs) o plataformas educativas.

---
