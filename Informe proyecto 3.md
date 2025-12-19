# INFORME FINAL  
## Análisis Computacional del Discurso Digital y Crisis de Sentido en la Generación Z

---

## 1. Introducción

En los últimos años, las redes sociales se han consolidado como espacios centrales para la expresión subjetiva, la construcción identitaria y la manifestación del malestar generacional. En particular, la Generación Z ha utilizado estas plataformas para articular discursos relacionados con la ansiedad, el agotamiento, la incertidumbre frente al futuro y la pérdida de sentido vital. Sin embargo, el análisis de estos discursos presenta desafíos metodológicos debido al volumen, la heterogeneidad y el ruido inherente a los datos textuales provenientes de entornos digitales.

El presente informe tiene como objetivo desarrollar y documentar un **sistema computacional de análisis discursivo** basado en técnicas de **Procesamiento de Lenguaje Natural (NLP)**, **modelos de embeddings semánticos**, **clustering temático** y un enfoque **RAG (Retrieval-Augmented Generation)**, con el fin de explorar patrones de significado, emociones dominantes y tensiones sociotécnicas presentes en un corpus de comentarios digitales asociados a la Generación Z.

---

## 2. Descripción del Dataset

El corpus analizado se compone de un archivo CSV denominado `dataset_comentarios_reales.csv`, el cual contiene comentarios textuales publicados en entornos digitales. Cada registro incluye un campo textual principal que representa la unidad de análisis discursivo.
### Código para generar el dataset
```python

import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import itertools

# 2. CONFIGURACIÓN
# Pon aquí los IDs de los videos donde hay mucho debate (Videoensayos, noticias, etc.)
ids_videos = [
    "9S_Ni0AZdLU",  # La sociedad del cansancio
    "vm5tGIDUS9E",  # Crisis Gen Z
    "qAFwDmLkRQQ",  # Crisis existencial
    "SQur0vrKtVs",   # Podcast creativo
    "d2McnpfyWOA",
    "fw9bmsqmi24",
    "evyvHA2zqhQ",
    "qvICv53OXSE",
    "s1tz9nKQA2Q",
    "5OD3OB7CkKc"
]

CANTIDAD_POR_VIDEO = 2000 # ¿Cuántos comentarios quieres bajar por video?

downloader = YoutubeCommentDownloader()
datos_comentarios = []

print(f"Iniciando extracción de opiniones reales ({CANTIDAD_POR_VIDEO} por video)...")

# 3. EL BUCLE DE EXTRACCIÓN
for video_id in ids_videos:
    print(f"   --> Leyendo comentarios del video: {video_id}...")

    try:
        # Descargamos los comentarios (ordenados por popularidad)
        comentarios = downloader.get_comments_from_url(
            f'https://www.youtube.com/watch?v={video_id}',
            sort_by=SORT_BY_POPULAR
        )

        # Tomamos solo los primeros N comentarios (usamos itertools para cortar el flujo)
        for comment in itertools.islice(comentarios, CANTIDAD_POR_VIDEO):

            texto = comment['text']

            # Filtro de calidad: Ignoramos comentarios muy cortos (tipo "jajaja" o "First")
            if len(texto) < 15: continue

            datos_comentarios.append({
                "id": f"yt_comm_{comment['cid']}",  # ID único del comentario
                "fecha": "2024-01-01",              # Fecha aprox (la librería la da en formato raro, mejor estandarizar)
                "usuario": "anom_user_yt",          # Anonimizado
                "texto": texto,                     # EL TESTIMONIO REAL
                "tema": "Opinión Pública / Testimonio",
                "sentimiento": "real",              # Etiqueta para tu RAG
                "likes": comment.get('votes', 0),   # Cuánta gente apoyó este comentario
                "reposts": 0                        # YouTube no tiene reposts en comentarios
            })

    except Exception as e:
        print(f"    Error en video {video_id}: {e}")

# 4. GUARDAR
if datos_comentarios:
    df = pd.DataFrame(datos_comentarios)

    # Aseguramos que las columnas sean idénticas a tu dataset maestro
    columnas_finales = ["id", "fecha", "usuario", "texto", "tema", "sentimiento", "likes", "reposts"]
    df = df[columnas_finales]

    nombre_archivo = "dataset_comentarios_reales.csv"
    df.to_csv(nombre_archivo, index=False)

    print("\n" + "="*50)
    print(f" ¡LISTO! Se capturaron {len(df)} testimonios humanos.")
    print(f" Guardado en: {nombre_archivo}")
    print("="*50)

    # Descarga automática
    try:
        from google.colab import files
        files.download(nombre_archivo)
    except: pass

else:
    print(" No se encontraron comentarios.")
```
**Características generales del dataset:**
- Idioma: Español  
- Tipo de datos: Informativos
- Dominio: Discurso digital y expresiones subjetivas  
- Unidad de análisis: Comentarios individuales  

Previo al análisis, se eliminaron registros con valores nulos en la columna textual, garantizando la integridad mínima del corpus.

---

## 3. Metodología

La metodología se estructuró en varias fases encadenadas, conformando un pipeline reproducible de análisis textual avanzado.

### 3.1 Limpieza y normalización del texto
**Código para limpiar el .csv**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. CARGAMOS EL DATASET SUCIO
archivo_entrada = "dataset_comentarios_reales.csv" # O el que quieras limpiar
print(f"Cargando {archivo_entrada}...")
df = pd.read_csv(archivo_entrada)
print(f"Original: {len(df)} filas.")

# 2. VECTORIZACIÓN (Convertir texto a números para comparar)
# Usamos TF-IDF que es rápido y efectivo para esto
vectorizer = TfidfVectorizer().fit_transform(df['texto'].astype(str))
vectors = vectorizer.toarray()

# 3. CÁLCULO DE SIMILITUD (Todos contra todos)
print("Calculando similitudes (esto puede tardar unos segundos)...")
cosine_sim = cosine_similarity(vectors)

# 4. DETECTAR DUPLICADOS SEMÁNTICOS
umbral = 0.85  # Si se parecen más del 85%, se considera duplicado
indices_borrar = set()

# Recorremos la matriz (solo la mitad superior para no repetir)
filas, cols = cosine_sim.shape
for i in range(filas):
    if i in indices_borrar: continue # Si ya marcamos este como borrable, saltar
    
    for j in range(i + 1, filas):
        if j in indices_borrar: continue
        
        # Si la similitud supera el umbral
        if cosine_sim[i, j] > umbral:
            indices_borrar.add(j) # Marcamos 'j' para borrar, mantenemos 'i'

# 5. BORRAR Y GUARDAR
df_limpio = df.drop(list(indices_borrar))

print(f"\n RESULTADO FINAL:")
print(f"   Antes: {len(df)}")
print(f"   Después: {len(df_limpio)}")
print(f"   Se eliminaron {len(indices_borrar)} textos repetitivos o muy similares.")

df_limpio.to_csv("dataset_comentarios_LIMPIO.csv", index=False)
print(" Guardado como 'dataset_comentarios_LIMPIO.csv'")
```
Se aplicó un proceso de limpieza básica que incluyó:
- Conversión del texto a minúsculas.
- Eliminación de URLs, menciones a usuarios y hashtags.
- Conservación de caracteres propios del español.
- Eliminación de stopwords utilizando el corpus de NLTK para español.

Este proceso permitió reducir el ruido sin eliminar términos emocionalmente relevantes.

---

### 3.2 Construcción de un glosario inicial

A partir del texto limpio se generó un glosario exploratorio basado en frecuencia léxica. Este permitió identificar conceptos recurrentes en el discurso y validar empíricamente los principales ejes temáticos del corpus, tales como agotamiento, identidad, futuro e incertidumbre.
```python
def extraer_glosario(df):
    """(Punto 4) Genera un glosario simple basado en frecuencia."""
    all_text = " ".join(df["texto_limpio"])
    words = [w for w in all_text.split() if len(w) > 3]
    common = Counter(words).most_common(10)
    print("\n [GLOSARIO INICIAL] Conceptos más frecuentes:")
    for word, freq in common:
        print(f"   - {word}: {freq}")
    print("-" * 50)
```
---

### 3.3 Fragmentación del corpus (Chunking)

Debido a la longitud variable de los textos, el corpus fue fragmentado en segmentos de tamaño fijo (400 caracteres). Esta estrategia facilita la indexación semántica, la recuperación eficiente de contexto y la integración con sistemas RAG, manteniendo coherencia semántica suficiente para el análisis.

---

### 3.4 Generación de embeddings semánticos

Se empleó el modelo **SentenceTransformer `all-mpnet-base-v2`** para generar representaciones vectoriales densas de cada fragmento textual. Estos embeddings permiten capturar similitudes conceptuales profundas más allá de coincidencias léxicas superficiales.

---

### 3.5 Clustering temático no supervisado

Con el fin de identificar estructuras latentes en el discurso, se aplicó el algoritmo **K-Means** con cinco clústeres. Este enfoque no supervisado permitió detectar agrupaciones temáticas emergentes sin imponer categorías predefinidas.

---

### 3.6 Visualización semántica

Para facilitar la interpretación de los resultados, se utilizó **t-SNE** para proyectar los embeddings a un espacio bidimensional. El mapa semántico resultante permite observar la proximidad conceptual entre fragmentos y la separación entre clústeres temáticos.

---

### 3.7 Análisis emocional fenomenológico

Se diseñó un lexicón emocional personalizado, organizado en cuatro categorías principales:
- Ansiedad y presión  
- Frustración e impotencia  
- Vacío y confusión  
- Esperanza y resistencia  

Este lexicón permitió identificar emociones dominantes en los fragmentos recuperados, aportando una lectura fenomenológica del discurso.

---

### 3.8 Sistema RAG y análisis interpretativo

Se implementó un sistema **Retrieval-Augmented Generation (RAG)** utilizando FAISS como motor de búsqueda vectorial y un modelo generativo local (Ollama – Mistral). El sistema recupera fragmentos relevantes y genera análisis interpretativos apoyados en marcos teóricos de autores como Byung-Chul Han, Zygmunt Bauman, Michel Foucault, Martin Heidegger y Jürgen Habermas.

---

## 4. Resultados

Los resultados evidencian una alta presencia de discursos asociados al agotamiento emocional, la ansiedad por el rendimiento y la sensación de vacío existencial. Asimismo, se identifican tensiones entre la búsqueda de autonomía individual y la percepción de control algorítmico, así como rasgos compatibles con la noción de modernidad líquida y autoexplotación.

El análisis emocional muestra que, aunque predominan estados afectivos negativos, también emergen narrativas de resistencia y esperanza, lo que refleja un discurso complejo y multifacético.

---

## 5. Limitaciones

- El análisis emocional depende de un lexicón diseñado manualmente.
- Las interpretaciones generadas están mediadas por modelos de lenguaje.
- El corpus no es representativo de la totalidad de la Generación Z.
- El clustering temático no garantiza correspondencia exacta con categorías sociológicas formales.

---

## 6. Conclusiones

Este trabajo demuestra la viabilidad de un enfoque computacional riguroso para el análisis del discurso digital y el malestar generacional. La combinación de técnicas de NLP, embeddings semánticos, clustering y sistemas RAG permite identificar patrones lingüísticos y generar interpretaciones críticas fundamentadas.

El corpus resultante constituye una base sólida para investigaciones futuras en análisis de discurso, sociología digital y estudios críticos de la tecnología.

---

## 7. Trabajo futuro

- Validación del análisis emocional mediante anotadores humanos.
- Ampliación del corpus y comparación intergeneracional.
- Entrenamiento de modelos específicos para discurso sociotécnico.
- Integración del sistema en plataformas interactivas de análisis.
