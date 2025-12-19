import pandas as pd
import numpy as np
import faiss
import re
import subprocess
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from textwrap import wrap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter

# -------------------------
# 1. CONFIGURACIÓN Y CONSTANTES
# -------------------------
CSV_PATH = "dataset_comentarios_LIMPIO.csv"
TEXT_COLUMN = "texto"
EMBED_MODEL = "all-mpnet-base-v2"  # Modelo denso de alta calidad
OLLAMA_MODEL = "mistral"           # Modelo de ollama
CHUNK_SIZE = 400
TOP_K = 5
NUM_CLUSTERS = 5                   # Para detectar temas ocultos

# Descargar stopwords si no existen
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("spanish"))

# -------------------------
# 2. LEXICÓN EMOCIONAL
# -------------------------
# Diccionario para detección fenomenológica de estados de ánimo
EMOTION_LEXICON = {
    "ansiedad_presion": [
        "ansiedad", "presión", "estrés", "agobio", "pánico", "miedo", 
        "nervios", "quemado", "burnout", "incertidumbre", "urgencia", "productivo"
    ],
    "frustracion_impotencia": [
        "frustración", "impotencia", "rabia", "harto", "imposible", 
        "bloqueo", "injusto", "cansancio", "agotamiento", "barrera"
    ],
    "vacio_confusion": [
        "vacío", "confusión", "perdido", "sin sentido", "nada", "hueco", 
        "extraño", "desorientado", "caos", "duda", "identidad", "efímero"
    ],
    "esperanza_resistencia": [
        "esperanza", "cambio", "futuro", "creer", "sueño", "ilusión", 
        "posibilidad", "construir", "libertad", "autonomía", "resistencia"
    ]
}

# -------------------------
# 3. FUNCIONES DE PROCESAMIENTO
# -------------------------

def limpiar_texto(texto):
    """Limpieza básica para normalizar el corpus."""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#\w+", "", texto)
    # Mantenemos caracteres latinos relevantes
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
    # Eliminación de stopwords
    texto = " ".join(w for w in texto.split() if w not in STOPWORDS)
    return texto

def extraer_glosario(df):
    """(Punto 4) Genera un glosario simple basado en frecuencia."""
    all_text = " ".join(df["texto_limpio"])
    words = [w for w in all_text.split() if len(w) > 3]
    common = Counter(words).most_common(10)
    print("\n [GLOSARIO INICIAL] Conceptos más frecuentes:")
    for word, freq in common:
        print(f"   - {word}: {freq}")
    print("-" * 50)

def detectar_emocion_dominante(textos_recuperados):
    """(Punto 14) Analiza el tono emocional de los chunks recuperados."""
    texto_completo = " ".join(textos_recuperados).lower()
    scores = {k: 0 for k in EMOTION_LEXICON.keys()}
    
    for emocion, palabras in EMOTION_LEXICON.items():
        for palabra in palabras:
            scores[emocion] += texto_completo.count(palabra)
    
    total_hits = sum(scores.values())
    if total_hits == 0:
        return "neutral/indefinido", scores
    
    emocion_dominante = max(scores, key=scores.get)
    return emocion_dominante, scores

# -------------------------
# 4. CARGA Y PIPELINE DE DATOS
# -------------------------
print(" Cargando datos...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COLUMN])
df["texto_limpio"] = df[TEXT_COLUMN].apply(limpiar_texto)

extraer_glosario(df)

print(" Fragmentando textos (Chunking)...")
chunks = []
# Guardamos metadatos originales si es necesario, aquí simplificado a lista de textos
for i, row in df.iterrows():
    for c in wrap(row["texto_limpio"], CHUNK_SIZE):
        chunks.append(c)
print(f"   Total de chunks procesados: {len(chunks)}")

# -------------------------
# 5. EMBEDDINGS Y CLUSTERING
# -------------------------
print(" Generando embeddings (SentenceTransformer)...")
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(chunks, show_progress_bar=True)

# --- (Punto 12) Clustering Temático ---
print(" Detectando clústeres temáticos (K-Means)...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# --- (Punto 13) Visualización Semántica ---
print(" Generando mapa semántico (t-SNE)...")
# Usamos t-SNE para proyectar a 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
vis_dims = tsne.fit_transform(embeddings[:2000]) # Limitamos a 2000 para velocidad en local si es necesario

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=vis_dims[:, 0], 
    y=vis_dims[:, 1],
    hue=cluster_labels[:2000],
    palette="viridis",
    s=15, alpha=0.7
)
plt.title("Mapa Semántico", fontsize=15)
plt.xlabel("Dimensión Latente 1")
plt.ylabel("Dimensión Latente 2")
plt.legend(title="Clúster Temático")
plt.savefig("mapa_semantico_gen_z.png")
print("    Imagen guardada: 'mapa_semantico_gen_z.png'")

# -------------------------
# 6. VECTOR STORE (FAISS)
# -------------------------
print(" Construyendo índice FAISS...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 7. MOTOR DE GENERACIÓN (OLLAMA)

def preguntar_ollama(contexto, pregunta, emocion_detectada, scores):
    detalle_emocional = ", ".join([f"{k}: {v}" for k,v in scores.items() if v > 0])
    
    # Analistas
    prompt = f"""
    [ROL]
    Eres un analista filosófico crítico. Utilizas marcos teóricos de:
    - Byung-Chul Han (cansancio, autoexplotación).
    - Zygmunt Bauman (modernidad líquida).
    - Michel Foucault (vigilancia, biopolítica).
    - Jean-François Lyotard (fin de los metarrelatos).
    - Martin Heidegger (la técnica como desocultamiento, autenticidad).
    - Jürgen Habermas (esfera pública, acción comunicativa).

    [EVALUACIÓN PREVIA]
    Evalúa del 1 al 10 qué tan relevante es el contexto recuperado para responder.

    [DATOS DEL SISTEMA]
    Emoción detectada: {emocion_detectada.upper()} ({detalle_emocional})

    [CONTEXTO RECUPERADO]
    {contexto}

    [PREGUNTA A RESPONDER]
    {pregunta}
    
    [FORMATO DE SALIDA]
    Responde en formato Markdown limpio:
    ### 1. Pertinencia del Contexto (Nota /10)
    [Breve justificación]
    
    ### 2. Análisis Filosófico
    [Desarrolla tu respuesta aquí citando a los autores pertinentes según la pregunta. 
    Ejemplo: si la pregunta es sobre "esfera pública", cita a Habermas. Si es sobre "ser", a Heidegger.]
    """

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="ignore"
    )
    return result.stdout.strip()

if __name__ == "__main__":
    # preguntas
    preguntas_investigacion = [
        "¿Qué expresiones o términos utiliza la Gen Z para describir el vacío existencial en redes sociales?",
        "¿Cómo influyen los algoritmos de recomendación en la construcción de su identidad?",
        "¿Qué emociones aparecen con mayor frecuencia cuando se habla de burnout o presión digital?",
        "¿La Gen Z percibe la autonomía como algo propio o como algo condicionado por la tecnología?",
        "¿Qué diferencias hay entre discursos auténticos vs discursos performativos en plataformas como TikTok?",
        "¿Existen patrones de lenguaje que indiquen crisis de sentido o desorientación vital?",
        "¿Cómo se refleja la idea de 'identidad líquida' en los datos recuperados?",
        "¿Qué menciones aparecen sobre libertad, control o manipulación algorítmica?",
        "¿Se observan señales de que los algoritmos crean deseos o hábitos?",
        "¿Qué temas o preocupaciones predominan en la conversación digital sobre propósito de vida?",
        "¿Hay evidencia de rechazo a los metarrelatos o valores tradicionales?",
        "¿Cómo aparece la figura del 'yo digital' en los textos analizados?",
        "¿Qué ejemplos concretos muestran pérdida del pensamiento crítico por efecto de la burbuja de filtros?",
        "¿Existen contrastes entre la visión que la Gen Z tiene de sí misma y lo que los datos sugieren?",
        "¿Qué rol juega la hiperconectividad en la ansiedad o depresión mencionada?",
        "¿Se observan patrones que apoyen las ideas de Byung-Chul Han sobre rendimiento y autoexplotación?",
        "¿Cómo interpretaría Foucault el régimen de vigilancia algorítmica detectado?",
        "¿Qué evidencias hay de que la tecnología 'desoculta' y transforma la vida según Heidegger?",
        "¿El espacio público digital está debilitado como afirma Habermas? ¿Qué muestran los datos?",
        "¿Cuáles son los principales miedos, frustraciones y esperanzas de la Gen Z frente al futuro?"
    ]

    print("\n" + "="*60)
    print(f" INICIANDO GENERACIÓN DE INFORME MASIVO ({len(preguntas_investigacion)} PREGUNTAS)")
    print("="*60)

    # Abrimos un archivo para guardar todo
    nombre_archivo = "Preguntas.md"
    
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        # Escribimos cabecera del informe
        f.write("# INFORME DE INVESTIGACIÓN: CRISIS DE SENTIDO EN LA GEN Z\n")
        f.write(f"Generado automáticamente por Sistema RAG con Modelo {OLLAMA_MODEL}\n")
        f.write("-" * 50 + "\n\n")

        for i, pregunta in enumerate(preguntas_investigacion, 1):
            print(f" Procesando {i}/{len(preguntas_investigacion)}: {pregunta[:40]}...")
            
            # --- Lógica RAG ---
            q_emb = embedder.encode([pregunta])
            D, I = index.search(q_emb, TOP_K)
            chunks_recuperados = [chunks[idx] for idx in I[0]]
            contexto_str = "\n---\n".join(chunks_recuperados)
            emocion, scores = detectar_emocion_dominante(chunks_recuperados)
            
            # Generar respuesta
            respuesta = preguntar_ollama(contexto_str, pregunta, emocion, scores)
            
            # --- Escribir en archivo ---
            f.write(f"## PREGUNTA {i}: {pregunta}\n\n")
            f.write(f"**Emoción detectada:** {emocion.upper()}\n\n")
            f.write(respuesta + "\n\n")
            f.write("-" * 50 + "\n\n")
            
            # Flush para asegurar que se guarda aunque se cancele
            f.flush() 

    print("\n" + "="*60)
    print(f" ¡PROCESO TERMINADO! Abre el archivo '{nombre_archivo}' para ver tu informe completo.")
    print("="*60)