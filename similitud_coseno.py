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