
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