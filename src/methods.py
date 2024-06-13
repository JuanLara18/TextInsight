# src/methods.py

import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re
import spacy
import Levenshtein as lev
import streamlit as st
import numpy as np
import networkx as nx

from typing import List

from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline

from src.connection import generar_respuesta 

# Inicialización de spaCy para el procesamiento de texto en español
nlp = spacy.load('es_core_news_sm')
nlp_sentimientos = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# src/methods.py

comandos = {
    "Ninguna": "No se realizará ninguna corrección.",
    "Leve": "Corrige únicamente errores ortográficos evidentes, como errores tipográficos o palabras mal escritas que alteren significativamente la comprensión del texto.",
    "Moderado": "Corrige ortografía, gramática y puntuación para que el texto esté correctamente estructurado según las reglas estándar del idioma.",
    "Exhaustivo": "Realiza una corrección exhaustiva incluyendo ortografía, gramática, estilo y claridad, y realiza mejoras sustanciales para optimizar la expresión y el impacto del texto."
}

def generar_prompt_correccion(frase: str, nivel_sensibilidad: str) -> str:
    comando = comandos.get(nivel_sensibilidad, "")
    prompt = f"Realiza una corrección {nivel_sensibilidad.lower()} siguiendo estas instrucciones: {comando} \n Frase a corregir: '{frase}'. \n Presenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta, tampoco el punto final."
    return prompt

def sensibilidad_a_comando(sensibilidad: str) -> str:
    """Convierte el nivel de sensibilidad en un comando específico para el modelo."""
    return comandos.get(sensibilidad, "No se realizará ninguna corrección.")

def corregir_y_procesar_datos(df, sensibilidad, modelo_seleccionado):
    """
    Corrige y preprocesa los datos del DataFrame.
    
    Args:
    - df: DataFrame original con la columna 'Originales'.
    - sensibilidad: Nivel de sensibilidad para la corrección.
    - modelo_seleccionado: Modelo de AI seleccionado para la corrección.
    
    Returns:
    - DataFrame con columnas 'Originales', 'Corregidos' y 'Procesados'.
    """
    if sensibilidad == "Ninguna":
        df['Corregidos'] = df['Originales']
    else:
        df['Corregidos'] = df['Originales'].apply(
            lambda frase: corregir_frase(frase, sensibilidad, modelo_seleccionado)
        )
    df['Procesados'] = df['Corregidos'].apply(preprocesar_texto)
    return df

def visualizar_datos(df):
    """
    Muestra un DataFrame en Streamlit.
    
    Args:
    - df: DataFrame a mostrar.
    """
    st.write(df)



###################################################################
####################### Taller de datos ###########################
###################################################################

# Preprocesamiento ---------------------------------------------------

def preprocesar_texto(texto: str) -> str:
    """
    Preprocesa el texto aplicando tokenización, eliminación de stopwords,
    caracteres no ASCII, puntuación y normalización.
    """
    tokens = spacy_clean_and_tokenize(texto)
    tokens = remove_non_ascii(tokens)
    tokens = remove_punctuation(tokens)
    texto_procesado = ' '.join(tokens)
    return texto_procesado

def spacy_clean_and_tokenize(text: str) -> List[str]:
    """Limpieza y tokenización del texto utilizando spaCy."""
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

def remove_non_ascii(words: List[str]) -> List[str]:
    """Elimina caracteres no ASCII de la lista de palabras."""
    return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in words]

def remove_punctuation(words: List[str]) -> List[str]:
    """Elimina la puntuación de la lista de palabras."""
    return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']

# Corrección -----------------------------------------------------------

# def generar_prompt_correccion(frase: str, nivel_sensibilidad: int) -> str:
   
#     comando = comandos.get(nivel_sensibilidad, "")
#     prompt = f"Realiza una corrección de nivel {nivel_sensibilidad} entre 0 y 10 donde 0 es no hacerle nada a la frase y 10 es la correción perfecta. Siguiendo estas instrucciones: {comando} \n Frase a corregir: '{frase}'. \n Presenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta, tampoco el punto final."
#     return prompt

# def sensibilidad_a_comando(sensibilidad: int) -> str:
#     """Convierte el nivel de sensibilidad en un comando específico para el modelo."""
#     return comandos.get(sensibilidad, "Realizar una corrección moderada.")

# def obtener_descripcion_sensibilidad(n):
#     return comandos[n]

# def normalizar_texto(texto: str) -> str:
#     """
#     Normaliza el texto convirtiendo todo a minúsculas y eliminando espacios extras.
#     """
#     texto = texto.lower()  # Convertir a minúsculas
#     texto = re.sub(r'\s+', ' ', texto).strip()  # Eliminar espacios extras
#     return texto

# def corregir_frase(frase: str, sensibilidad: int, modelo_seleccionado) -> str:
#     """
#     Función que corrige individualmente cada frase de acuerdo al nivel de sensibilidad.
#     Si la sensibilidad es 0, devuelve la frase sin cambios.
#     """
#     if sensibilidad == 0:
#         return frase  # Función identidad para nivel 0
    
#     # Generar el prompt de corrección
#     prompt_correccion = generar_prompt_correccion(frase, sensibilidad)
    
#     respuesta_corregida = generar_respuesta(modelo_seleccionado, prompt_correccion)
#     # Normalización a minúsculas y verificación de la corrección (implementar según lo discutido anteriormente)
#     respuesta_corregida = normalizar_texto(respuesta_corregida)
#     if es_correccion_valida(frase, respuesta_corregida):
#         return respuesta_corregida
#     else:
#         return normalizar_texto(frase)

# def es_correccion_valida(original: str, corregido: str) -> bool:
#     """
#     Verifica si la corrección es válida, es decir, si no agrega elementos innecesarios y corrige de forma apropiada.
#     """
#     original_norm = normalizar_texto(original)
#     corregido_norm = normalizar_texto(corregido)

#     # Comprueba si se han realizado correcciones ortográficas y gramaticales sin añadir elementos adicionales
#     if corregido_norm == original_norm or '->' in corregido_norm:
#         return False
#     # Permite correcciones específicas como "q" por "que"
#     corregido_norm = corregido_norm.replace(" q ", " que ")
#     # Otras correcciones específicas podrían agregarse aquí
#     # ...

#     # Comparar la frase original y la corregida para validar la corrección
#     return corregido_norm != original_norm

# def corregir_y_validar_frase(frase: str, sensibilidad: int, modelo_seleccionado) -> str:
#     """
#     Corrige una frase, la valida y la normaliza.
#     """
#     comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
#     respuesta_corregida = corregir_frase(frase, comando_sensibilidad, modelo_seleccionado)
    
#     # Si la corrección no es válida, se devuelve la frase original normalizada
#     return respuesta_corregida if es_correccion_valida(frase, respuesta_corregida) else normalizar_texto(frase)

# def corregir_frases(frases: List[str], sensibilidad: int) -> List[str]:
#     """Aplica corrección a cada frase en la lista basado en la sensibilidad."""
#     # Convertimos el nivel de sensibilidad a un comando entendible para el modelo
#     comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    
#     # Asegúrate de pasar el comando de sensibilidad a la función corregir_frase
#     return [corregir_frase(frase, comando_sensibilidad) for frase in frases]

# def corregir_frases_por_lote(frases: List[str], sensibilidad: int, tamaño_lote=5, modelo_seleccionado="gpt-3.5-turbo") -> List[str]:
#     """Aplica la corrección a cada frase en la lista en lotes."""
#     # Convertimos el nivel de sensibilidad a un comando entendible para el modelo
#     comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    
#     # Crea una lista para almacenar las frases corregidas
#     frases_corregidas = []
    
#     # Divide las frases en lotes
#     for i in range(0, len(frases), tamaño_lote):
#         lote = frases[i:i+tamaño_lote]
        
#         # Crea un prompt de lote con todas las frases en el lote
#         lote_prompt = ' '.join([f"{comando_sensibilidad} Corrige la siguiente frase: {frase}" for frase in lote])
        
#         # Obtiene la respuesta del modelo para el lote
#         respuesta_lote = generar_respuesta(modelo_seleccionado, lote_prompt)
        
#         # Divide la respuesta del lote en frases individuales y las añade a la lista de frases corregidas
#         frases_corregidas.extend(respuesta_lote.split('\n'))
    
#     return frases_corregidas

import re
from typing import List
import unicodedata
from .connection import generar_respuesta

comandos = {
    "Ninguna": "No se realizará ninguna corrección.",
    "Leve": "Corrige únicamente errores ortográficos evidentes, como errores tipográficos o palabras mal escritas que alteren significativamente la comprensión del texto.",
    "Moderado": "Corrige ortografía, gramática y puntuación para que el texto esté correctamente estructurado según las reglas estándar del idioma.",
    "Exhaustivo": "Realiza una corrección exhaustiva incluyendo ortografía, gramática, estilo y claridad, y realiza mejoras sustanciales para optimizar la expresión y el impacto del texto."
}

def obtener_descripcion_sensibilidad(n):
    return comandos.get(n, "Nivel de corrección no especificado")

def normalizar_texto(texto: str) -> str:
    """
    Normaliza el texto convirtiendo todo a minúsculas y eliminando espacios extras.
    """
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\s+', ' ', texto).strip()  # Eliminar espacios extras
    return texto

def corregir_frase(frase: str, sensibilidad: str, modelo_seleccionado) -> str:
    """
    Función que corrige individualmente cada frase de acuerdo al nivel de sensibilidad.
    Si la sensibilidad es "Ninguna", devuelve la frase sin cambios.
    """
    if sensibilidad == "Ninguna":
        return frase  # Función identidad para nivel "Ninguna"
    
    # Generar el prompt de corrección
    prompt_correccion = generar_prompt_correccion(frase, sensibilidad)
    
    respuesta_corregida = generar_respuesta(modelo_seleccionado, prompt_correccion)
    # Normalización a minúsculas y verificación de la corrección
    respuesta_corregida = normalizar_texto(respuesta_corregida)
    if es_correccion_valida(frase, respuesta_corregida):
        return respuesta_corregida
    else:
        return normalizar_texto(frase)

def es_correccion_valida(original: str, corregido: str) -> bool:
    """
    Verifica si la corrección es válida, es decir, si no agrega elementos innecesarios y corrige de forma apropiada.
    """
    original_norm = normalizar_texto(original)
    corregido_norm = normalizar_texto(corregido)

    # Comprueba si se han realizado correcciones ortográficas y gramaticales sin añadir elementos adicionales
    if corregido_norm == original_norm or '->' in corregido_norm:
        return False
    # Permite correcciones específicas como "q" por "que"
    corregido_norm = corregido_norm.replace(" q ", " que ")
    # Otras correcciones específicas podrían agregarse aquí
    # ...

    # Comparar la frase original y la corregida para validar la corrección
    return corregido_norm != original_norm

def corregir_y_validar_frase(frase: str, sensibilidad: str, modelo_seleccionado) -> str:
    """
    Corrige una frase, la valida y la normaliza.
    """
    respuesta_corregida = corregir_frase(frase, sensibilidad, modelo_seleccionado)
    
    # Si la corrección no es válida, se devuelve la frase original normalizada
    return respuesta_corregida if es_correccion_valida(frase, respuesta_corregida) else normalizar_texto(frase)

def corregir_frases(frases: List[str], sensibilidad: str) -> List[str]:
    """Aplica corrección a cada frase en la lista basado en la sensibilidad."""
    # Asegúrate de pasar el comando de sensibilidad a la función corregir_frase
    return [corregir_frase(frase, sensibilidad) for frase in frases]

def corregir_frases_por_lote(frases: List[str], sensibilidad: str, tamaño_lote=5, modelo_seleccionado="gpt-3.5-turbo") -> List[str]:
    """Aplica la corrección a cada frase en la lista en lotes."""
    # Crea una lista para almacenar las frases corregidas
    frases_corregidas = []
    
    # Divide las frases en lotes
    for i in range(0, len(frases), tamaño_lote):
        lote = frases[i:i+tamaño_lote]
        
        # Crea un prompt de lote con todas las frases en el lote
        if sensibilidad == "Ninguna":
            frases_corregidas.extend(lote)
        else:
            lote_prompt = ' '.join([f"{comandos[sensibilidad]} Corrige la siguiente frase: {frase}" for frase in lote])
            
            # Obtiene la respuesta del modelo para el lote
            respuesta_lote = generar_respuesta(modelo_seleccionado, lote_prompt)
            
            # Divide la respuesta del lote en frases individuales y las añade a la lista de frases corregidas
            frases_corregidas.extend(respuesta_lote.split('\n'))
    
    return frases_corregidas

# Distancias -----------------------------------------------------------

def distancia_levenshtein(str1, str2):
    return lev.distance(str1, str2)

def distancia_jaccard(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))

def similitud_coseno_tfidf(str1, str2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    return cosine_similarity(tfidf_matrix)[0, 1]

def distancias_palabras(df):
    resultados = []
    
    # Calcula distancia de Levenshtein
    levenshtein_o_c = np.mean([distancia_levenshtein(o, c) for o, c in zip(df['Originales'], df['Corregidos'])])
    levenshtein_c_p = np.mean([distancia_levenshtein(c, p) for c, p in zip(df['Corregidos'], df['Procesados'])])
    levenshtein_o_p = np.mean([distancia_levenshtein(o, p) for o, p in zip(df['Originales'], df['Procesados'])])  # Nueva línea
    
    # Calcula distancia de Jaccard
    jaccard_o_c = np.mean([distancia_jaccard(o, c) for o, c in zip(df['Originales'], df['Corregidos'])])
    jaccard_c_p = np.mean([distancia_jaccard(c, p) for c, p in zip(df['Corregidos'], df['Procesados'])])
    jaccard_o_p = np.mean([distancia_jaccard(o, p) for o, p in zip(df['Originales'], df['Procesados'])])  
    # Prepara TF-IDF + Similitud del coseno
    cosine_o_c = np.mean([similitud_coseno_tfidf(o, c) for o, c in zip(df['Originales'], df['Corregidos'])])
    cosine_c_p = np.mean([similitud_coseno_tfidf(c, p) for c, p in zip(df['Corregidos'], df['Procesados'])])
    cosine_o_p = np.mean([similitud_coseno_tfidf(o, p) for o, p in zip(df['Originales'], df['Procesados'])])  
    
    resultados.append({
        "Método": "Levenshtein", 
        "Originales a Corregidos": levenshtein_o_c, 
        "Corregidos a Procesados": levenshtein_c_p,
        "Originales a Procesados": levenshtein_o_p  
    })
    resultados.append({
        "Método": "Jaccard", 
        "Originales a Corregidos": jaccard_o_c, 
        "Corregidos a Procesados": jaccard_c_p,
        "Originales a Procesados": jaccard_o_p  
    })
    resultados.append({
        "Método": "TF-IDF + Cosine", 
        "Originales a Corregidos": cosine_o_c, 
        "Corregidos a Procesados": cosine_c_p,
        "Originales a Procesados": cosine_o_p 
    })
    
    return pd.DataFrame(resultados)

def show_analysis(df):
    """
    Muestra el análisis de los cambios en las frases originales, corregidas y procesadas.
    
    Parámetros:
    - df (DataFrame): DataFrame con las columnas 'Originales', 'Corregidos', 'Procesados'.
    """
    st.header("Análisis de los cambios")
    st.markdown("En esta sección se presentan varios análisis de los datos procesados. A continuación, se muestran las métricas calculadas y sus explicaciones, seguidas de sugerencias para mejorar el análisis.")

    # Descripciones estadísticas del texto
    st.markdown("### Descripciones estadísticas del texto")
    st.markdown("Se presentan las descripciones estadísticas de las columnas 'Originales', 'Corregidos' y 'Procesados'. Estas estadísticas incluyen la longitud promedio del texto, la cantidad de datos nulos y la cantidad promedio de palabras por columna.")
    estadisticas = calcular_estadisticas(df)
    st.write(estadisticas)
    st.markdown("- **Longitud Promedio de los Textos**: Muestra la longitud promedio de los textos en cada columna. Esto puede ayudar a entender la complejidad y la extensión del contenido procesado.")
    st.markdown("- **Datos Nulos**: Indica la cantidad de valores nulos en cada columna. Es importante asegurarse de que no haya demasiados datos faltantes, ya que esto puede afectar la calidad del análisis.")
    st.markdown("- **Cantidad de Palabras Promedio**: Muestra la cantidad promedio de palabras en los textos de cada columna. Esto puede dar una idea de la densidad informativa de los textos.")

    
    st.markdown("### Cuadro de métricas")
    st.markdown("A continuación se presentan las métricas de distancia de Levenshtein, distancia de Jaccard y similitud del coseno con TF-IDF para comparar las columnas 'Originales', 'Corregidos' y 'Procesados'. Estas métricas ayudan a evaluar la similitud y las diferencias entre los textos en cada etapa del procesamiento.")
    st.write(distancias_palabras(df))
    st.markdown("- **Distancia de Levenshtein**: Mide el número mínimo de operaciones necesarias para transformar una cadena de caracteres en otra. Operaciones posibles incluyen inserciones, eliminaciones o sustituciones de un solo carácter. Una distancia menor indica que las frases son más similares. [Más información](https://en.wikipedia.org/wiki/Levenshtein_distance).")
    st.markdown("- **Distancia de Jaccard**: Mide la similitud entre dos conjuntos de datos. Se calcula como el tamaño de la intersección dividido por el tamaño de la unión de los conjuntos de palabras. Un valor más alto indica una mayor similitud. [Más información](https://en.wikipedia.org/wiki/Jaccard_index).")
    st.markdown("- **Similitud del Coseno con TF-IDF**: Evalúa la similitud entre dos textos en función de sus representaciones vectoriales. Un valor cercano a 1 indica que los textos tratan sobre temas muy similares, mientras que un valor cercano a 0 sugiere que hablan de temas distintos. [Más información sobre Similitud del Coseno](https://en.wikipedia.org/wiki/Cosine_similarity) y [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).")
    
    
    # Sugerencias para mejorar el análisis
    st.markdown("### Sugerencias para Mejorar el Análisis")
    st.markdown("- **Asegúrese de la Calidad del Texto**: Textos con muchos errores tipográficos o gramaticales pueden afectar negativamente los resultados del análisis. Utilice niveles de corrección adecuados.")
    st.markdown("- **Elija el Modelo Apropiado**: Dependiendo de la complejidad y el volumen de los textos, seleccionar un modelo más avanzado como GPT-4 puede proporcionar mejores resultados, aunque a un costo mayor.")
    #st.markdown("- **Considere el Contexto de los Textos**: Al analizar textos, es importante considerar el contexto en el que fueron escritos. Esto puede proporcionar insights adicionales y mejorar la interpretación de los resultados.")

    
###################################################################
###################### Análisis de datos ##########################
###################################################################

# N-gramas ------------------------------------------------------------

def calculate_top_n_grams(texto_procesado: List[str], n: int, top_n: int = 10) -> List[tuple]:
    """
    Calcula y devuelve los n-gramas más comunes de un texto procesado.
    """
    tokens = [token for frase in texto_procesado for token in frase.split()]
    n_grams = list(ngrams(tokens, n))
    n_grams_counts = Counter(n_grams)
    return n_grams_counts.most_common(top_n)

def ngramas_a_dataframe(ngramas_resultado: List[tuple]) -> pd.DataFrame:
    """Convierte una lista de n-gramas y sus frecuencias en un DataFrame."""
    return pd.DataFrame(ngramas_resultado, columns=['N-Grama', 'Frecuencia'])

# Nube de palabras -----------------------------------------------------

def generate_wordcloud(frases: List[str]):
    """Genera y devuelve una nube de palabras a partir de una lista de frases, con fondo blanco y mejorada gráficamente."""
    texto = ' '.join(frases)
    
    # Crear una nube de palabras con fondo blanco y especificando algunas configuraciones adicionales
    wordcloud = WordCloud(width=2048, height=1080, background_color='white', colormap='viridis').generate(texto)
    
    fig = plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear') # Usar interpolación bilinear para suavizar los bordes de las palabras
    plt.axis("off")
    plt.show() # Muestra la figura en lugar de cerrarla para poder visualizarla directamente
    
    return fig

# Generación de temas ---------------------------------------------------

def generar_temas(texto: str, n_temas: int, modelo: str) -> pd.DataFrame:
    """
    Interactúa con ChatGPT para definir n temas basados en el texto dado
    y asigna un tema a cada frase del texto.
    """
    prompt = f"Por favor, analiza el siguiente texto y define {n_temas} temas principales:\n{texto}"
    respuesta_chatgpt = generar_respuesta(modelo, prompt, max_tokens=1024)
    
    temas = respuesta_chatgpt.split(',')
    if len(temas) < n_temas:
        temas += ['Tema no especificado'] * (n_temas - len(temas))
    
    df_temas = pd.DataFrame({'Frase': texto.split('. '), 'Tema': [temas[i % n_temas] for i in range(len(texto.split('. ')))]})
    return df_temas

# Sentimientos ---------------------------------------------------------

def analisis_sentimientos_transformers(frases):
    """Aplica análisis de sentimientos utilizando la biblioteca transformers."""
    resultados_sentimientos = []
    for frase in frases:
        resultado = nlp_sentimientos(frase)[0]
        sentimiento = {
            '1 star': 'Muy Negativo',
            '2 stars': 'Negativo',
            '3 stars': 'Neutro',
            '4 stars': 'Positivo',
            '5 stars': 'Muy Positivo'
        }.get(resultado['label'], 'Neutro')  # Predeterminado a 'Neutro' si la etiqueta no coincide
        resultados_sentimientos.append({
            'Sentimiento': sentimiento,
            'Confiabilidad': resultado['score']
        })
    return resultados_sentimientos

import matplotlib.pyplot as plt
import seaborn as sns

def mostrar_analisis_sentimientos(df):
    """Muestra el análisis de sentimientos con un gráfico de distribución."""
    resultados_sentimientos = analisis_sentimientos_transformers(df['Corregidos'].tolist())
    
    df_sentimientos = pd.DataFrame(resultados_sentimientos)
    
    # Mostrar el DataFrame en Streamlit
    st.write(df_sentimientos)
    
    # Crear un gráfico de la distribución de sentimientos
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_sentimientos, x='Sentimiento', palette='viridis', order=['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo'])
    plt.title('Distribución de Sentimientos')
    plt.xlabel('Sentimiento')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Grafo ----------------------------------------------------------------

def ngramas_a_grafo(frases_procesadas, n):
    """Genera un grafo a partir de los n-gramas de las frases procesadas."""
    # Calcula los n-gramas
    ngramas_resultado = calculate_top_n_grams(frases_procesadas, n)
    
    # Crea un grafo vacío
    G = nx.Graph()
    
    # Añade los n-gramas al grafo
    for ngrama, frecuencia in ngramas_resultado:
        # Añade los nodos y las aristas con la frecuencia como peso
        G.add_edge(*ngrama, weight=frecuencia)
    
    return G


###################################################################
####################### Exportar datos ############################
###################################################################

def extract_project_info_from_file(file_content: str) -> dict:
    """
    Extrae la información del proyecto desde el contenido de un archivo de texto.

    Args:
    - file_content: Contenido del archivo de texto.

    Returns:
    - Un diccionario con el nombre del proyecto, descripción, palabras clave y notas adicionales.
    """
    info = {
        "proyecto_nombre": "",
        "proyecto_descripcion": "",
        "palabras_clave": "",
        "notas_adicionales": ""
    }
    
    lines = file_content.split('\n')
    
    for line in lines:
        if line.lower().startswith("nombre del proyecto:"):
            info["proyecto_nombre"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("descripción del proyecto:"):
            info["proyecto_descripcion"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("palabras clave:"):
            info["palabras_clave"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("notas adicionales:"):
            info["notas_adicionales"] = line.split(":", 1)[1].strip()
    
    return info

# def corregir_y_procesar_datos(df, sensibilidad, modelo_seleccionado):
#     """
#     Corrige y preprocesa los datos del DataFrame.
    
#     Args:
#     - df: DataFrame original con la columna 'Originales'.
#     - sensibilidad: Nivel de sensibilidad para la corrección.
#     - modelo_seleccionado: Modelo de AI seleccionado para la corrección.
    
#     Returns:
#     - DataFrame con columnas 'Originales', 'Corregidos' y 'Procesados'.
#     """
#     df['Corregidos'] = df['Originales'].apply(
#         lambda frase: corregir_frase(frase, sensibilidad, modelo_seleccionado)
#     )
#     df['Procesados'] = df['Corregidos'].apply(preprocesar_texto)
#     return df

# def visualizar_datos(df):
#     """
#     Muestra un DataFrame en Streamlit.
    
#     Args:
#     - df: DataFrame a mostrar.
#     """
#     st.write(df)

# src/methods.py

import numpy as np

def calcular_estadisticas(df):
    """
    Calcula descripciones estadísticas del DataFrame.
    
    Args:
    - df: DataFrame con columnas 'Originales', 'Corregidos' y 'Procesados'.
    
    Returns:
    - Un DataFrame con las estadísticas calculadas.
    """
    estadisticas = {}
    
    for col in df.columns:
        estadisticas[col] = {
            "Longitud Promedio": df[col].apply(len).mean(),
            "Datos Nulos": df[col].isnull().sum(),
            "Cantidad de Palabras Promedio": df[col].apply(lambda x: len(x.split())).mean()
        }
    
    return pd.DataFrame(estadisticas).transpose()


# src/methods.py

def calcular_costo(tokens_entrada, tokens_salida, modelo):
    precios = {
        "gpt-3.5-turbo": {"entrada": 0.0015, "salida": 0.002},
        "gpt-3.5-turbo-16k": {"entrada": 0.003, "salida": 0.004},
        "gpt-4": {"entrada": 0.03, "salida": 0.06},
        "gpt-4-32k": {"entrada": 0.06, "salida": 0.12}
    }
    costo_entrada = tokens_entrada / 1000 * precios[modelo]["entrada"]
    costo_salida = tokens_salida / 1000 * precios[modelo]["salida"]
    return costo_entrada + costo_salida


def estimar_tiempo_procesamiento(df, modelo_seleccionado):
    tiempo_por_token = 0.05  # 50 ms por token como suposición
    total_tokens = df['Originales'].apply(len).sum() / 4  # Aproximación de tokens
    tiempo_estimado = total_tokens * tiempo_por_token  # Tiempo en segundos

    # Ajuste según el modelo
    modelo_tiempos = {
        "gpt-3.5-turbo": 1,
        "gpt-3.5-turbo-16k": 1.5,
        "gpt-4": 2,
        "gpt-4-32k": 2.5
    }
    factor_modelo = modelo_tiempos.get(modelo_seleccionado, 1)
    tiempo_estimado *= factor_modelo

    return tiempo_estimado / 60  # Convertir a minutos
