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

from src.connection import generar_respuesta 

comandos = {
        0: "No realizar ninguna corrección.",
        1: "Corregir solo errores ortográficos muy obvios.",
        2: "Corregir errores ortográficos obvios.",
        3: "Corregir errores ortográficos y algunos errores gramaticales simples.",
        4: "Realizar correcciones ortográficas y gramaticales básicas.",
        5: "Corregir ortografía, gramática y puntuación.",
        6: "Corregir ortografía, gramática, puntuación y algunos errores de estilo.",
        7: "Realizar una corrección completa de ortografía, gramática, puntuación y estilo.",
        8: "Además de corregir todo lo anterior, sugerir mejoras en la claridad.",
        9: "Corregir y sugerir mejoras en claridad y estilo para hacer el texto más atractivo.",
        10: "Realizar una corrección exhaustiva, incluyendo ortografía, gramática, estilo, claridad, y sugerir mejoras para optimizar la expresión del texto al máximo."
    }

# Inicialización de spaCy para el procesamiento de texto en español
nlp = spacy.load('es_core_news_sm')

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

def generar_prompt_correccion(frase: str, nivel_sensibilidad: int) -> str:
    comandos = {
        0: "No se requiere acción alguna.",
        1: "Corrige únicamente errores ortográficos evidentes como errores tipográficos o palabras mal escritas que alteren significativamente la comprensión del texto.",
        2: "Corrige errores ortográficos claros sin cambiar el estilo o la estructura del texto.",
        3: "Realiza correcciones de errores ortográficos y errores gramaticales simples que no requieran reestructuración del texto.",
        4: "Corrige errores ortográficos y gramaticales básicos, manteniendo el significado original del texto.",
        5: "Corrige ortografía, gramática y puntuación para que el texto esté correctamente estructurado según las reglas estándar del idioma.",
        6: "Corrige la ortografía, la gramática, la puntuación y realiza ajustes menores de estilo para mejorar la legibilidad.",
        7: "Realiza una corrección completa de ortografía, gramática, puntuación y estilo, sin cambiar el significado original o añadir contenido nuevo.",
        8: "Además de realizar todas las correcciones anteriores, sugiere mejoras para clarificar el texto donde sea ambiguo o confuso.",
        9: "Corrige todo lo anterior y sugiere mejoras en la claridad y el estilo para hacer el texto más atractivo y cautivador.",
        10: "Realiza una corrección exhaustiva incluyendo ortografía, gramática, estilo y claridad, y sugiere mejoras sustanciales para optimizar la expresión y el impacto del texto."
    }
    
    comando = comandos.get(nivel_sensibilidad, "")
    prompt = f"Por favor, realiza una corrección de nivel {nivel_sensibilidad} siguiendo estas instrucciones: {comando} Corrige la frase: '{frase}'. Presenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta."
    return prompt

def sensibilidad_a_comando(sensibilidad: int) -> str:
    """Convierte el nivel de sensibilidad en un comando específico para el modelo."""
    return comandos.get(sensibilidad, "Realizar una corrección moderada.")

def obtener_descripcion_sensibilidad(n):
    return comandos[n]

def normalizar_texto(texto: str) -> str:
    """
    Normaliza el texto convirtiendo todo a minúsculas y eliminando espacios extras.
    """
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\s+', ' ', texto).strip()  # Eliminar espacios extras
    return texto

def corregir_frase(frase: str, sensibilidad: int, modelo_seleccionado) -> str:
    """
    Función que corrige individualmente cada frase de acuerdo al nivel de sensibilidad.
    Si la sensibilidad es 0, devuelve la frase sin cambios.
    """
    if sensibilidad == 0:
        return frase  # Función identidad para nivel 0
    
    # Generar el prompt de corrección
    prompt_correccion = generar_prompt_correccion(frase, sensibilidad)
    
    # Si el nivel de sensibilidad requiere corrección, llama al modelo para obtener la respuesta
    if sensibilidad > 0:
        respuesta_corregida = generar_respuesta(modelo_seleccionado, prompt_correccion)
        # Normalización a minúsculas y verificación de la corrección (implementar según lo discutido anteriormente)
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
    if corregido_norm == original_norm or 'corrige la siguiente frase' in corregido_norm:
        return False
    # Permite correcciones específicas como "q" por "que"
    corregido_norm = corregido_norm.replace(" q ", " que ")
    # Otras correcciones específicas podrían agregarse aquí
    # ...

    # Comparar la frase original y la corregida para validar la corrección
    return corregido_norm != original_norm

def corregir_y_validar_frase(frase: str, sensibilidad: int, modelo_seleccionado) -> str:
    """
    Corrige una frase, la valida y la normaliza.
    """
    comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    respuesta_corregida = corregir_frase(frase, comando_sensibilidad, modelo_seleccionado)
    
    # Si la corrección no es válida, se devuelve la frase original normalizada
    return respuesta_corregida if es_correccion_valida(frase, respuesta_corregida) else normalizar_texto(frase)

def corregir_frases(frases: List[str], sensibilidad: int) -> List[str]:
    """Aplica corrección a cada frase en la lista basado en la sensibilidad."""
    # Convertimos el nivel de sensibilidad a un comando entendible para el modelo
    comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    
    # Asegúrate de pasar el comando de sensibilidad a la función corregir_frase
    return [corregir_frase(frase, comando_sensibilidad) for frase in frases]


def corregir_frases_por_lote(frases: List[str], sensibilidad: int, tamaño_lote=5, modelo_seleccionado="gpt-3.5-turbo") -> List[str]:
    """Aplica la corrección a cada frase en la lista en lotes."""
    # Convertimos el nivel de sensibilidad a un comando entendible para el modelo
    comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    
    # Crea una lista para almacenar las frases corregidas
    frases_corregidas = []
    
    # Divide las frases en lotes
    for i in range(0, len(frases), tamaño_lote):
        lote = frases[i:i+tamaño_lote]
        
        # Crea un prompt de lote con todas las frases en el lote
        lote_prompt = ' '.join([f"{comando_sensibilidad} Corrige la siguiente frase: {frase}" for frase in lote])
        
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
    st.subheader("Análisis de los cambios")
    st.markdown("- La Distancia de Levenshtein es como contar cuántos errores de tipeo necesitarías corregir para hacer que un texto se convierta en el otro; menor número, más parecidos son. [Más información](https://en.wikipedia.org/wiki/Levenshtein_distance)")
    st.markdown("- La Distancia de Jaccard es como mirar dos listas de palabras y calcular qué porcentaje comparten; un porcentaje más alto significa que los textos tienen más palabras en común. [Más información](https://en.wikipedia.org/wiki/Jaccard_index)")
    st.markdown("- La Similitud del Coseno con TF-IDF evalúa qué tan parecidos son dos textos en cuanto a sus temas principales, no solo por las palabras exactas que usan. Un valor cercano a 1 indica que los textos tratan sobre temas muy similares, mientras que un valor cercano a 0 sugiere que hablan de temas distintos. [Más información sobre Similitud del Coseno](https://en.wikipedia.org/wiki/Cosine_similarity) y [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)")
    st.write(distancias_palabras(df))
    
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

def sentimientos(frases_procesadas, modelo_seleccionado):
    """Aplica análisis de sentimientos a cada frase en la lista utilizando el modelo de OpenAI."""
    # Mapeo de respuestas del modelo a valores numéricos
    sentimiento_a_numero = {
        "Muy negativo": -1,
        "Negativo": -0.5,
        "Neutral": 0,
        "Positivo": 0.5,
        "Muy positivo": 1,
    }
    
    # Lista para almacenar los resultados del análisis de sentimientos
    resultados_sentimientos = []
    
    for frase in frases_procesadas:
        # Generar respuesta del modelo
        respuesta_modelo = generar_respuesta(modelo_seleccionado, f"¿Cuál es el sentimiento de la siguiente frase?: {frase}")
        
        # Convertir la respuesta del modelo a un número y añadirlo a la lista de resultados
        sentimiento_numero = sentimiento_a_numero.get(respuesta_modelo, 0)  # Si la respuesta del modelo no está en el mapeo, asumir neutral (0)
        resultados_sentimientos.append(sentimiento_numero)
    
    return resultados_sentimientos

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