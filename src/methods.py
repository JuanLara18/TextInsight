# src/methods.py

import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re
import spacy
import Levenshtein as lev

# from itertools import chain
from typing import List
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.connection import generar_respuesta 

# Inicialización de spaCy para el procesamiento de texto en español
nlp = spacy.load('es_core_news_sm')

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

def sensibilidad_a_comando(sensibilidad: int) -> str:
    """Convierte el nivel de sensibilidad en un comando específico para el modelo."""
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
    return comandos.get(sensibilidad, "Realizar una corrección moderada.")  # Por defecto, si no se reconoce el nivel

def corregir_frase(frase: str, comando_sensibilidad: str) -> str:
    """Función que corrige individualmente cada frase."""
    prompt = f"{comando_sensibilidad} Corrige la siguiente frase: {frase}"
    # Aquí, reemplaza 'modelo_seleccionado' con el identificador de tu modelo actual
    respuesta_corregida = generar_respuesta('modelo_seleccionado', prompt)
    return respuesta_corregida

def corregir_frases(frases: List[str], sensibilidad: int) -> List[str]:
    """Aplica corrección a cada frase en la lista basado en la sensibilidad."""
    # Convertimos el nivel de sensibilidad a un comando entendible para el modelo
    comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    
    # Asegúrate de pasar el comando de sensibilidad a la función corregir_frase
    return [corregir_frase(frase, comando_sensibilidad) for frase in frases]


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

def sentimientos():
    pass

# Grafo ----------------------------------------------------------------

def generar_grafo():
    pass

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


