# src/methods.py
from itertools import chain
from typing import List
import unicodedata
import re
import nltk
from nltk.util import ngrams
from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein as lev

from src.connection import generar_respuesta  # Asegúrate de que el import sea correcto, podría ser .connection dependiendo de tu estructura

# Inicialización de spaCy para el procesamiento de texto en español
nlp = spacy.load('es_core_news_sm')

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

def calculate_top_n_grams(texto_procesado: List[str], n: int, top_n: int = 10) -> List[tuple]:
    """
    Calcula y devuelve los n-gramas más comunes de un texto procesado.
    """
    tokens = [token for frase in texto_procesado for token in frase.split()]
    n_grams = list(ngrams(tokens, n))
    n_grams_counts = Counter(n_grams)
    return n_grams_counts.most_common(top_n)

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

def corregir_frase(frase: str) -> str:
    """Función placeholder para corrección de frases. Actualmente devuelve la entrada sin cambios."""
    return frase

def corregir_frases(frases: List[str], sensibilidad) -> List[str]:
    """Aplica corrección a cada frase en la lista."""
    return [corregir_frase(frase) for frase in frases]

def ngramas_a_dataframe(ngramas_resultado: List[tuple]) -> pd.DataFrame:
    """Convierte una lista de n-gramas y sus frecuencias en un DataFrame."""
    return pd.DataFrame(ngramas_resultado, columns=['N-Grama', 'Frecuencia'])

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

def sentimientos():
    pass

def generar_grafo():
    pass