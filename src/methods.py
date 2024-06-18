# src/methods.py

# Importaciones de bibliotecas estándar
import re
import io
import unicodedata
from collections import Counter
from typing import List

# Importaciones de bibliotecas externas
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import Levenshtein as lev
import streamlit as st
import numpy as np
import networkx as nx
from nltk.util import ngrams
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import seaborn as sns
from openpyxl.drawing.image import Image

# Importaciones de módulos internos
from .connection import generar_respuesta
from src.controllers import generar_prompt_con_contexto

# Inicialización de spaCy para el procesamiento de texto en español
nlp = spacy.load('es_core_news_sm')
nlp_sentimientos = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


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

# Carga y preparación de los datos ------------------------------------

def load_and_extract_data(file):
    """
    Carga datos desde un archivo subido y extrae el texto necesario.
    Soporta diferentes formatos de archivo como .csv, .xlsx, y .txt.
    Los archivos .sav están pendientes de implementación.
    
    Args:
    - file: Archivo subido por el usuario.
    
    Returns:
    - DataFrame con columnas 'Originales' y 'Corregidos' o None si el tipo de archivo no es soportado.
    """
    if file is not None:
        # Determinar el formato del archivo y cargarlo adecuadamente
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.sav'):
            # Implementación pendiente para archivos .sav
            pass
        elif file.name.endswith('.txt'):
            # Para archivos .txt, tratar cada línea como una entrada separada
            content = file.getvalue().decode("utf-8")
            lines = content.splitlines()
            df = pd.DataFrame(lines, columns=['Originales'])
        else:
            # Retornar None si el tipo de archivo no es soportado
            return None, "Unsupported file type"

        # Asegurar que la primera columna se trate como 'Originales' para archivos no .txt
        if not file.name.endswith('.txt'):
            df.columns = ['Originales'] + df.columns.tolist()[1:]
        return df
    return None

def preparar_datos_para_analisis(df):
    """
    Aplica preprocesamiento a las frases corregidas en el DataFrame.
    El texto procesado se utiliza para análisis pero no se añade como columna al DataFrame para evitar redundancia.
    
    Args:
    - df: DataFrame con las frases corregidas.
    
    Returns:
    - DataFrame original con una columna adicional 'Texto Procesado'.
    """
    # Preprocesar texto corregido para análisis
    if 'Corregidos' in df.columns:
        df['Procesados'] = df['Corregidos'].apply(preprocesar_texto)
    else:
        # Si la columna 'Corregidos' no existe, puedes optar por aplicar el procesamiento a otra columna
        # Por ejemplo, aplicar a 'Originales' o manejar de otra manera.
        df['Procesados'] = df['Originales'].apply(preprocesar_texto)
    return df


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

def corregir_frase(frase: str, sensibilidad: str, modelo_seleccionado: str, contexto: dict) -> str:
    """
    Función que corrige individualmente cada frase de acuerdo al nivel de sensibilidad,
    utilizando el contexto del proyecto.
    """
    if sensibilidad == "Ninguna":
        return frase  # No se realiza corrección

    # Generar el prompt de corrección con contexto
    prompt_correccion = generar_prompt_con_contexto(frase, sensibilidad, contexto)

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

def generar_prompt_correccion(frase: str, nivel_sensibilidad: str) -> str:
    comando = comandos.get(nivel_sensibilidad, "")
    prompt = f"Realiza una corrección {nivel_sensibilidad.lower()} siguiendo estas instrucciones: {comando} \n Frase a corregir: '{frase}'. \n Presenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta, tampoco el punto final."
    return prompt

def sensibilidad_a_comando(sensibilidad: str) -> str:
    """Convierte el nivel de sensibilidad en un comando específico para el modelo."""
    return comandos.get(sensibilidad, "No se realizará ninguna corrección.")

def corregir_y_procesar_datos(df: pd.DataFrame, sensibilidad: str, modelo_seleccionado: str, contexto: dict) -> pd.DataFrame:
    """
    Aplica la corrección y el preprocesamiento a todas las frases en el DataFrame,
    utilizando el contexto del proyecto.
    """
    df['Corregidos'] = df['Originales'].apply(lambda frase: corregir_frase(frase, sensibilidad, modelo_seleccionado, contexto))
    df['Procesados'] = df['Corregidos'].apply(preprocesar_texto)
    return df


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
    wordcloud = WordCloud(width=1280, height=720, background_color='white', colormap='viridis').generate(texto)
    
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

def mostrar_analisis_sentimientos(df):
    """
    Muestra el análisis de sentimientos con un gráfico de distribución.
    
    Args:
    - df: DataFrame con las frases corregidas.
    """
    resultados_sentimientos = analisis_sentimientos_transformers(df['Corregidos'].tolist())
    
    # Crear un DataFrame con los resultados y agregar la columna 'Originales'
    df_sentimientos = pd.DataFrame(resultados_sentimientos)
    df_sentimientos['Originales'] = df['Originales']
    
    # Reordenar las columnas para que 'Originales' aparezca primero
    df_sentimientos = df_sentimientos[['Originales', 'Sentimiento', 'Confiabilidad']]
    
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

    # Crear un gráfico de promedio de confiabilidad por sentimiento
    plt.figure(figsize=(10, 6))
    promedio_confiabilidad = df_sentimientos.groupby('Sentimiento')['Confiabilidad'].mean().reindex(['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo'])
    sns.barplot(x=promedio_confiabilidad.index, y=promedio_confiabilidad.values, palette='viridis')
    plt.title('Promedio de Confiabilidad por Sentimiento')
    plt.xlabel('Sentimiento')
    plt.ylabel('Promedio de Confiabilidad')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def generar_grafico_sentimientos(df_sentimientos):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_sentimientos, x='Sentimiento', palette='viridis', order=['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo'], ax=ax)
    ax.set_title('Distribución de Sentimientos')
    ax.set_xlabel('Sentimiento')
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

def generar_grafico_confiabilidad(df_sentimientos):
    fig, ax = plt.subplots(figsize=(10, 6))
    promedio_confiabilidad = df_sentimientos.groupby('Sentimiento')['Confiabilidad'].mean().reindex(['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo'])
    sns.barplot(x=promedio_confiabilidad.index, y=promedio_confiabilidad.values, palette='viridis', ax=ax)
    ax.set_title('Promedio de Confiabilidad por Sentimiento')
    ax.set_xlabel('Sentimiento')
    ax.set_ylabel('Promedio de Confiabilidad')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

# Grafo ----------------------------------------------------------------

def ngramas_a_grafo(frases_procesadas, n, min_weight=1):
    """Genera un grafo a partir de los n-gramas de las frases procesadas."""
    tokens = [token for frase in frases_procesadas for token in frase.split()]
    n_grams = list(ngrams(tokens, n))
    n_grams_counts = Counter(n_grams)
    
    G = nx.Graph()
    
    for ngrama, frecuencia in n_grams_counts.items():
        if frecuencia >= min_weight:  # Filtrar por frecuencia
            if len(ngrama) == n:
                for i in range(len(ngrama) - 1):
                    G.add_edge(ngrama[i], ngrama[i + 1], weight=frecuencia)
    
    return G

def generar_grafo(texto_procesado_para_grafo, n_value, min_weight):
    G = ngramas_a_grafo(texto_procesado_para_grafo, n_value, min_weight)
    if G.number_of_nodes() > 0:  # Verificar que el grafo no esté vacío
        fig, ax = plt.subplots()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, ax=ax, node_size=500, node_color='skyblue', font_size=10, width=[d['weight']*0.1 for (u, v, d) in G.edges(data=True)])
        return fig
    else:
        return None

###################################################################
####################### Exportar datos ############################
###################################################################

def exportar_resultados(seleccionados):
    # Crear un objeto de Excel
    with pd.ExcelWriter("Resultados_Exportados.xlsx", engine='openpyxl') as writer:
        # Exportar Datos Originales y Corregidos
        if "Datos Originales y Corregidos" in seleccionados and "df" in st.session_state:
            st.session_state["df"].to_excel(writer, sheet_name="Originales y Corregidos", index=False)

        # Exportar Datos Procesados
        if "Datos Procesados" in seleccionados and "corregidos_df" in st.session_state:
            st.session_state["corregidos_df"].to_excel(writer, sheet_name="Procesados", index=False)

        # Exportar Análisis de Sentimientos
        if "Análisis de Sentimientos" in seleccionados and "corregidos_df" in st.session_state:
            df_sentimientos = pd.DataFrame(analisis_sentimientos_transformers(st.session_state["corregidos_df"]['Corregidos'].tolist()))
            df_sentimientos['Originales'] = st.session_state["corregidos_df"]['Originales']
            df_sentimientos = df_sentimientos[['Originales', 'Sentimiento', 'Confiabilidad']]
            df_sentimientos.to_excel(writer, sheet_name="Sentimientos", index=False)

        # Exportar N-Gramas
        if "N-Gramas" in seleccionados and "corregidos_df" in st.session_state:
            texto_procesado_para_ngramas = st.session_state["corregidos_df"]['Procesados'].tolist()
            ngramas_resultado = calculate_top_n_grams(texto_procesado_para_ngramas, n_value=2, top_n=20)
            df_ngramas = ngramas_a_dataframe(ngramas_resultado)
            df_ngramas.to_excel(writer, sheet_name="N-Gramas", index=False)

        # Exportar Nube de Palabras y Gráfico de Sentimientos
        workbook = writer.book
        if "Nube de Palabras" in seleccionados or "Gráfico de Sentimientos" in seleccionados:
            imgdata = io.BytesIO()
            
            # Nube de Palabras
            if "Nube de Palabras" in seleccionados and "corregidos_df" in st.session_state:
                fig = generate_wordcloud([st.session_state["corregidos_df"]['Procesados'].tolist()])
                fig.savefig(imgdata, format='png')
                imgdata.seek(0)
                img = Image(imgdata)
                worksheet = workbook.create_sheet("Nube de Palabras")
                worksheet.add_image(img, 'A1')

            # Gráfico de Sentimientos
            if "Gráfico de Sentimientos" in seleccionados and "corregidos_df" in st.session_state:
                fig = plt.figure(figsize=(10, 6))
                df_sentimientos = pd.DataFrame(analisis_sentimientos_transformers(st.session_state["corregidos_df"]['Corregidos'].tolist()))
                sns.countplot(data=df_sentimientos, x='Sentimiento', palette='viridis', order=['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo'])
                plt.title('Distribución de Sentimientos')
                plt.xlabel('Sentimiento')
                plt.ylabel('Frecuencia')
                plt.xticks(rotation=45)
                fig.savefig(imgdata, format='png')
                imgdata.seek(0)
                img = Image(imgdata)
                worksheet = workbook.create_sheet("Gráfico de Sentimientos")
                worksheet.add_image(img, 'A1')

        writer.save()

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

###################################################################
########################## COSTOS #################################
###################################################################

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

    modelo_tiempos = {
        "gpt-3.5-turbo": 1,
        "gpt-3.5-turbo-16k": 1.5,
        "gpt-4": 2,
        "gpt-4-32k": 2.5
    }
    factor_modelo = modelo_tiempos.get(modelo_seleccionado, 1)
    tiempo_estimado *= factor_modelo

    return tiempo_estimado / 60  # Convertir a minutos
