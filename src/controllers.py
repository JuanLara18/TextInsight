# src/controllers.py
import pandas as pd
import numpy as np
from .methods import preprocesar_texto, corregir_frase, distancia_levenshtein, distancia_jaccard, similitud_coseno_tfidf

# Carga y preparación de los datos ----------------------------

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

        # Aplicar corrección a cada frase en 'Originales'
        df['Corregidos'] = df['Originales'].apply(corregir_frase)

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
    df['Procesados'] = df['Corregidos'].apply(preprocesar_texto)
    return df

# Distancias -------------------------------------------------

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
