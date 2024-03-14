# src/controllers.py
import pandas as pd
import numpy as np
from .methods import preprocesar_texto, distancia_levenshtein, distancia_jaccard, similitud_coseno_tfidf

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