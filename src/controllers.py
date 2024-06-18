# src/controllers.py

import pandas as pd
import streamlit as st
from .methods import preprocesar_texto, analisis_sentimientos_transformers

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



import matplotlib.pyplot as plt
import seaborn as sns

def mostrar_analisis_sentimientos(df):
    """Muestra el análisis de sentimientos con un gráfico de distribución."""
    resultados_sentimientos = analisis_sentimientos_transformers(df['Corregidos'].tolist())
    
    # Crear un DataFrame con los resultados
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

    
    
    
    
import pandas as pd
import streamlit as st
from .methods import preprocesar_texto, analisis_sentimientos_transformers

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
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.sav'):
            # Implementación pendiente para archivos .sav
            return None, "Unsupported file type"
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
        return df, None
    return None, "No file uploaded"


def generar_prompt_con_contexto(frase: str, sensibilidad: str, contexto: dict) -> str:
    """
    Genera un prompt que incluye el contexto del proyecto para una corrección de frase específica.
    """
    comando = comandos.get(sensibilidad, "")
    contexto_str = f"Nombre del proyecto: {contexto['proyecto_nombre']}\nDescripción del proyecto: {contexto['proyecto_descripcion']}\nPalabras clave: {', '.join(contexto['palabras_clave'])}\nNotas adicionales: {contexto['notas_adicionales']}\n"
    prompt = f"{contexto_str}\nRealiza una corrección de nivel {sensibilidad} entre Ninguna, Leve, Moderado y Exhaustivo. Siguiendo estas instrucciones: {comando} \nFrase a corregir: '{frase}'. \nPresenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta, tampoco el punto final."
    return prompt





