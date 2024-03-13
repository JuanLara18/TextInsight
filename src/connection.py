# src/connections.py
import openai
import os

from dotenv import load_dotenv

load_dotenv('src/openai_api.env')

api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No se encontró la clave API de OpenAI. Por favor, verifica el archivo de configuración.")

openai.api_key = api_key  # Inicializa la API key al cargar el módulo para simplificar

def generar_respuesta(modelo_seleccionado, prompt, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=modelo_seleccionado,
        messages=[{"role": "system", "content": "Ejecutar la siguiente tarea"},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content'].strip()


def obtener_descripcion_modelo(modelo_seleccionado):
    """
    Obtiene una descripción textual del modelo seleccionado.
    
    Args:
    - modelo_seleccionado: El identificador del modelo de OpenAI a describir.
    
    Returns:
    - Descripción del modelo seleccionado.
    """
    descripciones = {
        "gpt-3.5-turbo": "GPT-3.5-turbo: Óptimo para respuestas rápidas y eficientes en costos.",
        "gpt-4": "GPT-4: La última generación, ofrece respuestas más precisas y detalladas.",
        "davinci": "Davinci: Muy capaz en comprensión y generación de texto complejo."
    }
    return descripciones.get(modelo_seleccionado, "Modelo no especificado")

import matplotlib.pyplot as plt
import pandas as pd

def generar_grafico_comparativo():
    """
    Genera y retorna un gráfico comparativo de los modelos GPT en varias categorías.
    
    Returns:
    - Objeto plt con el gráfico generado.
    """
    datos = {
        'Modelos': ["GPT-3.5 Turbo", "GPT-4", "Davinci"],
        'Velocidad': [8, 6, 4],
        'Precisión': [7, 9, 10],
        'Costo': [9, 5, 4],
        'Capacidad': [6, 9, 10]
    }
    df = pd.DataFrame(datos)
    ax = df.plot(x='Modelos', kind='bar', stacked=False, title="Comparación de Modelos", figsize=(10, 6), legend=True, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel('Puntuaciones')
    ax.set_title('Comparación de Modelos GPT', fontsize=16, fontweight='bold', color='navy')
    ax.set_xticklabels(df['Modelos'], rotation=0, fontsize=12)
    ax.set_yticks(range(0, 11, 2))
    ax.grid(axis='y', linestyle='--', linewidth=0.7)
    ax.legend(title='Categorías', title_fontsize='13', fontsize='10', loc='upper right')
    ax.set_facecolor('white')
    plt.tight_layout()
    
    return plt