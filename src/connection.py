# src/connections.py
import openai
import matplotlib.pyplot as plt
import pandas as pd

# Establece tu API key de OpenAI aquí. Considera usar variables de entorno para mejorar la seguridad.
api_key = "sk-sa-ipsos-chile-jaime-vasquez-fPuvZVuufm9O2h8lOcoAT3BlbkFJxB5VafySZP9xoHZLlTij"
openai.api_key = api_key  # Inicializa la API key al cargar el módulo para simplificar

import openai

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