# src/connection.py

import openai
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo especificado.
load_dotenv('src/openai_api.env')

# Obtener la clave API de OpenAI de las variables de entorno.
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No se encontró la clave API de OpenAI. "
                     "Por favor, verifica el archivo de configuración.")

# Asigna la clave API a la configuración de OpenAI para su uso en todo el módulo.
openai.api_key = api_key


def generar_respuesta(modelo_seleccionado, prompt, max_tokens=500):
    """
    Genera una respuesta del modelo seleccionado de OpenAI dado un prompt específico.

    Args:
        modelo_seleccionado (str): El identificador del modelo de OpenAI a utilizar.
        prompt (str): El prompt o texto de entrada para el modelo.
        max_tokens (int): El máximo número de tokens a generar en la respuesta.

    Returns:
        str: La respuesta generada por el modelo.
    """
    try:
        response = openai.ChatCompletion.create(
            model=modelo_seleccionado,
            messages=[
                {"role": "system", "content": "Ejecutar la siguiente tarea"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        raise openai.error.OpenAIError(f"Error al generar respuesta del modelo: {e}")


def obtener_descripcion_modelo(modelo_seleccionado):
    """
    Devuelve una descripción del modelo de OpenAI basado en su identificador.

    Args:
        modelo_seleccionado (str): El identificador del modelo de OpenAI.

    Returns:
        str: Descripción textual del modelo.
    """
    descripciones = {
        "gpt-3.5-turbo": "Ideal para respuestas rápidas y a bajo costo. Más económico pero menos detallado.",
        "gpt-4": "Máxima precisión y detalle en las respuestas. Ideal para complejidad pero con mayor costo.",
        "davinci": "Experto en comprender y generar texto complejo. Gran capacidad pero no el más rápido.",
        "gpt-4-32k": "Óptimo para trabajar con textos extensos. Maneja grandes volúmenes de datos, pero puede ser más lento."
    }
    return descripciones.get(modelo_seleccionado, "Modelo no especificado")

def generar_grafico_comparativo():
    """
    Crea un gráfico comparativo de los modelos de GPT en términos de velocidad, precisión, costo y capacidad, con un diseño más estético.
    
    Returns:
        matplotlib.pyplot: El objeto plt configurado con el gráfico comparativo.
    """
    datos = {
        'Modelos': ["GPT-3.5 Turbo", "GPT-4", "Davinci", "GPT-4-32k"],
        'Velocidad': [8, 6, 4, 5],  # Suposiciones para la velocidad
        'Precisión': [7, 9, 10, 8],  # Suposiciones para la precisión
        'Costo': [9, 5, 4, 3],  # Suposiciones para el costo
        'Capacidad': [6, 9, 10, 10]  # Suposiciones para la capacidad
    }
    df = pd.DataFrame(datos)
    
    # Crear el gráfico
    ax = df.plot(x='Modelos', kind='bar', stacked=False,
             title="Comparación de Modelos GPT",
             figsize=(10, 5),  # Ajustado para hacer la gráfica más pequeña verticalmente
             legend=True,
             color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    
    # Configurar estilos y etiquetas
    ax.set_ylabel('Puntuaciones', fontsize=12)
    ax.set_title('Comparación de Modelos GPT', fontsize=16, fontweight='bold', color='navy')
    ax.set_xticklabels(df['Modelos'], rotation=0, fontsize=12)
    ax.set_yticks(range(0, 11, 1))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)  # Solo líneas horizontales más sutiles
    ax.legend(title='Categorías', title_fontsize='13', fontsize='10', bbox_to_anchor=(1, 1))  # Mover leyenda fuera del gráfico
    ax.set_facecolor('white')  # Fondo blanco
    ax.spines['top'].set_visible(False)  # Ocultar borde superior
    ax.spines['right'].set_visible(False)  # Ocultar borde derecho
    ax.spines['bottom'].set_color('#dddddd')  # Suavizar el borde inferior
    ax.spines['left'].set_color('#dddddd')  # Suavizar el borde izquierdo
    
    plt.tight_layout()
    return plt