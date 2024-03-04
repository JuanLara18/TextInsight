# Text Insight

Text Insight es una avanzada herramienta de análisis de texto impulsada por modelos de Lenguaje de Aprendizaje Profundo (LLM) de Inteligencia Artificial, diseñada para descifrar, interpretar y revelar patrones ocultos y tendencias significativas en datos textuales complejos.

## Estructura de Archivos

El proyecto está organizado de la siguiente manera:

- `main.py`: Archivo principal que inicia la aplicación Streamlit.
- `src/`:
  - `gui.py`: Define la interfaz de usuario de la aplicación.
  - `methods.py`: Contiene las funciones de procesamiento y análisis de texto.
  - `controllers.py`: Gestiona la carga y preparación de datos.
  - `connections.py`: Maneja la conexión con la API de OpenAI y la generación de gráficos comparativos.
- `requirements.txt`: Lista todas las dependencias necesarias para ejecutar la aplicación.

## Configuración

### Requisitos Previos

Asegúrate de tener Python instalado en tu sistema. Este proyecto ha sido desarrollado con Python 3.8, pero debería ser compatible con versiones más recientes.

### Instalación

1. Clona este repositorio en tu máquina local.
2. Abre una terminal y navega al directorio del proyecto.
3. Crea un entorno virtual Python:
   
   ```bash
   python -m venv .venv
