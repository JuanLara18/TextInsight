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
   ```
   
4. Activa el entorno virtual:

   - En Windows:
     ```cmd
     .venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

5. Instala las dependencias del proyecto:
   
   ```bash
   pip install -r requirements.txt
   ```

### Configuración de la API Key de OpenAI

Debes establecer tu API key de OpenAI como una variable de entorno:

- En Windows, ejecuta:
  ```cmd
  set OPENAI_API_KEY=tu_api_key_aquí
  ```
- En macOS/Linux, ejecuta:
  ```bash
  export OPENAI_API_KEY=tu_api_key_aquí
  ```

## Ejecución

Para ejecutar la aplicación, asegúrate de que el entorno virtual esté activado y ejecuta:

```bash
streamlit run main.py
```

La aplicación debería abrirse automáticamente en tu navegador predeterminado.
