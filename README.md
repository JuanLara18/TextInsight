# TextInsight

TextInsight es una herramienta de análisis de texto impulsada por modelos de Lenguaje de Aprendizaje Profundo (LLM) de Inteligencia Artificial, diseñada para descifrar, interpretar y revelar patrones ocultos y tendencias significativas en datos textuales complejos.

## Características Principales

### 1. Página de Bienvenida
- **Selección del Modelo de Lenguaje**: Elige entre GPT-3.5 Turbo, GPT-4, Davinci, y GPT-4-32k.
- **Descripción del Modelo**: Muestra una breve descripción del modelo seleccionado.
- **Comparación de Modelos**: Visualización gráfica comparativa de los modelos disponibles.

### 2. Marco del Proyecto
- **Ingreso de Información del Proyecto**:
  - **Formulario Manual**: Introduce el nombre del proyecto, descripción, palabras clave y notas adicionales.
  - **Cargar Archivo**: Sube un archivo de texto (.txt) para extraer automáticamente la información del proyecto.
  
### 3. Taller de Datos
- **Carga de Archivos**: Soporta archivos en formatos Excel, CSV, SPSS y TXT.
- **Corrección y Preprocesamiento de Datos**:
  - Selecciona el nivel de corrección (Ninguna, Leve, Moderado, Exhaustivo).
  - Estima el costo y el tiempo de procesamiento antes de realizar la corrección.
  - Visualización de los datos cargados y corregidos.
  - Análisis de datos corregidos, incluyendo descripciones estadísticas y visualizaciones.

### 4. Análisis de Datos
- **Nube de Palabras**: Genera una visualización de las palabras más frecuentes en el texto procesado.
- **Análisis de N-Gramas**: Identifica frases comunes y patrones lingüísticos en el texto.
- **Análisis de Sentimientos**:
  - Evaluación del tono emocional del texto.
  - Visualización de la distribución de sentimientos y confiabilidad.
  - Promedio de confiabilidad por sentimiento.
- **Generación de Grafos**: Visualiza las relaciones entre palabras en el texto.
- **Identificación de Temas**: Agrupa documentos en categorías significativas para entender mejor el contenido.

### 5. Exportar Resultados
- **Exportación de Resultados**: Configuración y exportación de los resultados del análisis.

## Instalación

1. Clona este repositorio:
   ```sh
   git clone https://github.com/tu_usuario/TextInsight.git
   ```
2. Navega al directorio del proyecto:
   ```sh
   cd TextInsight
   ```
3. Crea y activa un entorno virtual:
   ```sh
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```
4. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

## Uso

1. Ejecuta la aplicación Streamlit:
   ```sh
   streamlit run src/gui.py
   ```
2. Accede a la aplicación en tu navegador en `http://localhost:8501`.

## Contacto

[Juan Lara - Ipsos](juan.lara@ipsos.com).