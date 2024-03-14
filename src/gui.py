# src/gui.py
import streamlit as st
from .controllers import distancias_palabras, load_and_extract_data, preparar_datos_para_analisis
from .methods import calculate_top_n_grams, corregir_frase, generate_wordcloud, ngramas_a_dataframe, generar_temas, generar_grafo, sensibilidad_a_comando, sentimientos
from .connection import obtener_descripcion_modelo, generar_grafico_comparativo

modelo_seleccionado = "gpt-3.5-turbo"
sensibilidad = 5
comando_sensibilidad = sensibilidad_a_comando(sensibilidad)

# Función para mostrar la página de bienvenida
def welcome_page():
    
     # Inicializa el estado si es necesario
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'df_procesado' not in st.session_state:
        st.session_state['df_procesado'] = None
    if 'reprocess_data' not in st.session_state:
        st.session_state['reprocess_data'] = False
    if 'sensibilidad' not in st.session_state:
        st.session_state['sensibilidad'] = 5
    
    st.title("Bienvenido a TextInsight")
    # Lista de modelos disponibles para seleccionar
    st.write("Text Insight es una herramienta de análisis de texto impulsada por modelos de Lenguaje de Aprendizaje Profundo (LLM) de Inteligencia Artificial, diseñada para descifrar, interpretar y revelar patrones ocultos y tendencias significativas en datos textuales complejos.")
    modelos_disponibles = ["gpt-4-32k", "gpt-4", "gpt-3.5-turbo", "davinci"]
    # Desplegable para seleccionar el modelo
    modelo_seleccionado = st.selectbox("Selecciona el modelo de inteligencia artificial a utilizar:", modelos_disponibles)
    
    # Muestra la descripción del modelo seleccionado
    descripcion = obtener_descripcion_modelo(modelo_seleccionado)
    st.write(descripcion)
    
    # Genera y muestra el gráfico comparativo de los modelos
    plt = generar_grafico_comparativo()
    st.pyplot(plt)

# Función para cargar datos y corregir frases
def data_loading_page():
    st.title("Taller de Datos")
    
    # Carga de archivos
    st.header("Cargar Datos")
    uploaded_file = st.file_uploader("Carga un archivo:", type=['xlsx', 'csv', 'sav', 'txt'], key='uploaded_file')
    
    # Verifica si hay un archivo cargado
    if uploaded_file is not None:
        with st.spinner('Procesando datos...'):
            st.session_state.df = load_and_extract_data(uploaded_file)
            if st.session_state.df is None:
                st.error("Formato de archivo no soportado o los datos no tienen el formato esperado.")
                st.stop()  # Detiene la ejecución si hay un error
            else:
                st.session_state.df_procesado = None  # Restablece los datos procesados ya que se cargó un nuevo archivo
                st.success("Datos cargados correctamente.")

    # Comprueba si los datos están listos para ser procesados
    if st.session_state.df is not None:
        st.header("Corrección y Preprocesamiento de Datos")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.sensibilidad = st.slider("Sensibilidad de la corrección:", 0, 10, st.session_state.sensibilidad, 1)
        
        with col2:
            if st.button("Corregir y Preprocesar Datos"):
                # Prepara el comando de sensibilidad para la corrección basado en la selección del usuario
                comando_sensibilidad = sensibilidad_a_comando(st.session_state.sensibilidad)
                
                # Aplica la corrección y preprocesamiento sólo una vez, no cada vez que se regresa a la página
                if 'df_procesado' not in st.session_state or st.session_state.reprocess_data:
                    st.session_state.df['Corregidos'] = st.session_state.df['Originales'].apply(lambda frase: corregir_frase(frase, comando_sensibilidad, modelo_seleccionado))
                    
                    st.session_state.df_procesado = preparar_datos_para_analisis(st.session_state.df)
                    print(st.session_state.df_procesado)  # Debug line to print the DataFrame

                    st.session_state.reprocess_data = False
                    st.success("Textos corregidos y preprocesados con éxito.")

        st.header("Visualización de Datos")
        # Asumiendo que la columna 'Procesados' se añade en 'preparar_datos_para_analisis'
        if st.session_state.df_procesado is not None and 'Procesados' in st.session_state.df_procesado.columns:
            num_rows = len(st.session_state.df_procesado)
            slider_val = st.slider("Selecciona cuántas filas mostrar:", 1, max(10, num_rows), min(10, num_rows))
            
            # Muestra las frases originales, corregidas y procesadas
            st.write(st.session_state.df_procesado[['Originales', 'Corregidos', "Procesados"]].head(slider_val))
            show_analysis(st.session_state.df_procesado.head(slider_val))
        else:
            st.error("Por favor, carga un archivo y realiza la corrección y el preprocesamiento.")

# Función para la página de análisis
def analysis_page():
    st.title("Análisis")

    if 'df_procesado' in st.session_state:
        df = st.session_state['df_procesado']

        # Columnas para nube de palabras y análisis de n-gramas
        col1, col2 = st.columns(2)

        with col1:
            st.header("Nube de Palabras")
            if st.button("Generar Nube de Palabras"):
                if 'Procesados' in df.columns:
                    texto_procesado_para_nube = ' '.join(df['Procesados'].tolist())
                    fig = generate_wordcloud([texto_procesado_para_nube])
                    st.pyplot(fig)
                else:
                    st.error("Por favor, preprocesa el texto antes de generar la nube de palabras.")

        with col2:
            st.header("N-Gramas")
            n_value = st.number_input("Especifica el valor de n para los n-gramas", min_value=1, value=2, key='n_value_ngrams')
            top_n = st.slider("Selecciona cuántos n-gramas más comunes mostrar:", 1, 10, 5, key='top_n_ngrams')
            if st.button("Generar N-Gramas"):
                if 'Procesados' in df.columns:
                    texto_procesado_para_ngramas = df['Procesados'].tolist()
                    ngramas_resultado = calculate_top_n_grams(texto_procesado_para_ngramas, n_value, top_n)
                    df_ngramas = ngramas_a_dataframe(ngramas_resultado)
                    st.dataframe(df_ngramas)
                else:
                    st.error("Por favor, preprocesa el texto antes de generar los n-gramas.")
        
        # Columnas para sentimientos y generar grafo
        col1, col2 = st.columns(2)

        with col1:
            # Segmento de Análisis de Sentimientos
            st.header("Sentimientos")
            if st.button("Generar Análisis de Sentimientos"):
                if 'Procesados' in df.columns:
                    resultados_sentimientos = sentimientos(df['Procesados'].tolist())
                    st.write(resultados_sentimientos)  # Asume que 'sentimientos' devuelve algo que streamlit puede mostrar directamente
                else:
                    st.error("Por favor, preprocesa el texto antes de realizar el análisis de sentimientos.")
        with col2:
            # Segmento para Generar Grafo
            st.header("Grafo")
            if st.button("Generar Grafo"):
                if 'Procesados' in df.columns:
                    figura_grafo = generar_grafo(df['Procesados'].tolist())  # Asume que esta función devuelve una figura de grafo
                    st.pyplot(figura_grafo)
                else:
                    st.error("Por favor, preprocesa el texto antes de generar el grafo.")
        # Columnas para sentimientos y generar grafo
          
        # Segmento para generar temas
        st.header("Temas")
        # Input para número de temas a generar
        num_temas = st.number_input("Número de temas a generar:", min_value=1, value=5, step=1, key='num_temas')
        if st.button("Generar Temas"):
            if 'Procesados' in df.columns:
                # Asumiendo que todas las frases corregidas están en df['Corregidos']
                todas_las_frases = " ".join(df['Corregidos'].tolist())
                # Llamar a la función que interactúa con ChatGPT para generar temas
                # Esta función debería retornar un nuevo DataFrame con las frases y sus temas asignados
                df_temas = generar_temas(todas_las_frases, num_temas, modelo_seleccionado)
                st.session_state['df_temas'] = df_temas  # Opcional: Guardar el nuevo DataFrame en el estado de la sesión
                st.dataframe(df_temas)
            else:
                st.error("Por favor, asegúrate de que el texto ha sido preprocesado.")
                    
    else:
        st.write("Carga y procesa datos en la pestaña 'Carga de Datos' para habilitar el análisis.")

def export_page():
    st.title("Exportar Resultados")
    st.write("Acá vamos tener los parámetros y exportar todos los resultados de acuerdo a los parámetros dados")



def run_app():
    page = st.sidebar.radio("Navegación", ["Inicio", "Taller de Datos","Análisis de datos", "Exportar resultados"])

    if page == "Inicio":
        welcome_page()
    elif page == "Taller de Datos":
        data_loading_page()
    elif page == "Análisis de datos":
        analysis_page()
    elif page == "Exportar resultados":
        export_page()