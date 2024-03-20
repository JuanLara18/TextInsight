# src/gui.py
import streamlit as st
import pandas as pd
import networkx as nx

from .controllers import load_and_extract_data
from .methods import calculate_top_n_grams, corregir_frase, corregir_y_validar_frase, generate_wordcloud, ngramas_a_dataframe, generar_temas, ngramas_a_grafo, obtener_descripcion_sensibilidad, preprocesar_texto, sensibilidad_a_comando, sentimientos, show_analysis
from .connection import obtener_descripcion_modelo, generar_grafico_comparativo

st.session_state.modelo_seleccionado = "gpt-3.5-turbo"

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
    modelos_disponibles = ["gpt-3.5-turbo", "gpt-4", "davinci", "gpt-4-32k"]
    # Desplegable para seleccionar el modelo
    st.session_state.modelo_seleccionado = st.selectbox("Selecciona el modelo de inteligencia artificial a utilizar:", modelos_disponibles)
    
    # Muestra la descripción del modelo seleccionado
    descripcion = obtener_descripcion_modelo(st.session_state.modelo_seleccionado)
    st.write(descripcion)
    
    # Genera y muestra el gráfico comparativo de los modelos
    plt = generar_grafico_comparativo()
    st.pyplot(plt)

# Función para cargar datos y corregir frases
def data_loading_page():
    # Inicialización de variables de estado si es necesario
    if 'sensibilidad' not in st.session_state:
        st.session_state.sensibilidad = 5  # Valor por defecto
    if 'corregidos_df' not in st.session_state:  # inicialización de corregidos_df
        st.session_state.corregidos_df = None
        
    st.title("Taller de Datos")
    
    # Inicialización de variables de estado si es necesario
    if 'sensibilidad' not in st.session_state:
        st.session_state.sensibilidad = 5  # Valor por defecto

    # Carga de archivos
    st.header("Cargar Datos")
    uploaded_file = st.file_uploader("Carga un archivo:", type=['xlsx', 'csv', 'sav', 'txt'], key='uploaded_file')
    
    if uploaded_file is not None:
        with st.spinner('Procesando datos...'):
            df = load_and_extract_data(uploaded_file)
            if df is None:
                st.error("Formato de archivo no soportado o los datos no tienen el formato esperado.")
                st.stop()
            else:
                st.session_state.df = df
                st.success("Datos cargados correctamente.")
        
        # Convertir el DataFrame a HTML y aplicar estilos CSS para centrar
        # df_html = df.to_html(classes='table table-striped', index=False)
        # centered_df_html = f"""
        # <div style="display: flex; justify-content: center; align-items: center;">
        #     {df_html}
        # </div>
        # """
        with st.expander("Ver datos cargados"):
        #  # Mostrar el DataFrame centrado usando HTML
        #  st.markdown(centered_df_html, unsafe_allow_html=True)
            st.write(df)

    if 'df' in st.session_state and st.session_state.df is not None:
        st.header("Corrección y Preprocesamiento de Datos")
        st.write("Configure la sensibilidad de la correción")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.sensibilidad = st.slider("Sensibilidad de la corrección:", 0, 10, st.session_state.sensibilidad, 1)
            st.markdown("_"+obtener_descripcion_sensibilidad(st.session_state.sensibilidad)+"_")
        
        with col2:
            if st.button("Corregir"):
                with st.spinner("Corrigiendo datos:"):
                    comando_sensibilidad = sensibilidad_a_comando(st.session_state.sensibilidad)
                    st.session_state.df['Corregidos'] = st.session_state.df['Originales'].apply(
                        lambda frase: corregir_frase(frase, st.session_state.sensibilidad, st.session_state.modelo_seleccionado)
                    )

                    st.session_state.corregidos_df = st.session_state.df  # Guarda el DataFrame corregido en un estado de sesión separado
                st.success("Corrección finalizada")
                    
            if st.button("Procesar datos"):
                with st.spinner("Procesando datos"):
                    if 'Corregidos' in st.session_state.corregidos_df.columns:  # Verifica si la columna 'Corregidos' existe en el DataFrame corregido
                        st.session_state.corregidos_df['Procesados'] = st.session_state.corregidos_df['Corregidos'].apply(preprocesar_texto)
                        st.success("Datos procesados")
                    else:
                        st.error("Por favor, primero presiona el botón 'Corregir'.")
                        
        st.header("Visualización de Datos")
        if st.session_state.corregidos_df is not None:
            num_rows = len(st.session_state.corregidos_df)
        else:
            num_rows = 0
            
        slider_val = st.slider("Selecciona cuántas filas mostrar:", 1, max(10, num_rows), min(10, num_rows))
        
        if st.session_state.corregidos_df is not None:
            st.write(st.session_state.corregidos_df.head(slider_val))

        # Mostrar análisis basado en una condición más relevante, como si 'Procesados' existe
        if st.session_state.corregidos_df is not None and 'Procesados' in st.session_state.corregidos_df.columns:
            show_analysis(st.session_state.corregidos_df)
                
# Función para la página de análisis
def analysis_page():
    st.title("Análisis")

    if st.session_state.corregidos_df is not None and 'Procesados' in st.session_state.corregidos_df.columns:
        df = st.session_state.corregidos_df

        # Crea un expander para la nube de palabras
        with st.expander("Nube de Palabras"):
            if st.button("Generar Nube de Palabras"):
                texto_procesado_para_nube = ' '.join(df['Procesados'].tolist())
                fig = generate_wordcloud([texto_procesado_para_nube])
                st.pyplot(fig)

        # Crea un expander para los n-gramas
        with st.expander("N-Gramas"):
            n_value = st.number_input("Especifica el valor de n para los n-gramas", min_value=1, value=2, key='n_value_ngrams')
            top_n = st.slider("Selecciona cuántos n-gramas más comunes mostrar:", 1, 10, 5, key='top_n_ngrams')
            if st.button("Generar N-Gramas"):
                texto_procesado_para_ngramas = df['Procesados'].tolist()
                ngramas_resultado = calculate_top_n_grams(texto_procesado_para_ngramas, n_value, top_n)
                df_ngramas = ngramas_a_dataframe(ngramas_resultado)
                st.dataframe(df_ngramas)

        # Crea un expander para el análisis de sentimientos
        with st.expander("Análisis de Sentimientos"):
            if st.button("Generar Análisis de Sentimientos"):
                # Aplicar análisis de sentimientos a las frases procesadas
                resultados_sentimientos = sentimientos(df['Procesados'].tolist(), st.session_state.modelo_seleccionado)
                
                # Crear un DataFrame con las frases originales y los resultados del análisis de sentimientos
                df_sentimientos = pd.DataFrame({
                    'Originales': df['Originales'],
                    'Sentimiento': resultados_sentimientos,
                })
                
                # Mostrar el DataFrame
                st.dataframe(df_sentimientos)

        # Crea un expander para generar el grafo
        with st.expander("Grafo"):
            n_value = st.number_input("Número de palabras a relacionar en el grafo", min_value=2, value=2, key='n_value_graph')
            if st.button("Generar Grafo"):
                # Genera el grafo
                G = ngramas_a_grafo(df['Procesados'].tolist(), n_value)
                
                # Dibuja el grafo
                nx.draw(G, with_labels=True)
                st.pyplot()

        # Crea un expander para generar temas
        with st.expander("Temas"):
            num_temas = st.number_input("Número de temas a generar:", min_value=1, value=5, step=1, key='num_temas')
            if st.button("Generar Temas"):
                todas_las_frases = " ".join(df['Corregidos'].tolist())
                df_temas = generar_temas(todas_las_frases, num_temas, st.session_state.modelo_seleccionado)
                st.session_state['df_temas'] = df_temas
                st.dataframe(df_temas)

    else:
        st.write("Por favor, carga y procesa los datos en la pestaña 'Carga de Datos' antes de continuar con el análisis.")


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