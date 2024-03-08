# src/gui.py
import streamlit as st
from .controllers import distancias_palabras, load_and_extract_data, preparar_datos_para_analisis
# Asegúrate de que el nombre de la función esté correcto según methods.py
from .methods import calculate_top_n_grams, generate_wordcloud, corregir_frases, ngramas_a_dataframe, generar_temas, generar_grafo, sentimientos
from .connection import obtener_descripcion_modelo, generar_grafico_comparativo

modelo_seleccionado = "gpt-3.5-turbo"

# Función para mostrar la página de bienvenida
def welcome_page():
    st.title("Bienvenido a TextInsight")
    # Lista de modelos disponibles para seleccionar
    st.write("Text Insight es una herramienta de análisis de texto impulsada por modelos de Lenguaje de Aprendizaje Profundo (LLM) de Inteligencia Artificial, diseñada para descifrar, interpretar y revelar patrones ocultos y tendencias significativas en datos textuales complejos.")
    modelos_disponibles = ["gpt-3.5-turbo", "gpt-4", "davinci"]
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
    st.header("Cargar de Datos")
    uploaded_file = st.file_uploader("Carga un archivo (xlsx, csv, sav, txt):", type=['xlsx', 'csv', 'sav', 'txt'])
    if uploaded_file is not None:
        st.success("Datos cargados correctamente.")
    
    if uploaded_file is not None:
        st.header("Corrección y Preprocesamiento de Datos")
        df = load_and_extract_data(uploaded_file)
        if df is not None:
            
            col1, col2 = st.columns(2)
                
            with col1:
                sensibilidad = st.slider("Sensibilidad de la corrección:", 0, 10, 5, 1)
            with col2:
                with st.container():
                    if st.button("Corregir Datos"):
                        # Corrige frases originales y actualiza el estado de sesión
                        df['Corregidos'] = corregir_frases(df['Originales'].tolist(), sensibilidad)
                        st.session_state['df_procesado'] = df      
                        st.success("Textos corregidos con éxito.")
                    
                    if st.button("Preprocesar Texto"):
                        df = preparar_datos_para_analisis(df) # Agregado por la función en (Procesado)
                        st.session_state['df_procesado'] = df
                        st.success("Textos preprocesados con éxito.")
               
            st.header("Visualización de Datos")
            # Slider para seleccionar cuántas filas mostrar
            num_rows = len(df)
            slider_val = st.slider("Selecciona cuántas filas mostrar:", 1, max(10, num_rows), min(10, num_rows))
            
            # Muestra las frases originales y corregidas
            df_mostrar = st.session_state.get('df_procesado', df)
            if "Procesados" in df_mostrar.columns:
                st.write(df_mostrar[['Originales', 'Corregidos', "Procesados"]].head(slider_val))
                st.subheader("Análisis de los cambios")
                st.markdown("- La Distancia de Levenshtein es como contar cuántos errores de tipeo necesitarías corregir para hacer que un texto se convierta en el otro; menor número, más parecidos son. [Más información](https://en.wikipedia.org/wiki/Levenshtein_distance)")
                st.markdown("- La Distancia de Jaccard es como mirar dos listas de palabras y calcular qué porcentaje comparten; un porcentaje más alto significa que los textos tienen más palabras en común. [Más información](https://en.wikipedia.org/wiki/Jaccard_index)")
                st.markdown("- La Similitud del Coseno con TF-IDF evalúa qué tan parecidos son dos textos en cuanto a sus temas principales, no solo por las palabras exactas que usan. Un valor cercano a 1 indica que los textos tratan sobre temas muy similares, mientras que un valor cercano a 0 sugiere que hablan de temas distintos. [Más información sobre Similitud del Coseno](https://en.wikipedia.org/wiki/Cosine_similarity) y [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)")
                st.write(distancias_palabras(df_mostrar))
            else:
                st.write(df_mostrar[['Originales', 'Corregidos']].head(slider_val))
            
        else:
            st.error("Error al cargar los datos. Asegúrate de que el formato del archivo es correcto.")

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