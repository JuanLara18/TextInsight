# src/gui.py
import streamlit as st
from .controllers import load_and_extract_data, preparar_datos_para_analisis
# Asegúrate de que el nombre de la función esté correcto según methods.py
from .methods import calculate_top_n_grams, generate_wordcloud, corregir_frases, ngramas_a_dataframe, generar_temas
from .connection import obtener_descripcion_modelo, generar_grafico_comparativo

modelo_seleccionado = ""

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
    st.title("Carga de Datos")
    uploaded_file = st.file_uploader("Carga un archivo (xlsx, csv, sav, txt):", type=['xlsx', 'csv', 'sav', 'txt'])
    
    if uploaded_file is not None:
        df = load_and_extract_data(uploaded_file)
        if df is not None:
            if st.button("Corregir Datos"):
                # Corrige frases originales y actualiza el estado de sesión
                df['Corregidos'] = corregir_frases(df['Originales'].tolist())
                st.session_state['df_procesado'] = df

            # Slider para seleccionar cuántas filas mostrar
            num_rows = len(df)
            slider_val = st.slider("Selecciona cuántas filas mostrar:", 1, max(10, num_rows), min(10, num_rows))
            
            # Muestra las frases originales y corregidas
            df_mostrar = st.session_state.get('df_procesado', df)
            st.write(df_mostrar[['Originales', 'Corregidos']].head(slider_val))
            st.success("Datos cargados correctamente. Utiliza 'Corregir Datos' para aplicar correcciones.")
        else:
            st.error("Error al cargar los datos. Asegúrate de que el formato del archivo es correcto.")

# Función para la página de análisis
def analysis_page():
    st.title("Análisis")
    
    if 'df_procesado' in st.session_state:
        df = st.session_state['df_procesado']
        
        # Botón para preprocesar el texto
        if st.button("Preprocesar Texto"):
            if 'Texto Procesado' not in df.columns:
                df = preparar_datos_para_analisis(df)
                st.session_state['df_procesado'] = df
                st.success("Texto preprocesado con éxito.")
            else:
                st.info("El texto ya ha sido preprocesado.")

        # Configuración de columnas para la interfaz de usuario
        col1, col2, col3 = st.columns([2,1,2])
        with col1:
            # Input para especificar el valor de n para n-gramas
            n_value = st.number_input("Especifica el valor de n para los n-gramas", min_value=1, value=2, key='n_value_ngrams')
        with col2:
            # Slider para seleccionar cuántos n-gramas mostrar
            top_n = st.slider("Selecciona cuántos n-gramas más comunes mostrar:", 1, 10, 5, key='top_n_ngrams')
        with col3:
            # Botón para generar n-gramas
            generate_ngram = st.button("Generar N-Gramas")

        # Generación de n-gramas y nube de palabras
        if generate_ngram:
            if 'Texto Procesado' in df.columns:
                texto_procesado_para_ngramas = df['Texto Procesado'].tolist()
                ngramas_resultado = calculate_top_n_grams(texto_procesado_para_ngramas, n_value, top_n)
                df_ngramas = ngramas_a_dataframe(ngramas_resultado)
                st.dataframe(df_ngramas)
            else:
                st.error("Por favor, preprocesa el texto antes de generar los n-gramas.")

        if st.button("Generar Nube de Palabras"):
            if 'Texto Procesado' in df.columns:
                texto_procesado_para_nube = ' '.join(df['Texto Procesado'].tolist())
                fig = generate_wordcloud([texto_procesado_para_nube])
                st.pyplot(fig)
            else:
                st.error("Por favor, preprocesa el texto antes de generar la nube de palabras.")
                
        # Input para número de temas a generar
        num_temas = st.number_input("Número de temas a generar:", min_value=1, value=5, step=1, key='num_temas')

        # Botón para generar temas
        if st.button("Generar Temas"):
            if 'Texto Procesado' in df.columns:
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


def run_app():
    page = st.sidebar.radio("Navegación", ["Bienvenida", "Carga de Datos", "Análisis"])

    if page == "Bienvenida":
        welcome_page()
    elif page == "Carga de Datos":
        data_loading_page()
    elif page == "Análisis":
        analysis_page()