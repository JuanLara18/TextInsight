# src/gui.py
import streamlit as st

from .methods import (
    calcular_costo,
    calculate_top_n_grams,
    corregir_y_procesar_datos,
    estimar_tiempo_procesamiento,
    exportar_resultados,
    extract_project_info_from_file, generate_wordcloud,
    ngramas_a_dataframe, generar_temas,
    generar_grafo,
    obtener_descripcion_sensibilidad,
    show_analysis,
    visualizar_datos,
    generar_grafico_sentimientos,
    generar_grafico_confiabilidad,
    load_and_extract_data, 
    mostrar_analisis_sentimientos
)
from .connection import obtener_descripcion_modelo, generar_grafico_comparativo

st.session_state["modelo_seleccionado"] = "gpt-3.5-turbo"

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
    
    st.title("Bienvenido a TextInsight")
    # Lista de modelos disponibles para seleccionar
    st.write("Text Insight es una herramienta de análisis de texto impulsada por modelos de Lenguaje de Aprendizaje Profundo (LLM) de Inteligencia Artificial, diseñada para descifrar, interpretar y revelar patrones ocultos y tendencias significativas en datos textuales complejos.")
    modelos_disponibles = ["gpt-3.5-turbo", "gpt-4", "davinci", "gpt-4-32k"]
    # Desplegable para seleccionar el modelo
    st.session_state["modelo_seleccionado"] = st.selectbox("Selecciona el modelo de inteligencia artificial a utilizar:", modelos_disponibles)
    # Muestra la descripción del modelo seleccionado
    descripcion = obtener_descripcion_modelo(st.session_state.modelo_seleccionado)
    st.write(descripcion)
    
    # Genera y muestra el gráfico comparativo de los modelos
    plt = generar_grafico_comparativo()
    st.pyplot(plt)
    
def framework_project():
    st.title("Marco del Proyecto")
    
    # Preguntar al usuario si desea usar un archivo o un formulario
    opcion = st.radio("¿Cómo deseas ingresar la información del proyecto?", ("Cargar Archivo", "Formulario Manual"))
    
    # Inicializar variables para la información del proyecto
    pregunta_que_responde = ""
    proyecto_descripcion = ""
    palabras_clave = ""
    notas_adicionales = ""
    
    if opcion == "Cargar Archivo":
        # Opción para subir un archivo
        st.header("Cargar Archivo de Proyecto")
        uploaded_file = st.file_uploader("Carga un archivo de texto plano (.txt):", type=['txt'])
        
        if uploaded_file is not None:
            # Leer el archivo y extraer la información
            content = uploaded_file.getvalue().decode("utf-8")
            info = extract_project_info_from_file(content)
            
            pregunta_que_responde = info["pregunta_que_responde"]
            proyecto_descripcion = info["proyecto_descripcion"]
            palabras_clave = info["palabras_clave"]
            notas_adicionales = info["notas_adicionales"]
        
            st.success("Información extraída del archivo con éxito. Puedes revisar y modificar los campos a continuación.")
    
    elif opcion == "Formulario Manual":
        # Mostrar los campos de entrada (se usará tanto para archivo como para formulario manual)
        st.header("Pregunta que responde")
        pregunta_que_responde = st.text_area("Introduce la pregunta que responde el proyecto:", value=pregunta_que_responde)
        
        st.header("Descripción del Proyecto")
        proyecto_descripcion = st.text_area("Introduce una descripción detallada del proyecto:", value=proyecto_descripcion)
        
        st.header("Palabras Clave")
        palabras_clave = st.text_input("Introduce las palabras clave del proyecto (separadas por comas):", value=palabras_clave)
        
        st.header("Notas Adicionales")
        notas_adicionales = st.text_area("Introduce cualquier otra información relevante:", value=notas_adicionales)
        
        # Almacenar la información en el estado de sesión
        if st.button("Guardar Información"):
            try:
                st.session_state["pregunta_que_responde"] = pregunta_que_responde
                st.session_state["proyecto_descripcion"] = proyecto_descripcion
                st.session_state["palabras_clave"] = palabras_clave.split(',')
                st.session_state["notas_adicionales"] = notas_adicionales
                st.success("Información guardada con éxito.")
            except Exception as e:
                st.error(f"Error al guardar la información: {e}")
            
def data_loading_page():
    # Inicialización de variables de estado si es necesario
    if 'sensibilidad' not in st.session_state:
        st.session_state.sensibilidad = "Moderado"  # Valor por defecto
    if 'corregidos_df' not in st.session_state:  # inicialización de corregidos_df
        st.session_state.corregidos_df = None
        
    st.title("Taller de Datos")

    # Carga de archivos
    st.header("Cargar Datos")
    uploaded_file = st.file_uploader("Carga un archivo:", type=['xlsx', 'csv', 'sav', 'txt'], key='uploaded_file')
    
    if uploaded_file is not None:
        with st.spinner('Procesando datos...'):
            df, error_msg = load_and_extract_data(uploaded_file)
            if df is None:
                st.error(f"Error: {error_msg}")
                st.stop()
            else:
                st.session_state["df"] = df
                st.success("Datos cargados correctamente.")
        
        with st.expander("Ver datos cargados"):
            st.write(df)

    if 'df' in st.session_state and st.session_state.df is not None:
        st.header("Corrección y Preprocesamiento de Datos")
        st.write("Seleccione el nivel de corrección")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state["sensibilidad"] = st.selectbox("Nivel de corrección:", ["Ninguna", "Leve", "Moderado", "Exhaustivo"], index=["Ninguna", "Leve", "Moderado", "Exhaustivo"].index(st.session_state.sensibilidad))
            st.markdown(f"_{obtener_descripcion_sensibilidad(st.session_state.sensibilidad)}_")
        
        if 'df' in st.session_state:
            tokens_entrada = st.session_state["df"]['Originales'].apply(len).sum() / 4  # Aproximación de tokens
            tokens_salida = tokens_entrada  # Suponemos que la salida tiene la misma longitud que la entrada
            if st.session_state["sensibilidad"] == "Ninguna":
                costo = 0
                tiempo_estimado = 0
            else:
                costo = calcular_costo(tokens_entrada, tokens_salida, st.session_state["modelo_seleccionado"])
                tiempo_estimado = estimar_tiempo_procesamiento(st.session_state["df"], st.session_state["modelo_seleccionado"])
            

        with col2:
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Corregir y Procesar"):
                with st.spinner("Corrigiendo y procesando datos..."):
                    try:
                        contexto = {
                            "proyecto_nombre": st.session_state.get("proyecto_nombre", ""),
                            "proyecto_descripcion": st.session_state.get("proyecto_descripcion", ""),
                            "palabras_clave": st.session_state.get("palabras_clave", []),
                            "notas_adicionales": st.session_state.get("notas_adicionales", "")
                        }
                        st.session_state["corregidos_df"] = corregir_y_procesar_datos(st.session_state["df"], st.session_state["sensibilidad"], st.session_state["modelo_seleccionado"], contexto)
                        st.success("Corrección y procesamiento finalizados")
                    except Exception as e:
                        st.error(f"Error al corregir y procesar datos: {e}")
        
        if st.session_state.corregidos_df is not None:
            st.header("Visualización de Datos")
            visualizar_datos(st.session_state["corregidos_df"])

            # Mostrar análisis basado en una condición más relevante, como si 'Procesados' existe
            if 'Procesados' in st.session_state.corregidos_df.columns:
                show_analysis(st.session_state["corregidos_df"])

def analysis_page():
    st.title("Análisis de Datos")

    if st.session_state.corregidos_df is not None and 'Procesados' in st.session_state.corregidos_df.columns:
        df = st.session_state.corregidos_df

        # Expander para la nube de palabras
        with st.expander("Nube de Palabras"):
            st.markdown("Genera una visualización de las palabras más frecuentes en el texto procesado. Esto puede ayudarte a identificar rápidamente los temas principales y las palabras clave.")
            texto_procesado_para_nube = ' '.join(df['Procesados'].tolist())
            tokens_entrada = len(texto_procesado_para_nube.split())
            costo = 0  # No hay costo de API
            tiempo_estimado = 0.1  # 1 minuto como tiempo constante estimado
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Generar Nube de Palabras"):
                with st.spinner("Generando nube de palabras..."):
                    fig = generate_wordcloud([texto_procesado_para_nube])
                    st.pyplot(fig)

        # Expander para los n-gramas
        with st.expander("N-Gramas"):
            st.markdown("Los n-gramas son secuencias de n palabras consecutivas en el texto. Analizar los n-gramas puede ayudarte a identificar frases comunes y patrones lingüísticos.")
            n_value = st.number_input("Especifica el valor de n para los n-gramas", min_value=1, value=2, key='n_value_ngrams')
            top_n = st.slider("Selecciona cuántos n-gramas más comunes mostrar:", 1, 20, 5, key='top_n_ngrams')
            texto_procesado_para_ngramas = df['Procesados'].tolist()
            tokens_entrada = len(' '.join(texto_procesado_para_ngramas).split())
            costo = 0  # No hay costo de API
            tiempo_estimado = 0.1  # 1 minuto como tiempo constante estimado
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Generar N-Gramas"):
                with st.spinner("Generando n-gramas..."):
                    ngramas_resultado = calculate_top_n_grams(texto_procesado_para_ngramas, n_value, top_n)
                    df_ngramas = ngramas_a_dataframe(ngramas_resultado)
                    st.dataframe(df_ngramas)

        # Expander para el análisis de sentimientos
        with st.expander("Análisis de Sentimientos"):
            st.markdown("El análisis de sentimientos evalúa el tono emocional del texto. Esto es útil para comprender la opinión general y el sentimiento de los textos.")
            tokens_entrada = len(' '.join(df['Corregidos'].tolist()).split())
            tokens_salida = tokens_entrada  # Suponemos que la salida tiene la misma longitud que la entrada
            costo = calcular_costo(tokens_entrada, tokens_salida, st.session_state["modelo_seleccionado"])
            tiempo_estimado = estimar_tiempo_procesamiento(df, st.session_state["modelo_seleccionado"])
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Generar Análisis de Sentimientos"):
                with st.spinner("Generando análisis de sentimientos..."):
                    mostrar_analisis_sentimientos(st.session_state.corregidos_df)
                    fig_sentimientos = generar_grafico_sentimientos(st.session_state.corregidos_df)
                    fig_confiabilidad = generar_grafico_confiabilidad(st.session_state.corregidos_df)
                    st.pyplot(fig_sentimientos)
                    st.pyplot(fig_confiabilidad)

        # Expander para el grafo de n-gramas
        with st.expander("Grafo"):
            st.markdown("Genera un grafo basado en los n-gramas para visualizar las relaciones entre las palabras en el texto. Esto puede revelar estructuras y patrones en el uso del lenguaje.")
            n_value = st.number_input("Número de palabras a relacionar en el grafo", min_value=2, value=2, key='n_value_graph')
            min_weight = st.slider("Selecciona el mínimo número de menciones para mostrar:", 1, 10, 2, key='min_weight_graph')
            texto_procesado_para_grafo = df['Procesados'].tolist()
            tokens_entrada = len(' '.join(texto_procesado_para_grafo).split())
            costo = 0  # No hay costo de API
            tiempo_estimado = 0.2  # 2 minutos como tiempo constante estimado
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Generar Grafo"):
                with st.spinner("Generando grafo..."):
                    fig = generar_grafo(texto_procesado_para_grafo, n_value, min_weight)
                    if fig is not None:  # Verificar que se haya generado el grafo
                        st.pyplot(fig)
                    else:
                        st.write("No se generó ningún grafo. Intenta con un valor menor de n o ajusta el filtro de menciones.")

        # Expander para la generación de temas
        with st.expander("Temas"):
            st.markdown("Genera temas a partir del texto para agrupar los documentos en categorías significativas. Esto es útil para resumir grandes volúmenes de texto y entender mejor el contenido.")
            num_temas = st.number_input("Número de temas a generar:", min_value=1, value=5, step=1, key='num_temas')
            todas_las_frases = " ".join(df['Corregidos'].tolist())
            tokens_entrada = len(todas_las_frases.split())
            tokens_salida = tokens_entrada  # Suponemos que la salida tiene la misma longitud que la entrada
            costo = calcular_costo(tokens_entrada, tokens_salida, st.session_state["modelo_seleccionado"])
            tiempo_estimado = estimar_tiempo_procesamiento(df, st.session_state["modelo_seleccionado"])
            st.write(f"El costo estimado es: ${costo:.4f}")
            st.write(f"El tiempo estimado es: {tiempo_estimado:.2f} minutos")
            if st.button("Generar Temas"):
                with st.spinner("Generando temas..."):
                    df_temas = generar_temas(todas_las_frases, num_temas, st.session_state["modelo_seleccionado"])
                    st.session_state['df_temas'] = df_temas
                    st.dataframe(df_temas)

    else:
        st.write("Por favor, carga y procesa los datos en la pestaña 'Taller de Datos' antes de continuar con el análisis.")

def export_page():
    st.title("Exportar Resultados")
    st.write("Selecciona los elementos que deseas exportar:")

    opciones_exportacion = []
    if "df" in st.session_state:
        opciones_exportacion.append("Datos Originales y Corregidos")
    if "corregidos_df" in st.session_state and st.session_state.corregidos_df is not None:
        if "Procesados" in st.session_state.corregidos_df.columns:
            opciones_exportacion.append("Datos Procesados")
        opciones_exportacion.append("Análisis de Sentimientos")
        opciones_exportacion.append("N-Gramas")
        opciones_exportacion.append("Nube de Palabras")
        opciones_exportacion.append("Gráfico de Sentimientos")

    if not opciones_exportacion:
        st.info("No hay datos disponibles para exportar. Por favor, asegúrate de cargar y procesar los datos antes de intentar exportar.")
        return

    seleccionados = st.multiselect("Elige qué deseas exportar:", opciones_exportacion)
    
    if st.button("Exportar"):
        with st.spinner("Exportando..."):
            try:
                exportar_resultados(seleccionados)
                st.success("Exportación completada.")
            except Exception as e:
                st.error(f"Error al exportar resultados: {e}")

def run_app():
    page = st.sidebar.radio("Navegación", ["Inicio", "Marco del proyecto", "Taller de Datos", "Análisis de datos", "Exportar resultados"])

    if page == "Inicio":
        welcome_page()
    elif page == "Marco del proyecto":
        framework_project()
    elif page == "Taller de Datos":
        data_loading_page()
    elif page == "Análisis de datos":
        analysis_page()
    elif page == "Exportar resultados":
        export_page()