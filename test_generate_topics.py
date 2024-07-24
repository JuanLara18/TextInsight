from src.methods import generar_temas, cargar_frases  # Asegúrate de importar las funciones necesarias

def test_generar_temas():
    # Cargar frases desde el archivo de prueba
    archivo_prueba = "data/Frases.txt"
    texto = cargar_frases(archivo_prueba)
    print("Texto cargado:")
    print(texto)

    num_temas = 5
    modelo_seleccionado = "gpt-3.5-turbo"
    contexto = "Este es el contexto del proyecto."

    # Generar temas
    df_temas = generar_temas(texto, num_temas, modelo_seleccionado, contexto)
    print(df_temas)

if __name__ == "__main__":
    test_generar_temas()
