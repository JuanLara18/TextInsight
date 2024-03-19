from src.controllers import load_and_extract_data
from src.methods import corregir_frase, sensibilidad_a_comando

# Definir el modelo y la sensibilidad para la corrección
modelo_seleccionado = "gpt-3.5-turbo"
sensibilidad = 5

# Cargar los datos
ruta_archivo = "../data/Frases_pocas.txt"
df = load_and_extract_data(ruta_archivo)

# Aplicar la corrección a las frases
if df is not None and 'Originales' in df.columns:
    comando_sensibilidad = sensibilidad_a_comando(sensibilidad)
    df['Corregidos'] = df['Originales'].apply(lambda frase: corregir_frase(frase, comando_sensibilidad, modelo_seleccionado))

    # Mostrar los resultados
    print(df)