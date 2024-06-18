# src/controllers.py

###################################################################
####################### Generación de Prompts #####################
###################################################################

comandos = {
    "Ninguna": "No se realizará ninguna corrección.",
    "Leve": "Corrige únicamente errores ortográficos evidentes, como errores tipográficos o palabras mal escritas que alteren significativamente la comprensión del texto.",
    "Moderado": "Corrige ortografía, gramática y puntuación para que el texto esté correctamente estructurado según las reglas estándar del idioma.",
    "Exhaustivo": "Realiza una corrección exhaustiva incluyendo ortografía, gramática, estilo y claridad, y realiza mejoras sustanciales para optimizar la expresión y el impacto del texto."
}

def generar_prompt_con_contexto(frase: str, sensibilidad: str, contexto: dict) -> str:
    """
    Genera un prompt que incluye el contexto del proyecto para una corrección de frase específica.
    
    Args:
    - frase: Frase a corregir.
    - sensibilidad: Nivel de sensibilidad de la corrección.
    - contexto: Diccionario con el contexto del proyecto.
    
    Returns:
    - Prompt generado con contexto.
    """
    comando = comandos.get(sensibilidad, "")
    contexto_str = f"Nombre del proyecto: {contexto['proyecto_nombre']}\nDescripción del proyecto: {contexto['proyecto_descripcion']}\nPalabras clave: {', '.join(contexto['palabras_clave'])}\nNotas adicionales: {contexto['notas_adicionales']}\n"
    prompt = f"{contexto_str}\nRealiza una corrección de nivel {sensibilidad} entre Ninguna, Leve, Moderado y Exhaustivo. Siguiendo estas instrucciones: {comando} \nFrase a corregir: '{frase}'. \nPresenta SOLAMENTE el texto corregido, no añadas respuesta, texto o símbolos a la respuesta, tampoco el punto final."
    return prompt
