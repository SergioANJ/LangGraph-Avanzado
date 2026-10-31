"""
Cadenas (chains) para el agente de reflexión de linkedin

Este módulo define los prompts y cadenas de LangChain utilizados para generar
y evaluar publicaciones de LinkedIn mediante un proceso iterativo de reflexión.
"""

# la primera librería 'Chatprom..' permite definir roles en el prompt y entradas dinamicas {input}
# Es la memiria en LangGraph reserva un espcaio en la plantilla para inyectar historiales de chat completos
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importamos el modelo de lenguaje de OpenAI
from langchain_openai import ChatOpenAI

# ------------------------------ PLANTILLAS DE PROMPTS---------------------------------
# A. Plantilla de prompt para la fase de generación: el modelo crea o mejora publicaciones
generation_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define el modelo como asistente que escribe publicaciones
        (
            "system",
            (
                "Eres un asistente de influencer tecnológico de LinkedIn encargado de escribir excelentes publicaciones. "
                "Genera la mejor publicación posible según la petición del usuario. "
                "Si el usuario aporta crítica, responde con una versión revisada de tus intentos anteriores."
            ),
        ),
        # Marcador para el historial de mensajes
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# B. Plantilla de prompt para la fase de reflexión: el modelo actúa como evaluador
# Generando crítica y recomendaciones sobre la publición del usuario
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define el rol y comportamiento del modelo para el nodo de relfexión
        (
            "system",
            (
                "Eres un influencer viral de LinkedIn que evalúa publicaciones. Genera crítica y recomendaciones para la publicación del usuario",
                "Siempre proporciona recomendaciones detalladas, incluyendo aspectos como longitud, viralidad, estilo, etc.",
            ),
        ),
        # Marcador para el historial del mensaje
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ------------------------------LLAMADA AL MODELO---------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ------------------------------CREACIÓN DE CADENAS-------------------------------

# cadena de generación: combina el prompt de generación con el LLM mediante el operadaor |
# Flujo: Generation_prompt formates los mensaje: lLM genera la respuesta
generation_chain = generation_prompt | llm

# Cadena de reflexión : combina el prompt de reflexion con el LLM
# Flujo: reflection_prompt formatea los mensajes: LLM genera crítica y recomendaciones
reflection_chain = reflection_prompt | llm
