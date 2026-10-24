"""
Punto de entrada del agente de reflexión para LinkedIn.

Se define un grafo de estado con LangGraph que alterna entre generar publicaciones y reflexcionar
sobre ellas hasta alcanzar un resultado satisfactorio. 
"""

# ------------------------------1. IMPORTACIONES---------------------------------

#Typed se usa para crear un diccionario con estructura fija, Annoted se usa para añadir metadatos, instruccioens especiales
from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

#BaseMessage es la clase que hereda todos los mensajes en LangChain (Humman, AI, System)
from langchain_core.messages import BaseMessage, HumanMessage

#State, es la clase principal, instructor de mapas, permite crear grafos
from langgraph.graph import END, StateGraph

from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain