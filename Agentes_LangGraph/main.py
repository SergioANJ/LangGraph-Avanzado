"""
Punto de entrada del agente de reflexión para LinkedIn.

Se define un grafo de estado con LangGraph que alterna entre generar publicaciones y reflexcionar
sobre ellas hasta alcanzar un resultado satisfactorio. 
"""

# ------------------------------1. IMPORTACIONES---------------------------------

#Typed se usa para crear un diccionario con estructura fija, Annoted se usa para añadir metadatos, instruccioens especiales
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

#BaseMessage es la clase que hereda todos los mensajes en LangChain (Humman, AI, System), y HummanMessage representa los mensajes del usuario.
from langchain_core.messages import BaseMessage, HumanMessage

#State, es la clase principal, instructor de mapas, permite crear grafos
from langgraph.graph import END, StateGraph

#add_messages es una función que permite agregar mensajes a un grafo de estado en LangGraph.
from langgraph.graph.message import add_messages

#Importamos las cadenas de generación y reflexión para el agente de LinkedIn. Los prompts
from chains import generation_chain, reflection_chain


# ------------------------------2. DEFINICIÓN DEL GRAFO DE ESTADO---------------------------------
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------------------3. CREACIÓN DE LOS NODOS-----------------------------------------

#Identificadores de los nodos del grafo de estado.
GENERATE = "generate" # Nodo que genera o mejora la publicación en LinkedIn.
REFLECT = "reflect" # Nodo que reflexiona sobre la publicación en LinkedIn.

#Nodo de generación: Invoca la cadena de generación para crear o mejorar la publicación en LinkedIn, con el historial de mensajes previos.
#y devuelve el estado actualizado con la nueva publicación generada.
def generate_node(state: MessageGraph):
    # Invocamos la cadena de generación (prompt + LLM) con el historial.
    generated_response = generation_chain.invoke({"messages": state["messages"]})

    return {"messages": [generated_response]}

#Nodo de reflexión: Invoca la cadena de reflexión para evaluar la publicación en LinkedIn, con el historial de mensajes previos.
#y devuelve el estado actualizado con la reflexión realizada.
def reflect_node(state: MessageGraph):
    # Invocamos la cadena de reflexión (prompt + LLM) con todo el historial.
    res = reflection_chain.invoke({"messages": state["messages"]})
    #envolvemos la respuesta en HummanMessage para que el modelo la interprete como feedback del humano
    return {"messages": [HumanMessage(content=res.content)]}

# ------------------------------4. CREACIÓN DEL GRAFO DE ESTADO-----------------------------------------

#Creamos el grafo de estado con el esquema de MessageGraph.
builder = StateGraph(state_schema=MessageGraph)

#Agregamos los nodos al grafo de estado.
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)

#Definimios por donde el grafo comienza, que debe ser por el nodo de generación.
builder.set_entry_point(GENERATE)

# ------------------------5. CREACIÓN DE LAS ARISTAS DEL GRAFO DE ESTADO---------------------------------
#Debemos crear una función de decisión: para determinar si continua iterando o finaliza el proceso.
#Limitamos las iteraciones para evitar bucles infinitos 
def should_continue(state: MessageGraph):
    # si hay mas de 6 mensjaes, damos por finalizado el proceso
    if len(state["messages"]) > 6:
        return END
    # Si no, pasamos al node de reflexión para otra ronda de mejora
    return REFLECT

#Agregamos las aristas al grafo de estado.
#Arista condicional desde GENRATE: según should_continue, va a END o a REFLECT, y se define el path_map correspondiente. path_
builder.add_conditional_edges(GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END})

#arista fija: tras reflexionar, siempre volvemos a generar
builder.add_edge(REFLECT, GENERATE)

# ------------------------6. COMPILACIÓN DEL GRAFO DE ESTADO-----------------------------------------
graph = builder.compile()

#Mostramos la representación visual del grado (Mermaid y ASCII)
print(graph.get_graph().draw_mermaid()) #se visualiza en https://mermaidviewer.com/editor
print(graph.get_graph().draw_ascii()) #se visualiza en la consola

# ------------------------7. EJECUCIÓN DEL GRAFO DE ESTADO-----------------------------------------
if __name__ == "__main__":
    print("Hola LangGraph")

    #Entrada de ejemplo: un mensaje humano pidiendo mejorar una publicación
    inputs = {
        "messages": [
            HumanMessage(content="Por favor genera una públicación de LinkedIn enfocada a mi experiencia profesional")
        ]
    }

    #Invocamos el grafo completo con la entrada
    response = graph.invoke(inputs)

    print(response)