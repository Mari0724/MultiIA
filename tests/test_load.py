from app.nlp.application.etl_load import load_data

# Datos de ejemplo simulando la información ya transformada
data = [
    {"pregunta": "¿Cuál es tu nombre?", "respuesta": "Soy el chatbot de Emilia 🤖"},
    {"pregunta": "¿Qué puedes hacer?", "respuesta": "Puedo responder preguntas y ayudarte con tareas."}
]

# Llamamos la función load_data
load_data(data)
