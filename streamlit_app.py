import streamlit as st
from transformers import pipeline

# Título
st.title("Prueba de modelo HuggingFace")

# Descripción
st.write("Escribe un texto para analizar el sentimiento")

# Entrada de texto
text_input = st.text_area("Texto:", "")

# Cargar el modelo multilingüe
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

# Función para traducir la etiqueta
def interpretar_sentimiento(label):
    interpretaciones = {
        "1 star": "Muy negativo 😡",
        "2 stars": "Negativo 🙁",
        "3 stars": "Neutral 😐",
        "4 stars": "Positivo 🙂",
        "5 stars": "Muy positivo 😄"
    }
    return interpretaciones.get(label, "Desconocido")

# Botón de análisis
if st.button("Analizar"):
    if text_input.strip() != "":
        result = classifier(text_input)
        label = result[0]['label']
        score = result[0]['score']
        interpretacion = interpretar_sentimiento(label)

        # Mostrar resultado
        st.markdown(f"**Sentimiento detectado:** {interpretacion}")
        st.markdown(f"**Confianza:** {score:.2f}")
    else:
        st.warning("Por favor, escribe un texto para analizar.")
