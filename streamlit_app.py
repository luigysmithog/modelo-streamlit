import streamlit as st
from transformers import pipeline

# Título
st.title("Prueba de modelo HuggingFace")

st.write("Escribe un texto para analizar el sentimiento (palabra o frase completa).")

text_input = st.text_area("Texto:", "")

# Cargar el modelo
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

def interpretar_sentimiento(label):
    interpretaciones = {
        "1 star": "Muy negativo 😡",
        "2 stars": "Negativo 🙁",
        "3 stars": "Neutral 😐 (no se percibe una emoción fuerte)",
        "4 stars": "Positivo 🙂",
        "5 stars": "Muy positivo 😄"
    }
    return interpretaciones.get(label, "Desconocido")

if st.button("Analizar"):
    if text_input.strip() != "":
        result = classifier(text_input)
        label = result[0]['label']
        score = result[0]['score']
        interpretacion = interpretar_sentimiento(label)

        st.markdown(f"**Interpretación del sentimiento:** {interpretacion}")
        st.markdown(f"**Confianza del modelo:** {score:.2f}")
    else:
        st.warning("Por favor, escribe un texto para analizar.")
