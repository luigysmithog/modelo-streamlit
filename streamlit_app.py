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

# Botón de análisis
if st.button("Analizar"):
    if text_input.strip() != "":
        result = classifier(text_input)
        label = result[0]['label']
        score = result[0]['score']

        # Mostrar resultado
        st.markdown(f"**Sentimiento detectado:** {label}")
        st.markdown(f"**Confianza:** {score:.2f}")
    else:
        st.warning("Por favor, escribe un texto para analizar.")
