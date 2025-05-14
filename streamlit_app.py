import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# CSS personalizado
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f4f4f4;
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stMarkdown h1 {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("🔍 Análisis de Sentimiento Multilingüe")

st.write("Escribe una frase para analizar el sentimiento. El modelo devolverá una calificación entre 1 y 5 estrellas con nivel de confianza.")

text_input = st.text_area("✏️ Texto a analizar:", "")

# Cargar el modelo
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", return_all_scores=True)

classifier = load_model()

def interpretar_label(label):
    interpretaciones = {
        "1 star": "Muy Negativo 😡",
        "2 stars": "Negativo 🙁",
        "3 stars": "Neutral 😐",
        "4 stars": "Positivo 🙂",
        "5 stars": "Muy Positivo 😄"
    }
    return interpretaciones.get(label, "Desconocido")

if st.button("🔎 Analizar Sentimiento"):
    if text_input.strip():
        results = classifier(text_input)[0]  # Lista con resultados por clase
        df = pd.DataFrame(results)
        df['interpretacion'] = df['label'].map(interpretar_label)

        mejor = df.loc[df['score'].idxmax()]

        # Mostrar resultado principal
        st.markdown("## 📌 Resultado Principal")
        st.markdown(f"**Sentimiento:** `{interpretar_label(mejor['label'])}`")
        st.markdown(f"**Confianza del modelo:** `{mejor['score']:.2f}`")

        # Gráfica de barras
        st.markdown("## 📊 Distribución completa del modelo:")
        chart = alt.Chart(df).mark_bar(size=40).encode(
            x=alt.X('interpretacion', title='Sentimiento'),
            y=alt.Y('score', title='Confianza'),
            color=alt.Color('interpretacion', legend=None),
            tooltip=['label', 'score']
        ).properties(width=600, height=300)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("⚠️ Por favor, escribe una frase primero.")
