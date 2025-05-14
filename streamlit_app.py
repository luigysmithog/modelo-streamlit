import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# Título
st.title("🔍 Análisis de Sentimiento Multilingüe")

st.write("Escribe una frase y analiza el sentimiento con calificación de 1 a 5 estrellas (modelo multilingüe).")

text_input = st.text_area("Texto a analizar:", "")

# Cargar el modelo con cache
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
    if text_input.strip() != "":
        results = classifier(text_input)[0]  # Lista con 5 resultados
        df = pd.DataFrame(results)
        df['interpretacion'] = df['label'].map(interpretar_label)

        # Obtener el resultado más confiable
        mejor = df.loc[df['score'].idxmax()]

        # Mostrar resultado principal
        st.markdown(f"### 📌 Resultado principal:")
        st.markdown(f"**Interpretación:** {interpretar_label(mejor['label'])}")
        st.markdown(f"**Confianza:** {mejor['score']:.2f}")

        # Mostrar gráfica
        st.markdown("### 📊 Distribución de puntuaciones:")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('interpretacion', title='Etiqueta'),
            y=alt.Y('score', title='Confianza'),
            color='interpretacion',
            tooltip=['label', 'score']
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("✏️ Escribe una frase para analizar.")
