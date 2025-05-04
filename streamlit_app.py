import streamlit as st
from transformers import pipeline

st.title("Prueba de modelo HuggingFace")

modelo = pipeline("sentiment-analysis")
texto = st.text_area("Escribe un texto para analizar el sentimiento")

if st.button("Analizar"):
    resultado = modelo(texto)
    st.write(resultado)
