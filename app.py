import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

api = pd.read_csv("ml.csv")

machine = LinearRegression()
x = api[["kg"]]
y = api[["vlr"]]

machine.fit(x, y)

st.title("Aprendendo Machine Learning")
st.divider()

kg = st.number_input("Qual o peso do aluno?")

if kg :
    preco_previssto = machine.predict([[kg]])[0][0]
    st.write(f"o valor do Quilo : { kg } e de : R$ { preco_previssto } ")