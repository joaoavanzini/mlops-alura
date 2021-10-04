from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, json, jsonify
from textblob import TextBlob
import pandas as pd

df = pd.read_csv('casas.csv')
colunas = ['tamanho','ano','garagem']
# df = df[colunas]

x = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API"

@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    #tb_en = tb.translate(to="en")
    polaridade = tb.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True)