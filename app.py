# implementation of a neural network to predict future stock values using the algorithm of Long Short-Term Memory (LSTM)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

banco = 'BBAS3.SA' # Banco do Brasil

inicio = dt.datetime(2012,1,1) # data de inicio
fim = dt.datetime(2022,11,18) # data de hoje

dados = web.DataReader(banco, 'yahoo', inicio, fim) # pega os dados do yahoo finance

# Preparação dos dados
normalizando = MinMaxScaler(feature_range=(0,1)) # normalizando os dados entre 0 e 1
dados_normalizados = normalizando.fit_transform(dados['Close'].values.reshape(-1,1)) # normalizando os dados de fechamento 

previsao_dias = 3 # dias para prever

x_treinar, y_treinar = [], []

# pegando os dados de fechamento e colocando em x_treinar e y_label
for x in range(previsao_dias, len(dados_normalizados)):
    x_treinar.append(dados_normalizados[x-previsao_dias:x, 0])
    y_treinar.append(dados_normalizados[x, 0])

x_treinar, y_treinar = np.array(x_treinar), np.array(y_treinar) # transformando em array numpy 
x_treinar = np.reshape(x_treinar, (x_treinar.shape[0], x_treinar.shape[1], 1)) # redimensionando os dados

# construindo o modelo de rede neural LSTM

modelo = Sequential()

# camada de entrada e primeira camada oculta com 50 neurônios e 60 dias de previsão 
modelo.add(LSTM(units=50, return_sequences=True, input_shape=(x_treinar.shape[1], 1))) 
# camada de dropout para evitar overfitting 
modelo.add(Dropout(0.2)) # camada de dropout para evitar overfitting 
# 50 neurônios, return_sequences=True para retornar a sequência, input_shape=(x_treinar.shape[1], 1) para pegar a quantidade de dias e 1 para pegar o valor de fechamento
modelo.add(LSTM(units=50, return_sequences=True)) 
# camada de dropout para evitar overfitting 
modelo.add(Dropout(0.2)) 
# 50 neurônios, return_sequences=True para retornar a sequência
modelo.add(LSTM(units=50))
# camada de dropout para evitar overfitting
modelo.add(Dropout(0.2))
# camada de saída com 1 neurônio
modelo.add(Dense(units=1)) 

# compilando o modelo com o otimizador adam que é um otimizador estocástico de gradiente descendente
modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(x_treinar, y_treinar, epochs=25, batch_size=32)

# Testando o modelo com os dados de teste do futuro

# preparndo alguns dados para testar o modelo
teste_inicial = dt.datetime(2021,1,1) # data de inicio
teste_final = dt.datetime.now() # data de hoje

dados_teste = web.DataReader(banco, 'yahoo', teste_inicial, teste_final) # pega os dados do yahoo finance
precos_reais = dados_teste['Close'].values # pega os valores de fechamento

total_dados = pd.concat((dados['Close'], dados_teste['Close']), axis=0) # concatena os dados de treino e teste

modelo_entradas = total_dados[len(total_dados) - len(dados_teste) - previsao_dias:].values # pega os dados de entrada
modelo_entradas = modelo_entradas.reshape(-1,1) # redimensiona os dados
modelo_entradas = normalizando.transform(modelo_entradas) # normaliza os dados

# fazendo as previsões

x_teste = []

for x in range(previsao_dias, len(modelo_entradas)):
    x_teste.append(modelo_entradas[x-previsao_dias:x, 0])

x_teste = np.array(x_teste) # transformando em array numpy
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1)) # redimensionando os dados

previsao_precos = modelo.predict(x_teste) # fazendo as previsões
previsao_precos = normalizando.inverse_transform(previsao_precos) # desnormalizando os dados 

# representação gráfica dos dados

plt.plot(precos_reais, color='red', label=f'Preço real {banco}')
plt.plot(previsao_precos, color='green', label=f'Preço previsto {banco}')
plt.title(f'{banco} Preço real x Preço previsto')
plt.xlabel('Tempo')
plt.ylabel('Preço do Ação')
plt.legend()
plt.show()

# prevendo os próximos dias

# pegando os dados de fechamento e colocando em x_treinar e y_label para prever os próximos dias
dados_reais = [modelo_entradas[len(modelo_entradas) + 1 - previsao_dias:len(modelo_entradas+1), 0]] 
dados_reais = np.array(dados_reais) # transformando em array numpy
dados_reais = np.reshape(dados_reais, (dados_reais.shape[0], dados_reais.shape[1], 1)) # redimensionando os dados

previsao = modelo.predict(dados_reais) # fazendo as previsões
previsao = normalizando.inverse_transform(previsao) # desnormalizando os dados

print(f'Preço previsto para amanhã: {previsao}')








