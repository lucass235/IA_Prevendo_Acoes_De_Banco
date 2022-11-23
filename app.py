# implementation of a neural network to predict future stock values using the algorithm of Long Short-Term Memory (LSTM)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

banco = 'BB'

inicio = dt.datetime(2012,1,1)
fim = dt.datetime(2022,11,18)

dados = web.DataReader(banco, 'yahoo', inicio, fim)

print(dados)