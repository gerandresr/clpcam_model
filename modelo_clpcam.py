# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:18:52 2023

@author: grios5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xbbg import blp


tickets = {
'CHSWP10 ICCH Curncy': 'swap_10y_cl',
'USGG10YR Index': 'treasury_10y_us',
'ADSWAP10 BGN Curncy': 'swap_10y_au',
'CLP Curncy': 'usdclp',
'CLINNSYO Index': 'inflacion_1y_cl',
'CPI YOY Index': 'Inflacion_1y_us',
'CCHIL1U5 CBIN Curncy': 'cds_cl',
'CHUETOTL Index': 'desempleo_cl',
'CHTBBALM Index': 'balanza_cl',
    }


hoy = date.today()
bbg_data = blp.bdh(list(tickets.keys()), flds=['PX_LAST'],
                   start_date='2018-10-10', end_date=str(hoy),)
df = bbg_data[:]
df.columns = [tickets[i] for i, _ in df.columns]
df = df.fillna(method='ffill').dropna(how='any')

todayvalues = df.tail(1)
df.drop(df.tail(1).index, inplace=True)

y = df.swap_10y_cl
X = df[['inflacion_1y_cl', 'desempleo_cl', 'balanza_cl',
        'cds_cl', 'usdclp', 'treasury_10y_us',
        'swap_10y_au', 'Inflacion_1y_us']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# modelo de regresion lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train) # entreno modelo

# predicciones
predicciones = modelo.predict(X_test)


mae = mean_absolute_error(y_test, predicciones) # error absoluto medio
rsq = r2_score(y_test, predicciones)



inflacion_1y_cl = todayvalues.inflacion_1y_cl[0]
desempleo_cl = todayvalues.desempleo_cl[0]
balanza_cl = todayvalues.balanza_cl[0]
cds_cl = todayvalues.cds_cl[0]
usdclp = todayvalues.usdclp[0]
treasury_10y_us = todayvalues.treasury_10y_us[0]  # treasury_10y_us = 3.70
swap_10y_au = todayvalues.swap_10y_au[0]
Inflacion_1y_us = todayvalues.Inflacion_1y_us[0]

next_period = [[inflacion_1y_cl, desempleo_cl, balanza_cl,
        cds_cl, usdclp, treasury_10y_us,
        swap_10y_au, Inflacion_1y_us]]
prox_dato = modelo.predict(next_period)

plt.scatter(X_test.iloc[:,0], y_test, color='red')
plt.scatter(X_test.iloc[:,0], predicciones, color='blue')
plt.title("Prediccion")
plt.show()
