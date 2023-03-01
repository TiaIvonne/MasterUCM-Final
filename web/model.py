import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.linear_model import LogisticRegression
import pickle
pd.options.display.float_format = '{:.2f}'.format
#Carga y limpieza de datos

df = pd.read_csv("superstore_data.csv")
base = df.copy()
base['Id'] = range(len(base))
# Feature selection
base = df[['Id','MntMeatProducts','Recency','Year_Birth','NumStorePurchases', 'NumCatalogPurchases', 'Income','MntGoldProds', 'MntWines','Response']]
# Edad
base["Edad"] = 2023 - pd.to_datetime(base["Year_Birth"], format="%Y").apply(lambda x: x.year)
# Columna Income se imputa por la media los valores nulos
base = base.fillna(base.mean(numeric_only=True))
# Se elimina la columna year birth pues no es relevante
base = base.drop("Year_Birth", axis = 1)
data = base[['Id','MntMeatProducts','Recency','Edad','NumStorePurchases', 'NumCatalogPurchases', 'Income','MntGoldProds', 'MntWines','Response']]



#Training Data and Predictor Variable

X = data.drop('Response', axis = 1)
y = data.Response

classification = LogisticRegression(solver='lbfgs', max_iter=1000)


# Entrenamiento
classification.fit(X, y)



# Guarda el modelo
pickle.dump(classification, open('model.pkl','wb'))
#
# '''
# #Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2.6, 8, 10.1]]))
# '''