import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pandas as pd
import mysql.connector

def preprocesa(datos):
    # Carga y limpieza de datos
    df = pd.read_csv("superstore_data.csv")
    base = df.copy()
    base['Id'] = range(len(base))
    # Solo trabaja con las columnas seleccionadas
    base = df[['Id','MntMeatProducts','Recency','Year_Birth','NumStorePurchases', 'NumCatalogPurchases', 'Income','MntGoldProds', 'MntWines']]
    # Edad
    base["Edad"] = 2023 - pd.to_datetime(base["Year_Birth"], format="%Y").apply(lambda x: x.year)
    # Columna Income se imputa por la media los valores nulo
    base = base.fillna(base.mean(numeric_only=True))
    # Se elimina la columna year birth pues no es relevante
    base = base.drop("Year_Birth", axis = 1)
    data = base[['Id','MntMeatProducts','Recency','Edad','NumStorePurchases', 'NumCatalogPurchases', 'Income','MntGoldProds', 'MntWines']]

    # Insercion en base de datos
    miConexion = mysql.connector.connect(host='192.168.1.40', user='ivonne', passwd='cosmonauta', db='ucm')
    cursor = miConexion.cursor()

    train = data.copy()

    # train = train.fillna(0)

    for i, row in train.iterrows():
        try:
            sql = "INSERT INTO ucm VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)"
            cursor.execute(sql, tuple(row))
            miConexion.commit()
        except:
            print(cursor.statement)
            raise
    miConexion.close()
    return data


def insercion(df_id_b):
    miconexion = mysql.connector.connect(host='192.168.1.40', user='ivonne', passwd='cosmonauta', db='ucm')
    cursor = miconexion.cursor()

    for i, row in df_id_b.iterrows():
        sql = "UPDATE ucm.ucm SET Prediccion = %s WHERE ID = %s"
        cursor.execute(sql, (row['pred'], row['Id']))
        miconexion.commit()

    miconexion.close()

    return
