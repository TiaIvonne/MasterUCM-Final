import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pandas as pd
import mysql.connector
import datetime

def preprocesa(data):
    df = pd.read_csv("superstore_data.csv")
    base = df.copy()
    # Calculo anio
    today = datetime.date.today()
    year = today.year
    # La columna Dt_Customer pasa a contar solo con el anio
    base['Dt_Customer'] = pd.to_datetime(base['Dt_Customer'])
    # Anio de registro
    base["Dt_Customer_year"] = base["Dt_Customer"].apply(lambda x: x.year)
    # Se calcula el tiempo de participacion
    base['tiempo_participacion'] = year - base.Dt_Customer_year
    # Columnas con ceros se llenan con la media
    base1 = base.fillna(base.mean(numeric_only=True))
    data =  base1[['Id',"Recency","NumCatalogPurchases", "NumStorePurchases", "MntMeatProducts", "MntWines",
    "NumWebVisitsMonth", "MntGoldProds", "Income", "tiempo_participacion"]]


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


def inserta(df_id_b):
    miconexion = mysql.connector.connect(host='192.168.1.40', user='ivonne', passwd='cosmonauta', db='ucm')
    cursor = miconexion.cursor()

    for i, row in df_id_b.iterrows():
        sql = "UPDATE ucm.ucm SET Prediccion = %s WHERE ID = %s"
        cursor.execute(sql, (row['pred'], row['Id']))
        miconexion.commit()

    miconexion.close()

    return
