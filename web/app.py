import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, redirect, make_response, session
import joblib
import os
import pandas as pd
import io
from io import StringIO
import csv
from werkzeug.utils import secure_filename
from catboost import CatBoostClassifier
import datetime as dt
from basicas import preprocesa, inserta

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model_form = pickle.load(open('model_9.pkl', 'rb'))

uploads_dir = './instance/uploads'
os.makedirs(uploads_dir, exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')


# Ruta que lleva a formulario manual
@app.route('/manual')
def manual():
    return render_template('formulario.html')


# Predicciones x formulario
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Realiza predicciones en base a los datos recolectados en el formulario
    usa solo las variables mas relevantes
    """
    int_features = [float(x) for x in request.form.values()]  # Toma los valores del formulario
    final_features = [np.array(int_features)]
    prediction = model_form.predict(final_features)
    probability = np.max(model_form.predict_proba(final_features)) * 100  # Calcula las probabilidades
    output = prediction[0], 2
    if (output == 1):
        out = "Yes"
    else:
        out = "No"
    return render_template('formulario.html',
                           prediction_text='Chance de tomar la campaña de temporada: {} con un % de {}'.format(out,
                                                                                                               probability))


# Carga csv y conecta con BBDD
@app.route('/carga')
def carga():
    return render_template('carga.html')

@app.route('/predictcsv',methods=['POST'])
def predictcsv():
    modelo = open('model.pkl', 'rb')
    clf = joblib.load(modelo)

    if request.method == 'POST':
        message = request.files['fileupload']
        message.save(os.path.join(uploads_dir, secure_filename(message.filename)))
        data = pd.read_csv(message.filename)
        df_id = data["Id"]

        datos = preprocesa(data)

        my_prediction = np.expm1(clf.predict(datos))

        df_id_b = pd.DataFrame(df_id, columns=['Id'])
        df_mypred = pd.DataFrame(my_prediction, columns=['pred'])
        df_id_bi = pd.concat([df_id_b, df_mypred], axis=1)
        df_id_b = df_id_bi.to_numpy()

        inserta(df_id_bi)

    return render_template('resultados.html', prediction=my_prediction, tam=df_id_b)


if __name__ == '__main__':
    app.run(host='192.168.1.30', debug=True, port=5111)
