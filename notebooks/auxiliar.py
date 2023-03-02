from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, LeaveOneOut, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler, OneHotEncoder, LabelEncoder, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline, make_pipeline

def confusion(model, predictors, target):
    """
    Despliega la matriz de confusion

    model: algoritmo
    predictors: variables independientes
    target: variable objetivo
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def reportes(modelo, pred, tar, nombre):
    """
    Reportes de la performance del modelo

    modelo: algoritmo
    predictors: variables independientes
    target: variable objetivo
    nombre: String para nombrar el reporte
    """
    
    pre = modelo.predict(pred) # Crea las predicciones
    
    accuracy = round(accuracy_score(tar, pre), 4)
    recall = round(recall_score(tar, pre), 4)
    roc_auc = round(roc_auc_score(tar, pre), 4)
    precision = round(precision_score(tar, pre),4)
    f1 = round(f1_score(tar, pre, average = "binary"),4)  
    
    
    # Crea un data frame con los resultados
    salida = pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Recall": recall,
            "Roc-Auc": roc_auc,
            "Precision": precision,
            "F1": f1,
        },
        index=[nombre],
    )

    return salida
       

def modelos_ml(modelos, xtrain, ytrain):
    """
    Evalua modelos de machine learning

    modelos: Lista de tuplas con los algoritmos
    xtrain: Datos de entrenamiento
    ytrain: Variable objetivo de los datos de entrenamiento
    """
    nombres = []
    res = []
    scoring = metrics.make_scorer(metrics.f1_score)
    seed = 12345

    for nombre, algoritmo in modelos:
        scaler = QuantileTransformer() 
        pipeline = make_pipeline(scaler, algoritmo) # codigo-alumno
        kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
        results = cross_val_score(pipeline, xtrain, ytrain, cv = kfold, scoring = scoring)
        res.append(results)
        nombres.append(nombre)
        print("F1 score {}: {}".format(nombre, results.mean()))

    plt.figure(figsize=(8,8))
    plt.ylabel(scoring)
    plt.boxplot(res)
    plt.xticks(range(1, len(nombres) + 1), nombres, rotation = 45)
    plt.show()    
    
def modelos_pred(modelos, xtrain, ytrain, xtest, ytest):
    """
    Evalua modelos de machine learning

    modelos: Lista de tuplas con los algoritmos
    xtrain: Datos de entrenamiento
    ytrain: Variable objetivo de los datos de entrenamiento
    xtest: Datos de testeo
    ytest: Variable objetivo de los datos de testeo
    """
    
    for name, model in modelos:
        model.fit(xtrain, ytrain)
        scores = metrics.f1_score(ytest, model.predict(xtest))
        print("{}: {}".format(name, scores))   