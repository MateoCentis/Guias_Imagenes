import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

if __name__ == '__main__':
    RANDOM_STATE = 42
    TRAIN_SIZE = 0.7
    NUMERO_ESTIMADORES = 50
    TARGET_NAMES = ['early', 'healthy', 'late']
    CRITERIO = 'gini'
    # Carga de datos     
    
    # segmentadas 
    caracteristicas_early = np.loadtxt("caracteristicas_early.txt")
    caracteristicas_healthy = np.loadtxt("caracteristicas_healthy.txt")
    caracteristicas_late = np.loadtxt("caracteristicas_late.txt")

    # sin segmentar
    # caracteristicas_early = np.loadtxt("caracteristicas_early_sin_segmentar.txt")
    # caracteristicas_healthy = np.loadtxt("caracteristicas_healthy_sin_segmentar.txt")
    # caracteristicas_late = np.loadtxt("caracteristicas_late_sin_segmentar.txt")

    caracteristicas = np.concatenate((caracteristicas_early, caracteristicas_healthy, caracteristicas_late))

    labels_early = np.zeros(caracteristicas_early.shape[0])
    labels_healthy = np.ones(caracteristicas_healthy.shape[0])
    labels_late = np.ones(caracteristicas_late.shape[0])*2
    labels = np.concatenate((labels_early, labels_healthy, labels_late))
    
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, labels, test_size=1-TRAIN_SIZE, random_state=RANDOM_STATE)

    modelo = RandomForestClassifier(n_estimators=NUMERO_ESTIMADORES, random_state=RANDOM_STATE, criterion=CRITERIO)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("---------------------------")
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    print("Report")
    print(report)
    print("---------------------------")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Accuracy balanceada: {balanced_acc}')

