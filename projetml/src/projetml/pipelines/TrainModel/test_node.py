
from .nodes import train_model
import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def test_train_model():
    print("ok")

    # Données de test fictives
    x_train = np.random.rand(100, 10, 7)
    y_train = np.random.rand(100, 7)
    x_val = np.random.rand(20, 10, 7)
    y_val = np.random.rand(20, 7)

    input_shape = x_train.shape[1:]  # Définition de input_shape en utilisant x_train

    # Appel de la fonction à tester
    model, locc, accuracy = train_model(input_shape, x_train, y_train, x_val, y_val)

    # Vérification des résultats
    assert isinstance(model, tf.keras.Model), "Le modèle retourné doit être une instance de tf.keras.Model"
    assert isinstance(locc, float), "La métrique locc doit être de type float"
    assert isinstance(accuracy, float), "La métrique accuracy doit être de type float"

    print("Les tests passent avec succès !")

   

  

