"""
This is a boilerplate pipeline 'TrainModel'
generated using Kedro 0.18.10
"""
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import mlflow
import mlflow
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from mlflow.models.signature import infer_signature
from kedro.extras.datasets.pandas import CSVDataSet

"""
La fonction create_model créé un modele ML basé sur un réseau de neurone, elle a pour but de renvoyé ce modéle pour des futures utilisations
"""


def create_model(
    input_shape,
    units=128,
    activation="relu",
    l2_value=0.01,
    dropout_rate=None,
    learning_rate=1e-3,
):
    # Modification de la forme de la données d'entré en ajoutant [0]
    shape = input_shape.shape[1:][0]

    # Couche d'entré du modéle
    inputs = tf.keras.Input(shape=shape)

    # 	Redéfinition de la forme des données d'entrée pour le modéle
    x = layers.Reshape(target_shape=(shape, 1))(inputs)

    # Ajout d'une couche de convolution 1D avec 32 filtres et une taille de noyau de 3
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    # Ajout d'une couche de pooling 1D pour réduire la dimensionnalité spatiale
    x = layers.MaxPooling1D(pool_size=2)(x)
    # ZeroPadding1D pour préserver la taille
    x = layers.ZeroPadding1D(padding=1)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation)(x)

    x = layers.ZeroPadding1D(padding=1)(x)
    # Ajout d'une autre couche de pooling 1D
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Conversion des données en un vecteur 1D
    x = layers.Flatten()(x)
    # Ajout d'une couche dense une activation relu
    x = layers.Dense(
        units=shape, activation="relu", kernel_regularizer=regularizers.l2(l2_value)
    )(x)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # Création du modèle en spécifiant les entrées et les sorties
    model = tf.keras.Model(inputs=inputs, outputs=x)

    # Compilation du modèle en spécifiant l'optimiseur, la fonction de perte et les métriques
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",  # ici modification de la fonction loss
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    # retourner le modéle créé
    return model


def train_model(input_shape, x_train, y_train, x_val, y_val):
    # Exécution de mlFlow
    with mlflow.start_run() as run:
        mlflow.autolog()

        # Récupération du modéle créé en appelant la fonction create_model
        model = create_model(
            input_shape,
            units=128,
            activation="relu",
            l2_value=0.01,
            dropout_rate=None,
            learning_rate=1e-3,
        )

        # Entraînement du modèle sur les données d'entraînement
        model.fit(
            x=x_train,
            y=y_train,
            epochs=30,
            batch_size=32,
            validation_data=(x_val, y_val),
        )

        # Prédiction sur les données de validation
        y_val_pred = model.predict(x_val)

        signature = infer_signature(x_val, y_val_pred)

        # Enregistrement du modèle dans le log de MLflow
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Affichage de l'ID de l'exécution actuelle
        print("Run ID: {}".format(run.info.run_id))

        # Retour du modèle entraîné
        return model
