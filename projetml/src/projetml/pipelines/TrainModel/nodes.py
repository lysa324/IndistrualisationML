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

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):


    shape = input_shape.shape[1:][0]

    inputs = tf.keras.Input(shape=shape)

    x = layers.Reshape(target_shape=(shape, 1))(inputs)

    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.ZeroPadding1D(padding=1)(x) 
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation)(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=shape, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    return model


def train_model(input_shape, x_train, y_train, x_val, y_val):
    
    with mlflow.start_run() as run:
        mlflow.autolog()
        model = create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3)
        model.fit(
            x=x_train,
            y=y_train,
            epochs=2,
            batch_size=32,
            validation_data=(x_val, y_val)
        )

        # Prédire sur les données de validation
        y_val_pred = model.predict(x_val)

        signature = infer_signature(x_val, y_val_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print("Run ID: {}".format(run.info.run_id))

        # Calculer la métrique LOC (Mean Absolute Error)
        locc = mean_squared_error(y_val, y_val_pred)
        print("locc calculé")

        return model





   
