# app.py
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask.ctx import RequestContext
import json
import pandas as pd
import tempfile
import os
from pathlib import Path
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


app = Flask(__name__)
bootstrap_project(Path.cwd())


# Récupération des données
@app.route("/save", methods=["POST"])
def save_user_data():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Aucun fichier CSV n'a été envoyé"})

    if file.filename == "":
        return jsonify({"error": "Le nom du fichier est vide"})

    output_dir = "."

    output_filename = "UserData.csv"
    output_path = os.path.join(output_dir, output_filename)
    file.save(output_path)

    return jsonify({"message": "Fichier enregistré avec succès"})


# Entrainement des données
@app.route("/train", methods=["GET"])
def train_model():
    print("Début d'entrainement du modéle")
    with KedroSession.create("projetml", project_path=".") as session:
        session.run(pipeline_name="TrainModel")

    return jsonify({"message": "Fin entrainement du modéle!"})


# Prédictions
@app.route("/predict", methods=["GET"])
def predict():
    with KedroSession.create("projetml", project_path=".") as session:
        session.run(pipeline_name="GetPredictions")

    prediction = {"prediction": "Résultats de la prédiction en ligne de commande!"}
    return jsonify(prediction)


# Gestion des erreurs 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "attention erreur"}), 404


if __name__ == "__main__":
    app.run(debug=True)
