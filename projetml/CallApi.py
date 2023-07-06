# CallApi.py
# -*- coding: utf-8 -*-
import requests


url = "http://localhost:5000/save"


file_path = "/home/amroun/Documents/M2/IndistrualisationML/ProjetTP1/projetml/data/05_model_input/DataForPredictions.csv"


with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

if response.status_code == 200:
    print("Fichier envoyé avec succès !")
else:
    print("Erreur lors de l'envoi du fichier :", response.json())

