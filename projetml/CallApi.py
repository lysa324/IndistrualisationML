# CallApi.py
# -*- coding: utf-8 -*-
import requests

"""
Ce fichier a pour but d'executer le chemin /save de l'api afin de récupérer les données du user
"""
url = "http://localhost:5000/save"


file_path = "DataForPredictions.csv"
with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

if response.status_code == 200:
    print("Fichier envoyé avec succès !")
else:
    print("Erreur lors de l'envoi du fichier :", response.json())
