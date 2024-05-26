from flask import Flask, render_template, request
from FakeNewsDetection.fakenews_detection import FakeNewsPredictor
from FakeNewsDetection.data_storage_in_csv import sauvegarder_resultat_csv
from tensorflow.keras.models import load_model
import csv 
import uuid
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = load_model("/Users/yassineseidou/Desktop/PROJETS TECHNIQUE/PROJET D'IA - DEEP FAKES:FAKE NEWS/FINALS_DOCS/IA/FakeNewsDetection/my_model2.keras")  # Assurez-vous de spécifier le bon chemin

# Chemin du fichier CSV pour stocker l'historique
fichier_historique = "/Users/yassineseidou/Desktop/PROJETS TECHNIQUE/PROJET D'IA - DEEP FAKES:FAKE NEWS/FINALS_DOCS/IA/FakeNewsDetection/historique_verifications.csv"

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['information']
        result = FakeNewsPredictor(model, sentence)
        # Récupérer l'ID utilisateur généré automatiquement
        id_utilisateur = str(uuid.uuid4())

        # Enregistrer les résultats dans le CSV avec l'ID utilisateur
        sauvegarder_resultat_csv(sentence, result)
 
        return render_template('index3.html', prediction=result)
  
@app.route('/historique')
def historique():
    # Charger le fichier CSV en tant que DataFrame avec pandas
    historique_df = pd.read_csv(fichier_historique)

    # Récupérer les 5 dernières lignes de l'historique
    historique_df_tail = historique_df.tail(5)

    # Convertir le DataFrame en HTML pour l'affichage dans le template
    historique_html = historique_df_tail.to_html(index=False)

    # Passer l'historique HTML au template
    return render_template('index3.html', historique=historique_html)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
