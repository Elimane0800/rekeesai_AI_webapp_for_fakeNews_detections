from flask import Flask, request, jsonify, render_template
from FakeNewsDetection.fakenews_detection import FakeNewsPredictor

# Initialisation de l'application Flask
app = Flask(__name__, static_folder="Static")

# Chargement du modèle de détection de fausses nouvelles
from tensorflow.keras.models import load_model
model = load_model("/Users/yassineseidou/Desktop/PROJETS TECHNIQUE/PROJET D'IA - DEEP FAKES:FAKE NEWS/FINALS_DOCS/IA/FakeNewsDetection/my_model.keras")  # Assurez-vous que le nom du modèle est correct

# Définir une route pour l'interface utilisateur
@app.route('/fakenewsdetection', methods=['POST'])
def fn_detector():
    if request.method == 'POST':
        sentence = request.form['sentence']  # Récupérer le texte soumis par l'utilisateur
        result = FakeNewsPredictor(model, sentence)  # Utiliser la fonction de détection
        
        return jsonify({'result': result})  # Retourner le résultat au format JSON
    return '''
        <form method="POST">
            <textarea name="user_input" cols="40" rows="5"></textarea>
            <br/>
            <input type="submit" value="Check">
        </form>
    '''

# Point d'entrée de l'application
@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application page over the Flask channel'''
    return render_template('index2.html')
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)

