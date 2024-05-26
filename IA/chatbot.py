# chatbot.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Charger le modèle GPT-2 pré-entraîné
model_path = "gpt2"  # Utilise le modèle GPT-2 de base
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generer_texte_conseils(texte_entree, est_faux):
    # Concaténer le texte d'entrée avec une indication sur la fiabilité de l'information
    prompt = f"Texte à vérifier : {texte_entree}\n"
    if est_faux:
        prompt += "Ce texte a été classé comme fausse information par notre modèle.\n"
    else:
        prompt += "Ce texte a été classé comme information vraie par notre modèle.\n"

    # Prétraitement du prompt avec le tokenizer GPT-2
    inputs = tokenizer(prompt, return_tensors="pt")

    # Générer du texte avec le modèle GPT-2
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], max_length=100, num_return_sequences=1)

    # Décode le texte généré à partir des logits
    texte_genere = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return texte_genere
