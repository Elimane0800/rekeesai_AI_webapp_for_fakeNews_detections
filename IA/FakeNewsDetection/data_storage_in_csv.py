import csv
from datetime import datetime
import uuid


VerificationStorage = "historique_verifications.csv"  # Nom du fichier CSV constant

def sauvegarder_resultat_csv(texte_verifie, resultat):
    now = datetime.now()
    date_heure = now.strftime("%Y-%m-%d %H:%M:%S")

    # Générer un ID utilisateur unique à l'aide du module uuid
    id_utilisateur = str(uuid.uuid4())

    # Écrire les données dans le fichier CSV en mode ajout ('a')
    with open(VerificationStorage, mode='a', newline='') as fichier:
        writer = csv.writer(fichier)
        # Écriture d'en-têtes si le fichier est vide
        if fichier.tell() == 0:
            writer.writerow(["ID Utilisateur", "Date Heure", "Texte Vérifié", "Résultat"])
        writer.writerow([id_utilisateur, date_heure, texte_verifie, resultat])


