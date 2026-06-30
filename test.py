import numpy as np

# Remplace 'ton_fichier.npz' par le chemin exact de ton fichier
chemin_fichier = 'D:/BCI/FII_BCI_Corpus/MI/BNCI2014001/subject_01_session_01.npz'

# L'utilisation de 'with' garantit que le fichier sera correctement fermé après la lecture
with np.load(chemin_fichier) as data:
    
    # data.files retourne une liste des noms des tableaux contenus dans l'archive
    print("Tableaux présents dans l'archive :", data.files)
    
    # Parcourir et afficher le contenu de chaque tableau
    for nom_tableau in data.files:
        print(f"\n--- Contenu du tableau '{nom_tableau}' ---")
        
        # On accède aux données du tableau comme avec un dictionnaire
        tableau = data[nom_tableau]
        
        print("Dimensions (shape) :", tableau.shape)
        print("Données :\n", tableau)