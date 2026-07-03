import pandas as pd

# Je suppose que tu charges ton fichier comme ça :
df = pd.read_csv("results_TSMNet.csv")

# 1. On cible la bonne colonne
colonne = df["Balanced_Accuracy"]

# 2. On calcule les statistiques AVEC les parenthèses ()
moyenne = colonne.mean()
ecart_type = colonne.std()
score_min = colonne.min()
score_max = colonne.max()

# 3. On affiche ça proprement en pourcentage
print("\n=== RÉSULTATS TSMNet ===")
print(f"Moyenne Globale : {moyenne * 100:.2f}%")
print(f"Écart-type      : ± {ecart_type * 100:.2f}%")
print(f"Pire score      : {score_min * 100:.2f}%")
print(f"Meilleur score  : {score_max * 100:.2f}%")
print("========================")