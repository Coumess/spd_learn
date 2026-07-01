import os
import sys
import numpy as np

# Ajout du chemin racine pour pouvoir importer tes modules
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# On importe ta fonction d'extraction directement
from preprocessing.data_scripts.preprocess_raw_eeg import _epoch_file

def test_single_subject_extraction():
    # 1. Définis les chemins vers UN SEUL sujet test
    # Remplace par les vrais chemins sur ta machine
    data_dir = "./data/raw_eeg" 
    npz_path = os.path.join(data_dir, "mon_sujet_test.npz")
    yml_path = os.path.join(data_dir, "mon_sujet_test.yml")
    
    # 2. Paramètres de test (comme dans ton config.yaml)
    l_freq = 8.0
    h_freq = 32.0
    target_labels = [1, 2]  # Ex: on ne veut que main gauche (1) et main droite (2)
    
    print(f"🔍 Test d'extraction sur : {os.path.basename(npz_path)}")
    
    try:
        # 3. Lancement de l'extraction
        X, y = _epoch_file(
            npz_path=npz_path,
            yml_path=yml_path,
            l_freq=l_freq,
            h_freq=h_freq,
            target_labels=target_labels
        )
        
        # 4. Vérification des résultats
        print("\n✅ Extraction réussie !")
        print("-" * 30)
        print(f"Forme des données (X) : {X.shape}") 
        print(f"  -> {X.shape[0]} essais (trials)")
        print(f"  -> {X.shape[1]} canaux (électrodes)")
        print(f"  -> {X.shape[2]} points temporels par essai")
        
        print(f"\nForme des labels (y)  : {y.shape}")
        print(f"Classes uniques       : {np.unique(y)} (Devrait commencer à 0)")
        
        # 5. Vérification de la "santé" du signal
        print(f"\nStatistiques globales du signal :")
        print(f"  Moyenne  : {np.mean(X):.4f}")
        print(f"  Variance : {np.var(X):.4f}")
        print(f"  Min/Max  : {np.min(X):.4f} / {np.max(X):.4f}")
        
    except FileNotFoundError:
        print("❌ Erreur : Fichiers introuvables. Vérifie les chemins npz_path et yml_path.")
    except Exception as e:
        print(f"❌ Une erreur est survenue pendant l'extraction : {e}")

if __name__ == "__main__":
    test_single_subject_extraction()