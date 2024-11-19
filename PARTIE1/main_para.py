import dataloader
import embedding
from eval import test_file
import numpy as np
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# Définir les combinaisons de paramètres
L_vals = [100]
lr_vals = [0.1]
it_vals = [5, 7, 10, 20]
w_vals = [1, 2, 3, 5, 7]
k_vals = [1, 3, 5, 10]
occ_min_vals = [1, 5, 10]
listpath = ["../tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt"]

# Générer toutes les combinaisons de paramètres
param_combinations = list(product(k_vals, w_vals, it_vals, lr_vals, L_vals, occ_min_vals))

def run_experiment(exp, k, w, it, lr, L, occ_min):
    # Préparer les informations de l'expérience
    info = {'experience': exp, 'k': k, 'w': w, 'it': it, 'lr': lr, 'L': L, 'occ_min': occ_min}
    
    # Charger les données
    data = dataloader.Dataloader(listpath, w, k, from_files=False, nb_occ=occ_min)

    # Entraîner le modèle
    embed = embedding.Embedding(len(data.indexer), L)
    embed.generate(data.dataset, lr, it)
    info['loss'] = embed.losses
    embed.save_param(data.indexer)
    
    # Évaluer le modèle
    M = np.load('data/matriceM.npy')
    result = test_file('../plongement_statiques/Le_comte_de_Monte_Cristo.100.sim', embed.M, data.indexer)
    info['resultat (%)'] = result

    # Sauvegarder les résultats dans un fichier distinct pour chaque expérience
    with open(f'results/result_{exp}.txt', 'w') as file_result:
        json.dump(info, file_result, indent=4)

if __name__ == '__main__':
    # Exécuter les expériences en parallèle avec 28 CPU
    with ProcessPoolExecutor(max_workers=28) as executor:
        # Soumettre chaque expérience avec des paramètres uniques
        futures = [
            executor.submit(run_experiment, exp, k, w, it, lr, L, occ_min)
            for exp, (k, w, it, lr, L, occ_min) in enumerate(param_combinations, 1)
        ]
        # Attendre la fin de toutes les expériences
        for future in futures:
            future.result()  # Assure que les erreurs éventuelles sont capturées
