import dataloader
import embedding
from eval import test_file
import numpy as np
import json
from itertools import product

def run_experiment(L, lr, it, w, k, occ_min, listpath):
    # Charger les données uniquement si w ou k change
    key = (w, k)
    if key not in cached_data:
        cached_data[key] = dataloader.Dataloader(listpath, w, k, from_files=False, nb_occ=occ_min)
    data = cached_data[key]
    
    results = []
    for exp in range(10):
        embed = embedding.Embedding(len(data.indexer), L)
        embed.generate(data.dataset, lr, it)
        result = test_file('../plongement_statiques/Le_comte_de_Monte_Cristo.100.sim', embed.M, data.indexer)
        results.append(result)
    return results

if __name__ == '__main__':
    # Hyperparamètres à tester
    L_values = [100]  # tailles possibles des embeddings
    lr_values = [0.001]  # différents taux d'apprentissage
    it_values = [5, 10, 20]  # différents nombres d'itérations
    w_values = [2, 5]  # tailles de la demi-fenêtre
    k_values = [10]  # nombre de cneg pour 1 cpos
    occ_min = 2  # occurrence minimale
    listpath = ["../tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt"]

    # Dictionnaire pour éviter les rechargements inutiles
    cached_data = {}

    # Résultats finaux
    all_results = []

    # Générer toutes les combinaisons possibles de paramètres
    parameter_combinations = product(L_values, lr_values, it_values, w_values, k_values)

    for i, params in enumerate(parameter_combinations):
        L, lr, it, w, k = params
        print(f"\nTest numéro : {i+1}/{len(list(parameter_combinations))} avec L={L}, lr={lr}, it={it}, w={w}, k={k}")
        
        # Exécuter les expériences
        results = run_experiment(L, lr, it, w, k, occ_min, listpath)

        # Calcul des statistiques
        mean_result = np.mean(results)
        std_result = np.std(results)
        result_entry = {
            "parameters": {"L": L, "lr": lr, "it": it, "w": w, "k": k},
            "raw_results": results,
            "mean": mean_result,
            "std": std_result,
        }
        all_results.append(result_entry)
    
    # Sauvegarder les résultats dans un fichier JSON
    with open("experiment_results.json", "w") as outfile:
        json.dump(all_results, outfile, indent=4)

    print("Résultats sauvegardés dans 'experiment_results.json'.")
