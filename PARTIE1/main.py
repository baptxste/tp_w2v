import dataloader
import embedding
from eval import test_file
import numpy as np
import json


result_list = []
if __name__ == '__main__': 


    L = 100  # tailles possibles des embeddings
    lr = 0.01  # différents taux d'apprentissage
    it = 7  # différents nombres d'itérations
    w = 8  # tailles de la demi-fenêtre
    k = 10  # nombre de cneg pour 1 cpos
    occ_min = 5  # occurrence minimale

    
    listpath = ["../tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt"]
    # Variables pour suivre l'état des paramètres et éviter de recharger inutilement
    data = dataloader.Dataloader(listpath, w, k, from_files=False, nb_occ=occ_min)

    # Entraînement avec les paramètres actuels
    print(f"\nEntraînement avec L={L}, lr={lr}, it={it}, w={w}, k={k}")
    for exp in range(10) : 
        embed = embedding.Embedding(len(data.indexer), L)
        embed.generate(data.dataset, lr, it)
        # embed.plot_loss()
        losses = embed.losses
        embed.save_param(data.indexer)

        # Évaluation du modèle entraîné
        M = np.load('data/matriceM.npy')
        result_list.append(test_file('../plongement_statiques/Le_comte_de_Monte_Cristo.100.sim', embed.M, data.indexer))

        # Mise à jour des valeurs précédentes de w et k
        previous_w, previous_k = w, k

    print(result_list)