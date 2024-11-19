import dataloader
import embedding
from eval import test_file
import numpy as np
import json

if __name__ == '__main__': 
    # Fixer les paramètres initiaux
    # L_vals = [50, 100, 150]  # tailles possibles des embeddings
    # lr_vals = [0.01, 0.05, 0.1]  # différents taux d'apprentissage
    # it_vals = [5, 7, 10, 20]  # différents nombres d'itérations
    # w_vals = [1, 2, 3, 5, 7]  # tailles de la demi-fenêtre
    # k_vals = [1, 3, 5, 10]  # nombre de cneg pour 1 cpos
    # occ_min_vals = [1, 5, 10]  # occurrence minimale

    L_vals = [100]  # tailles possibles des embeddings
    lr_vals = [0.1]  # différents taux d'apprentissage
    it_vals = [7]  # différents nombres d'itérations
    w_vals = [2]  # tailles de la demi-fenêtre
    k_vals = [4]  # nombre de cneg pour 1 cpos
    occ_min_vals = [5]  # occurrence minimale

    
    listpath = ["../tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt"]
    


    # Variables pour suivre l'état des paramètres et éviter de recharger inutilement
    previous_w, previous_k = None, None
    exp =0
    file_result = open('result.txt','a')

    for k in k_vals:
        for w in w_vals:
            for it in it_vals:
                for lr in lr_vals:
                    for L in L_vals:
                        for occ_min in occ_min_vals:
                            exp += 1
                            info = {'experience':exp,
                                    'k':k,
                                    'w':w,
                                    'it':it,
                                    'lr':lr,
                                    'L':L,
                                    'occ_min':occ_min
                                    }
                            data = dataloader.Dataloader(listpath, w, k, from_files=False, nb_occ=occ_min)

                            # Entraînement avec les paramètres actuels
                            print(f"\nEntraînement avec L={L}, lr={lr}, it={it}, w={w}, k={k}")
                            embed = embedding.Embedding(len(data.indexer), L)
                            embed.generate(data.dataset, lr, it)
                            # embed.plot_loss()
                            losses = embed.losses
                            info['loss']=losses
                            embed.save_param(data.indexer)

                            # Évaluation du modèle entraîné
                            M = np.load('data/matriceM.npy')
                            result = test_file('../plongement_statiques/Le_comte_de_Monte_Cristo.100.sim', embed.M, data.indexer)
                            info['resultat (%)']=result
                            # Mise à jour des valeurs précédentes de w et k
                            previous_w, previous_k = w, k

                            json.dump(info, file_result, indent=4 )


    file_result.close()