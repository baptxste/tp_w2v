import dataloader
import embedding
from eval import test_file

import numpy as np

if __name__ == '__main__': 

    L = 100 # taille des embeddings
    lr = 0.01 # learning rate
    it = 5 # nombre d'iteration
    w = 2 # taille de la  demi fenêtre
    k = 10 # nombre de cneg pour 1 cpos
    occ_min = 5 # nombre d'occurence minimal

    # listpath = ["tlnl_tp1_data/alexandre_dumas/La_Reine_Margot.txt", 
    #             "tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt",
    #             "tlnl_tp1_data/alexandre_dumas/Le_Vicomte_de_Bragelonne.txt",
    #             "tlnl_tp1_data/alexandre_dumas/Les_Trois_Mousquetaires.txt",
    #              "tlnl_tp1_data/alexandre_dumas/Vingt_ans_apres.txt"]

    # listpath = ['Test/data/text_test2.txt']

    listpath = ["tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt"]

    #
    ##
    ### CHARGER ET ENTRAINER DEPUIS LES DONNEES BRUT
    ##
    #

    data = dataloader.Dataloader(listpath, w, k, from_files=False, nb_occ = occ_min)
    embed = embedding.Embedding(len(data.indexer),L)
    embed.generate(data.dataset, lr, it)
    embed.plot_loss()
    embed.save_param(data.indexer)

    #
    ##
    ### CHARGER  ET ENTRAINER DEPUIS DES DONNEES PRETRAITEES
    ##
    #
        
    # data = dataloader.Dataloader(from_files=True, nb_occ = occ_min)
    # embed = embedding.Embedding(len(data.indexer),L)
    # embed.generate(data.dataset, lr, it)
    # embed.plot_loss()
    # embed.save_param(data.indexer)

    #
    ##
    ### CHARGER UN MODELE ENTRAINE ET EVALUE 
    ##
    #
    M = np.load('data/matriceM.npy')
    test_file('plongement_statiques/Le_comte_de_Monte_Cristo.100.sim',embed.M, data.indexer)

    




