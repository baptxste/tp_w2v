import numpy as np


class Embedding : 

    def __init__(self,  voc : dict, len_voc: int, L : int ):
        """
        L est la taille des embeddings
        """
    
        self.M = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots cibles
        self.C = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots contextes

    def sigmoid(self, x ): 
        return 1 / (1+np.exp(-x))

    def update_embeddings(self, m, cpos, cnegs, lr):
        # Cpos
        score_pos = np.dot(self.M[m], self.C[cpos])
        grad_pos = self.sigmoid(score_pos) - 1
        self.M[m] -= lr * grad_pos * self.C[cpos]
        self.C[cpos] -= lr * grad_pos * self.M[m]

        # Cneg
        for cneg in cnegs:
            score_neg = np.dot(self.M[m], self.C[cneg])
            grad_neg = self.sigmoid(score_neg)
            self.M[m] -= lr * grad_neg * self.C[cneg]
            self.C[cneg] -= lr * grad_neg * self.M[m]



    def generate( self, dataset, lr, it):
        """
        it = nombre d'iteration
        lr = taux d'apprentissage
        """
        for epoch in range(it): 
            for (m, cpos, cnegs) in dataset:  # dataset est constitué des exemples (positifs et négatifs)
                self.update_embeddings(m, cpos, cnegs, lr)

