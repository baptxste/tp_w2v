# import numpy as np
# import matplotlib.pyplot as plt

# class Embedding:

#     def __init__(self, len_voc: int, L: int):
#         """
#         len_voc : taille du vocabulaire
#         L : dimension des embeddings
#         """
#         self.M      = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots cibles
#         self.C      = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots contextes
#         self.losses = []  # Stockage des pertes

#     def sigmoid(self, x):
#         x = np.clip(x, -500, 500) # pour limiter les overflow avec les valeurs importantes
#         return 1 / (1 + np.exp(-x))

#     def update_embeddings(self, m, cpos, cnegs, lr):
#         loss = 0  # Initialiser la perte pour chaque mise à jour
        
#         # Contexte positif
#         for pos in cpos:
#             for w in pos : 
#                 score_pos  = np.dot(self.M[m], self.C[w])
#                 grad_pos   = self.sigmoid(score_pos) - 1
#                 # print(score_pos)
#                 loss      -= np.log(self.sigmoid(score_pos)+1e-9)  # Ajouter la perte positive
#                 self.M[m] -= lr * grad_pos * self.C[w]
#                 self.C[w] -= lr * grad_pos * self.M[m]

#         # Contexte négatif
#         for cneg in cnegs:
#             for w in cneg : 
#                 score_neg = np.dot(self.M[m], self.C[w])
#                 grad_neg = self.sigmoid(score_neg)
#                 loss -= np.log(1 - self.sigmoid(score_neg)+1e-9) # Ajouter la perte négative , le 1e-9 est juse pour éviter le log de 0
#                 self.M[m] -= lr * grad_neg * self.C[w]
#                 self.C[w] -= lr * grad_neg * self.M[m]

#         return loss

#     def generate(self, dataset, lr, it):
#         """
#         dataset : ensemble d'exemples (mot cible, contexte positif, contextes négatifs)
#         lr : taux d'apprentissage
#         it : nombre d'itérations
#         """
#         for epoch in range(it):
#             total_loss = 0
#             for (m, cpos, cnegs) in dataset:  # dataset est constitué des exemples (positifs et négatifs)
#                 total_loss += self.update_embeddings(m, cpos, cnegs, lr)

#             # Enregistrer la perte moyenne pour l'itération actuelle
#             average_loss = total_loss / len(dataset)
#             self.losses.append(average_loss)

#             print(f"Époque {epoch+1}/{it}, Perte : {average_loss}")

#     def plot_loss(self):
#         """
#         Fonction pour tracer la courbe de la perte au cours des itérations
#         """
#         plt.plot(self.losses)
#         plt.title("Loss")
#         plt.xlabel("epoch")
#         plt.ylabel("loss")
#         plt.show()

#     def save_param(self, indexer):
#         np.save('data/matriceM.npy', self.M)  # Binaire efficace pour les grosses matrices
#         np.savetxt('data/matriceM.csv', self.M, delimiter=',')  # fichier lisible mais plus lent

#         with open("data/embedding.txt","w") as l : 
#             l.write(f"nombre de mots, taille de l'embedding : {self.M.shape}\n")
#             for i in range(self.M.shape[0]):
#                 l.write(f"{indexer[i], self.M[i]}\n")


import numpy as np
import matplotlib.pyplot as plt

class Embedding:

    def __init__(self, len_voc: int, L: int):
        """
        len_voc : taille du vocabulaire
        L : dimension des embeddings
        """
        self.M = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots cibles
        self.C = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots contextes
        self.losses = []  # Stockage des pertes

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)  # Pour limiter les overflow avec les valeurs importantes
        return 1 / (1 + np.exp(-x))

    def update_embeddings(self, m, cpos, cnegs, lr):
        loss = 0  # Initialiser la perte pour chaque mise à jour
        
        # Convertir les indices en tableaux NumPy pour opérations vectorisées
        cpos = np.array([w for pos in cpos for w in pos], dtype=int)
        cnegs = np.array([w for cneg in cnegs for w in cneg], dtype=int)

        # Vecteurs pour contexte positif et négatif
        vec_cpos = self.C[cpos]
        vec_cnegs = self.C[cnegs]

        # **Contexte positif**
        scores_pos = np.dot(vec_cpos, self.M[m])
        sigmoid_pos = self.sigmoid(scores_pos)
        grad_pos = sigmoid_pos - 1

        loss -= np.sum(np.log(sigmoid_pos + 1e-9))  # Ajouter la perte positive
        grad_matrix_pos = lr * grad_pos[:, np.newaxis] * vec_cpos
        self.M[m] -= np.clip(lr * grad_pos[:, np.newaxis] * self.C[cpos], -1, 1).sum(axis=0)
        np.add.at(self.C, cpos, np.clip(-lr * grad_pos[:, np.newaxis] * self.M[m], -1, 1))


        # **Contexte négatif**
        scores_neg = np.dot(vec_cnegs, self.M[m])
        sigmoid_neg = self.sigmoid(scores_neg)
        grad_neg = sigmoid_neg

        loss -= np.sum(np.log(1 - sigmoid_neg + 1e-9))  # Ajouter la perte négative
        grad_matrix_neg = lr * grad_neg[:, np.newaxis] * vec_cnegs
        self.M[m] -= np.clip(lr * grad_neg[:, np.newaxis] * self.C[cnegs], -1, 1).sum(axis=0)
        np.add.at(self.C, cnegs, np.clip(-lr * grad_neg[:, np.newaxis] * self.M[m], -1, 1))


        return loss

    def generate(self, dataset, lr, it):
        """
        dataset : ensemble d'exemples (mot cible, contexte positif, contextes négatifs)
        lr : taux d'apprentissage
        it : nombre d'itérations
        """
        for epoch in range(it):
            total_loss = 0
            for (m, cpos, cnegs) in dataset:  # dataset est constitué des exemples (positifs et négatifs)
                total_loss += self.update_embeddings(m, cpos, cnegs, lr)

            # Enregistrer la perte moyenne pour l'itération actuelle
            average_loss = total_loss / len(dataset)
            self.losses.append(average_loss)

            print(f"Époque {epoch+1}/{it}, Perte : {average_loss}")

    def plot_loss(self):
        """
        Fonction pour tracer la courbe de la perte au cours des itérations
        """
        plt.plot(self.losses)
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    def save_param(self, indexer):
        np.save('data/matriceM.npy', self.M)  # Binaire efficace pour les grosses matrices
        np.savetxt('data/matriceM.csv', self.M, delimiter=',')  # fichier lisible mais plus lent

        with open("data/embedding.txt", "w") as l:
            l.write(f"nombre de mots, taille de l'embedding : {self.M.shape}\n")
            for i in range(self.M.shape[0]):
                l.write(f"{indexer[i], self.M[i]}\n")
