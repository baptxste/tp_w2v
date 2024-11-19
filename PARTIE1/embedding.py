import numpy as np
import matplotlib.pyplot as plt

class Embedding:

    def __init__(self, len_voc: int, L: int):
        """
        len_voc : taille du vocabulaire
        L : dimension des embeddings
        """
        self.M      = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots cibles
        self.C      = np.random.uniform(-0.1, 0.1, (len_voc, L))  # Vecteurs pour les mots contextes
        self.losses = []  # Stockage des pertes

    def sigmoid(self, x):
        x = np.clip(x, -500, 500) # pour limiter les overflow avec les valeurs importantes
        return 1 / (1 + np.exp(-x))

    def update_embeddings(self, m, cpos, cnegs, lr):
        loss = 0  # Initialiser la perte pour chaque mise à jour
        
        # Contexte positif
        for pos in cpos:
            for w in pos : 
                score_pos  = np.dot(self.M[m], self.C[w])
                grad_pos   = self.sigmoid(score_pos) - 1
                # print(score_pos)
                loss      -= np.log(self.sigmoid(score_pos)+1e-9)  # Ajouter la perte positive
                self.M[m] -= lr * grad_pos * self.C[w]
                self.C[w] -= lr * grad_pos * self.M[m]

        # Contexte négatif
        for cneg in cnegs:
            for w in cneg : 
                score_neg = np.dot(self.M[m], self.C[w])
                grad_neg = self.sigmoid(score_neg)
                loss -= np.log(1 - self.sigmoid(score_neg)+1e-9) # Ajouter la perte négative , le 1e-9 est juse pour éviter le log de 0
                self.M[m] -= lr * grad_neg * self.C[w]
                self.C[w] -= lr * grad_neg * self.M[m]

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

        with open("data/embedding.txt","w") as l : 
            l.write(f"nombre de mots, taille de l'embedding : {self.M.shape}\n")
            for i in range(self.M.shape[0]):
                l.write(f"{indexer[i], self.M[i]}\n")
