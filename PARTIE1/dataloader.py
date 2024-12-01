import re
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import random
import math
import numpy as np

class Dataloader:
    
    def __init__(self, listpath=None, L=None, k=None, from_files=False, nb_occ=None):
        if from_files:
            print("Chargement des données à partir des fichiers...")
            self.load_from_files(nb_occ)
            print("Données chargées avec succès.")
        else:
            self.listpath = listpath
            self.L = L
            self.k = k
            print("Nettoyage du texte ...")
            self.extract_clean_text()
            print("Création du vocabulaire ...")
            self.create_lexicon(nb_occ)
            print(len(self.minlexicon))
            print("Génération des exemples positifs et négatifs ...")
            self.cpos_cneg()
            print("Création du dataset ...")
            self.dataset_creation()
            self.save_to_json()
            print("Fin de l'initialisation du Dataloader")

    def extract_clean_text(self):
        self.text = ""
        text_parts = []
        word_pattern = re.compile(r'\b\w+\b')  # Compile une seule fois

        for i, path in enumerate(self.listpath):
            print(f"Fichier : {i + 1} / {len(self.listpath)}")
            with open(path) as file:
                for line in tqdm(file):
                    cleaned_line = " ".join(word_pattern.findall(line.strip()))
                    text_parts.append(cleaned_line)
        
        self.text = " ".join(text_parts)

    def create_lexicon(self, nb_occ):
        word_count = Counter(self.text.lower().split()) 
        self.lexicon = dict(word_count)
        self.minlexicon = {key: count for key, count in self.lexicon.items() if count >= nb_occ}
        self.indexer = sorted(self.minlexicon.keys())
        self.text = ' '.join([word for word in self.text.split() if word in self.indexer])
    
    def tirer_mot_aleatoire(self):
        """
        tirage random
        """
        if not hasattr(self, 'minlexicon_keys'):
            self.minlexicon_keys = list(self.minlexicon.keys())
        return random.choice(self.minlexicon_keys)
    

    # def tirer_mot_aleatoire(self):
    #     """
    #     tirage selon la fréquence
    #     """
    #     if not hasattr(self, 'minlexicon_keys'):
    #         self.minlexicon_keys = list(self.minlexicon.keys())
    #     if not hasattr(self, 'minlexicon_weights'):
    #         total_count = sum(self.minlexicon.values())
    #         self.minlexicon_weights = [self.minlexicon[word] / total_count for word in self.minlexicon_keys]
    #     return random.choices(self.minlexicon_keys, weights=self.minlexicon_weights, k=1)[0]
    
    # def tirer_mot_aleatoire(self):
    #     """
    #     tirage avec la fréquence pondéré en fonction de alpha
    #     """
    #     if not hasattr(self, 'minlexicon_keys'):
    #         self.minlexicon_keys = list(self.minlexicon.keys())
    #     alpha = 0.75
    #     if not hasattr(self, 'minlexicon_weights_alpha'):

    #         self.minlexicon_keys = list(self.minlexicon.keys())
    #         total_count = sum(self.minlexicon.values())
    #         self.minlexicon_weights = [self.minlexicon[word]**alpha / total_count**alpha for word in self.minlexicon_keys]

    #     return random.choices(self.minlexicon_keys, weights=self.minlexicon_weights, k=1)[0]

    # def tirer_mot_aleatoire(self, t=10**(-5)):
    #     """
    #     Tirage selon la fréquence avec régularisation Mikolov
    #     P(w_i) = 1 - sqrt(t / f(w_i))
    #     """
    #     if not hasattr(self, 'minlexicon_keys'):
    #         self.minlexicon_keys = np.array(list(self.minlexicon.keys()))
    #         self.minlexicon_values = np.array(list(self.minlexicon.values()), dtype=np.float64)

        #     # Calcul des fréquences normalisées
        #     total_count = self.minlexicon_values.sum()
        #     self.normalized_frequencies = self.minlexicon_values / total_count

        #     # Calcul des probabilités Mikolov
        #     sqrt_t = math.sqrt(t)
        #     self.minlexicon_weights = np.maximum(
        #         0, 1 - sqrt_t / np.sqrt(self.normalized_frequencies)
        #     )

        #     # Normalisation pour obtenir une distribution valide
        #     total_weight = self.minlexicon_weights.sum()
        #     if total_weight > 0:
        #         self.minlexicon_weights /= total_weight
        #     else:
        #         raise ValueError("La somme des poids est nulle. Vérifiez les fréquences et le seuil t.")

        # # Tirage selon les poids ajustés
        # return np.random.choice(self.minlexicon_keys, p=self.minlexicon_weights)

    def cpos_cneg(self):
        self.cpos = defaultdict(list)
        self.cneg = defaultdict(list)
        self.intcpos = defaultdict(list)
        self.intcneg = defaultdict(list)

        mots = self.text.split(" ")
        index_cache = {word: i for i, word in enumerate(self.indexer)}  # Cache des index pour éviter les recherches répétées

        for i in tqdm(range(self.L, len(mots) - self.L)):
            mot = mots[i]
            mot_index = index_cache.get(mot)

            if mot_index is not None:
                # Contexte positif
                ctx = mots[i - self.L : i] + mots[i + 1 : i + self.L + 1]
                intctx = [index_cache.get(e) for e in ctx]
                self.cpos[mot].append(ctx)
                self.intcpos[mot_index].append(intctx)

                # Contexte négatif
                for _ in range(self.k):
                    neg_ctx = [self.tirer_mot_aleatoire() for _ in range(2 * self.L)]
                    int_neg_ctx = [index_cache.get(e) for e in neg_ctx]
                    self.cneg[mot].append(neg_ctx)
                    self.intcneg[mot_index].append(int_neg_ctx)


    def dataset_creation(self):
        """
        [(index du mot, liste positifs, liste negatifs),( . , . , . ),...]
        """
        self.dataset = []
        for e in self.indexer:
            try : 
                # les mots à l'extrémité du texte ne sont pas pris en compte dans la fenêtre donc pas de contexte pour eux
                # ce qui peut créer une erreur avec les clefs
                self.dataset.append([self.indexer.index(e),self.intcpos[self.indexer.index(e)],self.intcneg[self.indexer.index(e)]])
            except : 
                pass

    def save_to_json(self):
        with open("data/cpos.json", "w", encoding="utf-8") as cpos:
            json.dump(self.cpos, cpos, ensure_ascii=False, indent=4)
        with open("data/cneg.json", "w", encoding="utf-8") as cneg:
            json.dump(self.cneg, cneg, ensure_ascii=False, indent=4)
        with open("data/intcpos.json", "w", encoding="utf-8") as intcpos:
            json.dump(self.intcpos, intcpos, ensure_ascii=False, indent=4)
        with open("data/intcneg.json", "w", encoding="utf-8") as intcneg:
            json.dump(self.intcneg, intcneg, ensure_ascii=False, indent=4)
        with open("data/lexicon.json", "w", encoding="utf8") as lexicon:
            json.dump(self.lexicon, lexicon, ensure_ascii=False, indent=4)

    def load_from_files(self, nb_occ):
        print( "Attention la valeur d'occurence minimale doit être la même que lors de l'entraînement ")
        """ Charge les données à partir des fichiers JSON """
        with open("data/cpos.json", "r", encoding="utf-8") as cpos:
            self.cpos = json.load(cpos)
        with open("data/cneg.json", "r", encoding="utf-8") as cneg:
            self.cneg = json.load(cneg)
        # with open("data/intcpos.json", "r", encoding="utf-8") as intcpos:
        #     self.intcpos = json.load(intcpos)
        # with open("data/intcneg.json", "r", encoding="utf-8") as intcneg:
        #     self.intcneg = json.load(intcneg)
        with open("data/intcpos.json", "r", encoding="utf-8") as intcpos:
            self.intcpos = {int(k): v for k, v in json.load(intcpos).items()}  # Conversion des clés en entiers
        with open("data/intcneg.json", "r", encoding="utf-8") as intcneg:
            self.intcneg = {int(k): v for k, v in json.load(intcneg).items()}  # Conversion des clés en entiers
    
        with open("data/lexicon.json", "r", encoding="utf8") as lexicon:
            self.lexicon = json.load(lexicon)
            self.minlexicon = dict()
            for key in self.lexicon.keys():
                if self.lexicon[key] >= nb_occ : 
                    self.minlexicon[key] = self.lexicon[key]
            self.indexer = sorted(list(self.minlexicon.keys()))
        self.dataset_creation()