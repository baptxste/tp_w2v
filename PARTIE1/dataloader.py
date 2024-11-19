import re
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import random

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
    
    # def tirer_mot_aleatoire(self):
    #     if not hasattr(self, 'minlexicon_keys'):
    #         self.minlexicon_keys = list(self.minlexicon.keys())
    #     return random.choice(self.minlexicon_keys)
    
    def tirer_mot_aleatoire(self): # tiens compte de la fréquence des mots
        if not hasattr(self, 'lexiconproba'):
            self.lexiconproba = []
            for key in self.minlexicon.keys() :
                for i in range( self.minlexicon[key]):
                    self.lexiconproba.append(key)
        return random.choice(self.lexiconproba)


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