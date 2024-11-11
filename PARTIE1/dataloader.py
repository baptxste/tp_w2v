import re
from collections import Counter
import random
import json
from tqdm import tqdm

class Dataloader:
    # def __init__(self, listpath, L, k) :
    #     self.listpath = listpath # ['path1', 'path2',...]
    #     self.L = L
    #     self.k = k
    #     print("Nettoyage du texte ...")
    #     self.extract_clean_text()
    #     print(("Création du vocabulaire ..."))
    #     self.create_lexicon()
    #     print("génération des exemples positifs et négatifs ...")
    #     self.cpos_cneg()
    #     print("création du dataset ...")
    #     self.dataset_creation()
    #     self.save_to_json()
    #     print(" Fin initialisation Dataloader")
    def __init__(self, listpath=None, L=None, k=None, from_files=False, nb_occ=None):
        """
        listpath : liste de chemins vers les fichiers texte
        L : taille de la fenêtre de contexte
        k : nombre de contextes négatifs pour un contexte positif
        from_files : booléen pour indiquer si l'initialisation se fait à partir des fichiers sauvegardés
        """
        if from_files:
            # Initialisation à partir des fichiers sauvegardés
            print("Chargement des données à partir des fichiers...")
            self.load_from_files(nb_occ)
            print("Données chargées avec succès.")
        else:
            # Initialisation complète
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
        """
        -> path du fichier
        -> renvoie le text
        """
        self.text = ""
        for path in self.listpath : 
            file = open(path)
            print(f"Fichier : {self.listpath.index(path)+1} / {len(self.listpath)}")
            for line in tqdm(file.readlines()):
                self.text += line.strip()
                words = re.findall(r'\b\w+\b', self.text)
                self.text = " ".join(words)

    def create_lexicon(self, nb_occ):
            """ 

            """
            text_tmp = self.text.lower()
            words = re.findall(r'\b\w+\b', text_tmp) # inutile ? deja fait dans l'extract ?
            word_count = Counter(words)
            self.lexicon = dict(word_count)
            self.minlexicon = dict()
            for key in self.lexicon.keys():
                if self.lexicon[key] >= nb_occ : 
                    self.minlexicon[key] = self.lexicon[key]
            self.indexer = sorted(list(self.minlexicon.keys()))
            ll = [e for e in words if e in self.indexer ]
            self.text = ' '.join(ll)



    def tirer_mot_aleatoire(self):
        return random.choice([keys for keys in self.minlexicon.keys()])

    def cpos_cneg(self):
        """
        --> text
        --> dict[cpos] {mot1 : [[ctx1], [ctx2]...], mot2 : ...}
        --> dict[cneg] {mot1 : [[ctx_faux1], [ctx_faux2]...], mot2 : ...} avec k fois plus de contexte par mot
        --> dict[intcpos] {index(mot1):[[index(ctx1), ...], [], ...]}
        """
        self.cpos = dict()
        self.cneg = dict()
        self.intcpos = dict()
        self.intcneg = dict()

        mots = self.text.split(" ")
        for i in tqdm(range(self.L, len(mots)-self.L+1)): 
            if mots[i] in self.cpos.keys():
                ctx = mots[i-self.L:i]
                ctx+=(mots[i+1 : i+self.L+1])
                intctx = [self.indexer.index(e) for e in ctx]
                self.cpos[mots[i]].append(ctx)
                self.intcpos[self.indexer.index(mots[i])].append(intctx)
                for j in range(self.k):
                    ctx = [self.tirer_mot_aleatoire() for i in range(2*(self.L))]
                    intctx = [self.indexer.index(e) for e in ctx]
                    self.cneg[mots[i]].append(ctx)
                    self.intcneg[self.indexer.index(mots[i])].append(intctx)

            else : 
                self.cpos[mots[i]] = []
                self.intcpos[self.indexer.index(mots[i])] = []
                ctx = mots[i-self.L:i]
                ctx+=(mots[i+1 : i+self.L+1])
                intctx = [self.indexer.index(e) for e in ctx]
                self.cpos[mots[i]].append(ctx)
                self.intcpos[self.indexer.index(mots[i])].append(intctx)

                self.cneg[mots[i]] = []
                self.intcneg[self.indexer.index(mots[i])] = []
                for j in range(self.k):
                    ctx = [self.tirer_mot_aleatoire() for i in range(2*(self.L))]
                    self.cneg[mots[i]].append(ctx)
                    intctx = [self.indexer.index(e) for e in ctx]
                    self.intcneg[self.indexer.index(mots[i])].append(intctx)

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