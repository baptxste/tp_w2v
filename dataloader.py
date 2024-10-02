import re
from collections import Counter
import random
import json

class Dataloader:
    def __init__(self, path, L, k) :
        self.path = path
        self.L = L
        self.k = k
        self.text = self.extract_clean_text()
        self.lexicon = self.create_lexicon()
        self.cpos, self.cneg = self.cpos_cneg()
        self.save_to_json()

        print(" Fin initialisation Dataloader")



    def extract_clean_text(self):
        """
        -> path du fichier
        -> renvoie le text
        """
        file = open(self.path)
        text = ""
        for line in file.readlines():
            text += line.strip()

            # nettoyage du texte : 
            words = re.findall(r'\b\w+\b', text)
            text = " ".join(words)

        return text

    def create_lexicon(self):
            """ 
            -> text
            -> renvoie {mot1 : occurence, mot2 ...}
            """
            words = re.findall(r'\b\w+\b', self.text)
            word_count = Counter(words)
            lexicon = dict(word_count)
            return lexicon

    
    def tirer_mot_aleatoire(self):
        return random.choice([keys for keys in self.lexicon.keys()])
    

    def cpos_cneg(self):
        """
        --> text
        --> dict[cpos] {mot1 : [[ctx1], [ctx2]...], mot2 : ...}
        --> dict[cpos] {mot1 : [[ctx_faux1], [ctx_faux2]...], mot2 : ...} avec k fois plus de contexte par mot
        """
        cpos = dict()
        cneg = dict()
        mots = self.text.split(" ")
        for i in range(self.L, len(mots)-self.L) : 
            if mots[i] in cpos.keys():
                ctx = mots[i-self.L:i]
                ctx+=(mots[i+1 : i+self.L+1])
                cpos[mots[i]].append(ctx)
                for j in range(self.k):
                    ctx = [self.tirer_mot_aleatoire() for i in range(2*(self.L))]
                    cneg[mots[i]].append(ctx)

            else : 
                cpos[mots[i]] = []
                ctx = mots[i-self.L:i]
                ctx+=(mots[i+1 : i+self.L+1])
                cpos[mots[i]].append(ctx)

                cneg[mots[i]] = []
                for j in range(self.k):
                    ctx = [self.tirer_mot_aleatoire() for i in range(2*(self.L))]
                    cneg[mots[i]].append(ctx)

        return cpos, cneg
    

    def save_to_json(self):

        with open("cpos.json", "w", encoding="utf-8") as cpos:
            json.dump(self.cpos, cpos, ensure_ascii=False, indent=4)
        with open("cneg.json", "w", encoding="utf-8") as cneg:
            json.dump(self.cneg, cneg, ensure_ascii=False, indent=4)








    # def cpos_ancien(self):
    #     """
    #     -> text
    #     -> dict[cpos] {mot1 : dict[context]{ mot1 :  occurence, mot2 : occurence }}
    #     """
    #     cpos = dict()
    #     mots = self.text.split(" ")
    #     for i in range(self.L, len(mots)-self.L) : 
    #         if mots[i] in cpos.keys():
    #             for j in range(i-self.L, i+self.L):
    #                 if i==j:
    #                      continue
    #                 if mots[j] in ctx.keys():
    #                         ctx[mots[j]]+=1
    #                 else : ctx[mots[j]]=1
    #         else :
    #             ctx = dict()
    #             for j in range(i-self.L, i+self.L):
    #                 if i==j:
    #                      continue
    #                 ctx[mots[j]]=1
    #             cpos[mots[i]] = ctx
    #     return cpos
    
    # def cneg_ancien(self):
    #     """
    #     --> dictionnaire cpos
    #     --> dict[cneg] {mot1 : dict[context]{ mot1' :  occurence, mot2' : occurence }} avec les mots qui ne sont pas déja dans le context
    #     """

    #     cneg = dict()
    #     for key in self.cpos.keys():
    #         nb_pos = 0 
    #         for val in self.cpos[key].values():
    #          nb_pos += val

 
            
    #          cneg[key]
    #     return 




