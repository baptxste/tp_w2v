import re
from collections import Counter


class Dataloader:
    def __init__(self, path, L, k) :
        self.path = path
        self.L = L
        self.k = k
        self.text = self.extract_clean_text()
        self.lexicon = self.create_lexicon()
        self.cpos = self.cpos()

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


    def cpos(self):
        """
        -> text
        -> dict[cpos] {mot1 : dict[contect]{ mot1 :  occurence, mot2 : occurence }}
        """
        cpos = dict()
        mots = self.text.split(" ")
        for i in range(self.L, len(mots)-self.L) : 
            if mots[i] in cpos.keys():
                ctx = mots[i]
                for j in range(i-self.L, i+self.L):
                    if mots[j] in ctx.keys():
                            ctx[mots[j]]+=1
                    else : ctx[mots[j]]=1
            else :
                ctx = dict()
                for j in range(i-self.L, i+self.L):
                    ctx[mots[j]]=1
                cpos[mots[j]] = ctx
        return cpos
    
    def cneg(self):
        return 




