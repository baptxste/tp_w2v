import gensim
import numpy as np
import random
from collections import defaultdict

def load_model(path):

        with open(path, 'rb') as file:
            return gensim.models.KeyedVectors.load_word2vec_format("43/model.bin", binary=True)
    
def sim_cos(a,b):
    return abs(np.cos(np.dot(a,b)))/(np.linalg.norm(a, ord=2)*np.linalg.norm(b, ord=2))

def eval(path, model):
    sample = []
    cpt = 0
    valid = 0
    dist_moy = 0
    score_dict = defaultdict(int)
    with open('result.txt','w') as result : 
        with open(path,'r') as file : 
            for line in file:
                cpt+=1
                line = line.strip()
                line = line.lower()
                calcul = line.split(' ')
                target = calcul.pop(-1) # récupère la target 
                calcul.pop(-1) # enlève le =
                mot1 = model[calcul[0]]
                mot2 = model[calcul[2]]
                mot3 = model[calcul[4]]
                if calcul[1]=="+":
                    embed= mot1 + mot2
                else : embed= mot1 - mot2 
                if calcul[3]=='+':
                    embed += mot3
                else : embed-=mot3
                pred = model.similar_by_vector(embed, topn=50)
                # print(model.similar_by_vector(embed, topn=3))
                score = -1
                for i, tup in enumerate(pred):
                    if tup[0] == target : 
                        score = i 
                        if score == 0 : 
                            valid +=1
                score_dict[score]+=1
                dist_moy +=sim_cos(embed, model[target])#distance entre le mot le plus probable et la target
                result.write(f"{calcul}, target : {target}, score:{score}\n")


        # pour comparaison distance entre 100 mots aléatoire
        dist_rand = 0
        for i in range(100):
            rand1 = random.choice(model.index_to_key)
            rand2 = random.choice(model.index_to_key)
            dist_rand+=sim_cos(model[rand1], model[rand2])
        moy_rand = dist_rand / 100
            
        
        result.write(f"\n\n{'='*30}\n\n")
        result.write(f" {valid}/{cpt} mots correctement trouvés\n")
        result.write(f"Erreur moyenne : {dist_moy/(100)}, distance moyenne entre les mots : {moy_rand}\n")
        result.write(f"score dict : {dict(sorted(score_dict.items()))}")
            
    
                 

               

if __name__ == '__main__': 
    model = load_model('43/model.bin')
    eval('example.txt', model)

    # pour améliorer le modèle faire une fonction qui prend les 3/4 mots les plus proches et enlève les mots déja présent dans le calcul.