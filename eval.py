import numpy as np
import os 



def sim_cos(a,b):
    return np.cos(np.dot(a,b))/(np.linalg.norm(a, ord=2)*np.linalg.norm(b, ord=2))
    

def test_file( path_file, M, indexer):
    """
    file de la forme : mots1   mots2   mots3
    M est la matrice des embedding
    indexer
    """
    nb_line = 0
    ok = 0
    result = os.path.join(os.path.dirname(path_file),'result.txt')
    with open(result,'w') as res : 
        with open(path_file,'r') as test : 
            for line in test : 
                nb_line +=1
                line = line.strip()
                mot1, mot2, mot3 = line.split(' ')

                sim1 = sim_cos(M[indexer.index(mot1)],M[indexer.index(mot2)])
                sim2 = sim_cos(M[indexer.index(mot1)],M[indexer.index(mot3)])

                res.write(f"{mot1}  {mot2}  {mot3}  {sim1}  {sim2}\n")

                if sim1>sim2 : 
                    ok+=1
    print(f"Résultat du test : {ok*100/nb_line}%")
