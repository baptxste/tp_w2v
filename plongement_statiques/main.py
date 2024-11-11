import gensim
def load_model(path):

        with open(path, 'rb') as file:
            return gensim.models.KeyedVectors.load_word2vec_format("43/model.bin", binary=True)
    


def eval(path, model):
    sample = []
    cpt = 0
    valid = 0
    dist_moy = 0
    with open(path,'r') as file : 
        for line in file:
            cpt +=1
            line = line.strip()
            line = line.lower()
            calcul = line.split(' ')
            target = calcul.pop(-1) # récupère la target 
            calcul.pop(-1) # enlève le =
            mot1 = model[calcul[0]]
            mot2 = model[calcul[2]]
            mot3 = model[calcul[4]]
            if calcul[1]=="+":
                 v= mot1 + mot2
            else : v= mot1 - mot2 
            if calcul[3]=='+':
                 v += mot3
            else : v-=mot3

            pred = model.similar_by_vector(1, topn=3)[0][0]
            if pred == target :
                valid +=1
            else :

               

if __name__ == '__main__': 
    model = load_model('43/model.bin')
    eval('example.txt', model)