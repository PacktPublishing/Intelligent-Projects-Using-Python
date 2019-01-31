import keras 
import pickle 
import fire
from elapsedtimer import ElapsedTimer

#path = '/home/santanu/Downloads/Mobile_App/aclImdb/tokenizer.pickle'
#path_out = '/home/santanu/Downloads/Mobile_App/word_ind.txt'
def tokenize(path,path_out):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)



    dict_ = tokenizer.word_index

    keys = list(dict_.keys())[:50000] 
    values = list(dict_.values())[:50000]
    total_words = len(keys)
    f = open(path_out,'w')
    for i in range(total_words):
        line = str(keys[i]) + ',' + str(values[i]) + '\n'
        f.write(line)

    f.close()

if __name__ == '__main__':
    with ElapsedTimer('Tokenize'):
        fire.Fire(tokenize)













