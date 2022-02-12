from nltk import word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import GloVe
import torch
import json 
import pandas as pd 
import pickle 
import os

sw = set(stopwords.words('english'))
words_set = set(['<unk>'])
itos = ['<unk>']
stoi = {'<unk>': 0}
indices = []

def tokenize_data(dir, window_size=150, mode='train'):
    sentences_pos = []
    sentences_neg = []

    for filename in os.listdir(os.path.join(dir, mode, 'pos')):
        with open(os.path.join(dir, mode, 'pos', filename)) as f:
            sentences_pos.append(f.readlines())
    for filename in os.listdir(os.path.join(dir, mode, 'neg')):
        with open(os.path.join(dir, mode, 'neg', filename)) as f:
            sentences_neg.append(f.readlines())

    indices_pos_neg = []
    ancien_bar = 0 
    for c in range(20):
        print('-', end= '')
    print()

    for k, sentences in enumerate([sentences_neg, sentences_pos]):
        for l, sentence in enumerate(sentences):

            # print progress
            at = k*len(sentences_neg) + l 
            bar = int(20*at/(len(sentences_neg)+len(sentences_pos)))

            if bar>ancien_bar:
                for c in range(bar-1):
                    print('=', end= '')
                print('>', end= '')
                for c in range(20-ancien_bar):
                    print('-', end= '')
                print()
            ancien_bar = bar
            
            tokenized_sentence = word_tokenize(sentence[0])
            tokenized_sentence = [w for w in tokenized_sentence if w not in sw]
            i, j = 0, min(window_size, len(tokenized_sentence))
            while True:
                s = tokenized_sentence[i:j]
                idxs =[]
                for word in s:
                    if word not in glove_vocab.stoi or (word not in words_set and mode!='train'):
                        idxs.append(0)
                    elif word in words_set:
                        idxs.append(stoi[word])
                    else:
                        itos.append(word)
                        words_set.add(word)
                        stoi[word] = len(itos)-1
                        indices.append(glove_vocab.stoi[word])
                        idxs.append(stoi[word])
                correction = 0
                while len(idxs) < window_size//2:
                    idxs.append(idxs[correction])
                    correction+=1

                while len(idxs) < window_size:
                    idxs.append(0)
                idxs.append(k)
                indices_pos_neg.append(idxs)

                if len(tokenized_sentence) - j < window_size/4:
                    break
                if j+window_size > len(tokenized_sentence):
                    i = len(tokenized_sentence) - window_size
                    j = len(tokenized_sentence)
                else:
                    i+= window_size
                    j+= window_size
                
    return indices_pos_neg
        
if __name__ == '__main__':
    
    with open('config.json', 'r') as f:
        config = json.load(f)['DATASET']
    print('*----------------------------------------*')
    print('Loading GLOVE embeddings of dimension ', config["embedding_dim"], '...')
    glove_vocab = GloVe(dim=config["embedding_dim"])
    print('Embeddings done!')

    print('*----------------------------------------*')
    print('Tokenizing training data... ')
    indices_train = tokenize_data(config["data_dir"], config["window_size"], 'train') 
    data_train = pd.DataFrame(indices_train)
    print('Done! ')

    print('*----------------------------------------*')
    print('Tokenizing testing data... ')
    indices_test = tokenize_data(config["data_dir"], config["window_size"], 'test')
    print('Done! ')

    data_test = pd.DataFrame(indices_test)

    vocab = {
        "itos": itos,
        "stoi": stoi,
        "vectors": torch.cat([torch.ones(1, config["embedding_dim"]), glove_vocab.vectors[indices]], 0)
    }
    print('*----------------------------------------*')
    print('Saving datasets... ')
    data_train.to_csv(os.path.join(config["path_to_save"], "train.csv"))
    print('Train dataset saved in ', os.path.join(config["path_to_save"], "train.csv") , 'with ', len(data_train), ' examples: ')

    count = data_train[config["window_size"]].value_counts()
    print(' -- positives', count[1])
    print(' -- negatives', count[0])



    data_test.to_csv(os.path.join(config["path_to_save"], "test.csv"))
    print('Test dataset saved in ', os.path.join(config["path_to_save"], "test.csv") ,'with ', len(data_train), ' examples: ')
    count = data_test[config["window_size"]].value_counts()
    print(' -- positives', count[1])
    print(' -- negatives', count[0])

    print('*----------------------------------------*')
    print('Saving vocabulary... ')
    with open(os.path.join(config["path_to_save"], "vocab"), "wb") as f:
        pickle.dump(vocab, f)
    print('Test dataset saved in ', os.path.join(config["path_to_save"], "vocab") ,' with ', len(vocab['itos']), ' words')



        
    

                        
                
                    



    
        
