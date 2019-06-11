import gensim
import logging
import os
import csv
import numpy as np
import pandas as pd
import re, os
from sklearn.decomposition import PCA
import argparse

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize



from sklearn.manifold import TSNE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)




def read_input(filename):
    #logging.info("reading file {0}...this may take a while".format(input_file))
    data = pd.read_csv(filename)

    print("Preprocessing texts. Please wait")
    contador = 0
    for row in data['texto']:
        contador = contador + 1
        print("Preprocessing row " + str(contador) + "/" + str(len(data)))
        yield gensim.utils.simple_preprocess(clean_str(row))


	
def clean_str(string):


    stop_words = set(stopwords.words('portuguese')) 
    word_tokens = word_tokenize(string) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(" " + w) 
    
    string = ''.join(filtered_sentence)

    #Remove links
    string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string, flags=re.MULTILINE)
    #Remove mentions e hashtags
    string = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', string, flags=re.MULTILINE)


    return string.strip().lower()
	
def options():
    ap = argparse.ArgumentParser(prog="Word2Vec.py",
            usage="python3 %(prog)s [options]",
            description="Ferramenta para gerar vetores embebidos")
    ap.add_argument("-m", "--modelo", help="Nome do modelo")
    ap.add_argument("-l", "--load", help="Path do modelo a ser carregado")
	ap.add_argument("-l", "--filename", help="Path do arquivo contendo os textos a serem vetorizados")
    ap.add_argument("-e", "--epochs", help="Epochs a serem utilizadas para treinamento")
	
    args = ap.parse_args()
    return args


if __name__ == '__main__':

    args = options() 
    
    if args.epochs is None:
        args.epochs = 10

    if args.load is not None:
       print("Carregando o modelo " + args.load)
       model = gensim.models.KeyedVectors.load_word2vec_format(args.load)
	   # read the tokenized reviews into a list
    else:
        documents = list(read_input(args.filename))
        logging.info("Done reading data file")

        # build vocabulary and train model
        model = gensim.models.Word2Vec(documents, size=50, window=5, min_count=10, workers=12)
        model.train(documents, total_examples=len(documents), epochs=int(args.epochs))

        # save only the word vectors
        model.wv.save(args.modelo) 

    
    while True:
        sentence = input("input> ")

        if sentence == "exit":
            break
    
        try:
            similarities = model.wv.most_similar(positive=sentence)
            print(similarities)
        except:
            print("Error on analysing the similarities for " + sentence)
