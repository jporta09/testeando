import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
import re
import unicodedata

import nltk
import sklearn
from nltk.corpus import stopwords  
from nltk import word_tokenize  
from nltk.data import load  
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from string import punctuation 

#stopword list to use
spanish_stopwords = stopwords.words('spanish')

#spanish stemmer
stemmer = SnowballStemmer('spanish')

#punctuation to remove
non_words = list(punctuation)  
#we add spanish punctuation
non_words.extend(['¿', '¡', ':', ';'])  
non_words.extend(map(str,range(10)))

stemmer = SnowballStemmer('spanish')  
def stem_tokens(tokens, stemmer):  
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(WordNetLemmatizer().lemmatize(item, pos='v')))
    return stemmed

def tokenize(text):  
    # remove punctuation
    tokens2 = []
    text = ''.join([c for c in text if c not in non_words])
    # tokenize
    tokens =  word_tokenize(text)
    for token in tokens:
        if token not in spanish_stopwords:
            tokens2.append(token)


    return tokens2

def remove_accents(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

datt = pd.read_csv("final2.txt",encoding="utf-8", delimiter="\t", low_memory=False, keep_default_na=False)
datt["todo"] = datt["todo"].str.replace("\r", " ").str.replace("\n", " ")

from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

ldapos = models.LdaModel.load("ldapos")
ldaneg = models.LdaModel.load("ldaneg")
ldanull = models.LdaModel.load("ldanull")
dictionarypos = corpora.Dictionary.load("posdict.dict")
dictionaryneg = corpora.Dictionary.load("negdict.dict")
dictionarynull = corpora.Dictionary.load("nulldict.dict")
def free(x):
    todo = ''.join([c for c in x["todo"] if c not in non_words])
    todo = remove_accents(todo)
    #print(x)
    fafa = get_ipython().getoutput("echo $todo | docker run -i --rm freeling analyze -f es.cfg")
    if fafa == []:
        rank = [0]*8
        asda = rank
        return asda
    #print(fafa)
    if "" in fafa:
        fafa.remove("")
    fafa = pd.DataFrame([a.split(" ") for a in fafa])
    aa = " ".join(map(str, fafa[1].tolist()))
    if x["tipo"] == "+":
        #print("+")
        bow_vector = dictionarypos.doc2bow(tokenize(aa))
        rank = [0]*8
        lstfill = [a[0] for a in ldapos[bow_vector]]
        rank2 = [a[1] for a in ldapos[bow_vector]]
        if len(rank2) < 8:
            #print("here")
            u = 0
            for i in range(8):
                if i in lstfill:

                    rank[i] = rank2[u]
                    u+=1
        else:
            rank = [a[1] for a in ldapos[bow_vector]]
        #print(rank, lstfill, rank2)
        asda = [tokenize(aa)] + rank
    elif x["tipo"] == "-":
        #print("-")
        bow_vector = dictionaryneg.doc2bow(tokenize(aa))
        rank = [0]*8
        lstfill = [a[0] for a in ldaneg[bow_vector]]
        rank2 = [a[1] for a in ldaneg[bow_vector]]
        if len(rank2) < 8:
            #print("here")
            u = 0
            for i in range(8):
                if i in lstfill:

                    rank[i] = rank2[u]
                    u+=1
        else:
            rank = [a[1] for a in ldaneg[bow_vector]]
        #print(rank, lstfill, rank2)
        asda = [tokenize(aa)] + rank
    elif x["tipo"] == "null":
        #print("null")
        bow_vector = dictionarynull.doc2bow(tokenize(aa))
        rank = [0]*8
        lstfill = [a[0] for a in ldanull[bow_vector]]
        rank2 = [a[1] for a in ldanull[bow_vector]]
        try:
            if len(rank2) < 8:
                #print("here")
                u = 0
#                if x.name == 53806:
#                    import ipdb
#                    ipdb.set_trace()
                for i in range(8):
                    if i in lstfill:
                        print("u",u,"i",i,"rank",rank,"lenlistfill",len(lstfill), "index", x.name)
                        rank[i] = rank2[u]
                        u+=1
            else:
                rank = [a[1] for a in ldanull[bow_vector]]
            asda = [tokenize(aa)] + rank
        except IndexError:
            print("index error")    
            rank = [0]*8
            asda = [tokenize(aa)] + rank
        #print(rank, lstfill, rank2)
        
    else:
        rank = ["err"]*8
        asda = [tokenize(aa)] + rank
    print(rank)
    return asda

#datt2 = datt[["todo","tipo"]].loc[53806:53808]
#display(datt2)
#free1 = datt2.apply(lambda x: free(x), axis=1)
free1 = datt[["todo","tipo"]].parallel_apply(lambda x: free(x), axis=1)

import pickle
file = open('dump1', 'wb')

pickle.dump(free1, file)

file.close()
