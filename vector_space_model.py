import nltk 
from nltk import *
import difflib
import math
import os
import re
from functools import reduce
import json

files = []
directory = "/Volumes/buildBox/inforect/dataDump/"
tokenzied = []
posting_lists = {}
tfidf = {}
query = {}
tokenized_query = {}
normalized_query = {}
special_corpus_query = []
weights = {}
stemmer = PorterStemmer()

def fetch_files():
    for file in os.listdir(directory + "/"):
        if file.endswith(".txt"):
            files.append(directory + file)

def token_normalize(func_tokenzied, func_stemmer):
    for buffer in range(0,len(files)):
        read_file = open(files[buffer], "r", encoding="utf-8", errors="ignore").read()
        tokenized_buffer = nltk.word_tokenize(read_file)
        normalized_buffer = [func_stemmer.stem(word.lower())  
                             for word in tokenized_buffer
                        ]
        tokenzied.append(normalized_buffer)

def create_posting_list(func_tokenzied, posting_lists):
    buffer1 = 1
    for file_buffer in tokenzied:
        buffer2 = 0 
        for token_buffer in file_buffer:
            if token_buffer in posting_lists :
                if buffer1 in posting_lists[token_buffer]:
                    posting_lists[token_buffer][buffer1].append(buffer2)
                else:
                    posting_lists[token_buffer][buffer1] = [buffer2]
            else:
                posting_lists[token_buffer] = {}
                posting_lists[token_buffer][buffer1] = [buffer2]
            buffer2 += 1
        buffer1 += 1

def calc_tfidf( func_tfidf, func_tokenzied, func_posting_lists):
    buffer1 = 1
    for file_buffer in tokenzied:
        for token_buffer in file_buffer:
            count = len( func_posting_lists[token_buffer][buffer1])
            if count != 0:
                tf = 1 + math.log10( count )
                idf = math.log10( float( len(files))/len(func_posting_lists[token_buffer] ))
                if buffer1 in func_tfidf:
                    func_tfidf[buffer1][token_buffer] = tf * idf
                else: 
                    func_tfidf[buffer1] = {}
                    func_tfidf[buffer1][token_buffer] = tf * idf
        buffer1 += 1

def find_cosine_similarity(func_weights):
    buffer = 0
    dictionary_keys = list( sorted(func_weights, key=func_weights.__getitem__, reverse=True))
    while( buffer < len(dictionary_keys)):
        cosine_similarity = str(weights[dictionary_keys[buffer]])
        print("File "+ str(buffer+1)+ ": "+str(files[dictionary_keys[buffer]]))
        print("Cosine Similarity: "+ cosine_similarity)
        buffer += 1

def pretty_tfidf(func_tfidf):
    print("Term frequency and Inverse document frequency: ")
    print(json.dumps(func_tfidf, indent=2))


def main():
    fetch_files()
    token_normalize( tokenzied, stemmer )
    calc_tfidf(tfidf, toenzied, posting_lists )
    find_cosine_similarity(weights)
    pretty_tfidf(tfidf)

if __name__== "__main__":
    main()