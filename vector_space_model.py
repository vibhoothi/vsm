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
def main():
    fetch_files()
    token_normalize( tokenzied, stemmer )

if __name__== "__main__":
    main()