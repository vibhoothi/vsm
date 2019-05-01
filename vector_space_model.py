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

def main():
    fetch_files()

if __name__== "__main__":
    main()