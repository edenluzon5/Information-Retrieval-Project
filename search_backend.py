import itertools
from itertools import islice, count, groupby, chain
import pandas as pd
import numpy as np
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
from collections import defaultdict, Counter
# from tqdm import tqdm
import operator
import json
from io import StringIO
import math
import google
import hashlib
from collections import Counter, OrderedDict, defaultdict
import heapq
from inverted_index_gcp import *
nltk.download('stopwords')

norma = pd.read_pickle(r'gs://irproject1/docsnorm.pkl')
page_view = pd.read_pickle(r'gs://irproject1/pageviews-202108-user.pkl')
page_rank = pd.read_pickle(r'gs://irproject1/pageRank.pkl')
docs_titles = pd.read_pickle(r'gs://irproject1/titlesdocs.pkl')
''''QUERY TOKANIZATION'''
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

def queryRepresent(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens

def queryRepresent1(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    corpus_stopwords1 = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became","best"]
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    all_stopwords1 = english_stopwords.union(corpus_stopwords1)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords1]
    return list_of_tokens

def best_search(query,indexTitle,body_index,titlePath,bodyPath):
    query=queryRepresent1(query)
    bm25=BM25_from_index(body_index,bodyPath)
    listi1=dict(binarySearch1(query,indexTitle,titlePath))
    listi=bm25.bm25_search(query,bodyPath,100)
    for tup in listi:
        if tup[0] in listi1:
            if str(tup[0]) in page_rank:
                listi1[tup[0]]+=tup[1]+ math.log(float(page_rank[str(tup[0])]))
            else:
                listi1[tup[0]] += tup[1]
        else:
            if str(tup[0]) in page_rank:
                listi1[tup[0]]=tup[1]+ math.log(float(page_rank[str(tup[0])]))
            else:
                listi1[tup[0]] = tup[1]
    for tup in listi1:
        if tup not in listi and len(query)<3:
            if str(tup) in page_rank:
                listi1[tup]=listi1[tup]*6 + math.log(float(page_rank[str(tup)]))
            else:
                listi1[tup] = listi1[tup] * 6
    listi1= sorted([(doc_id,score) for doc_id, score in listi1.items()],key=lambda x:x[1],reverse=True)[:10]
    return [(tup[0],docs_titles[tup[0]]) for tup in listi1 if tup[0] in docs_titles]

''''BINARY SEARCH FOR THE GENERAL SEARCH METHOD'''
def binarySearch1(query,index,path):
  dict_results={}
  query=np.unique(query)
  for term in query:
    if term in index.df:
      postings=InvertedIndex.read_posting_list(index,term,path)
      for doc_id, tf in postings:
        if doc_id==0:
            continue
        dict_results[doc_id]=dict_results.get(doc_id,0)+1
  # sort result by the number of tokens appearance
  return sorted(dict_results.items(), key=lambda x: x[1], reverse=True)

''''BINARY SEARCH METHOD'''
def binarySearch(query,index,path):
  dict_results={}
  query=np.unique(query)
  for term in query:
    if term in index.df:
      postings=InvertedIndex.read_posting_list(index,term,path)
      for doc_id, tf in postings:
            if doc_id == 0:
                continue
            dict_results[doc_id]=dict_results.get(doc_id,0)+1
  # sort result by the number of tokens appearance
  res= sorted(dict_results.items(), key=lambda x: x[1], reverse=True)
  res1=[]
  for tup in res:
    if tup[0] in docs_titles.keys():
      res1.append((tup[0],docs_titles[tup[0]]))
    else:
      res1.append(tup[0])
  return res1

''''COSINE SIMILARITY METHOD'''
def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_query_size = len(query_to_search)
    Q = np.zeros((total_query_size))
    counter = Counter(query_to_search)
    ind=0
    sumQ=0.0
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
            sumQ+=(tf*idf)**2
            Q[ind] = tf * idf
            ind+=1
    return Q,sumQ

def get_candidate_documents_and_scores(query_to_search, index, path):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    path: path to posting list.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
      if term in index.term_total:
        posting_list=InvertedIndex.read_posting_list(index,term,path)
        normlized_tfidf=[]
        for tpl in posting_list:
            if tpl[0] in index.DL:
              tf=tpl[1]/index.DL[tpl[0]]
              idf=math.log(len(index.DL)/index.df[term],10)
              tfidf=tf*idf
              normlized_tfidf.append((tpl[0],tfidf))
        for doc_id, tfidf in normlized_tfidf:
            candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               

    return candidates

def cosine_similarity(Doc_vector, Q_vector,doc_id,sumQ):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    Q_array=np.array(Q_vector)
    doc_array=np.array(Doc_vector)
    if doc_id in norma:
        denominator = norma[doc_id]*math.sqrt(sumQ)
    else:
        denominator = 0.0
    if denominator==0.0:
      cosine=0.0
    else:
      cosine = np.dot(Q_array, doc_array) / denominator
    return cosine

def top_n_cosine_sim(query,index,bodyPath,N=100):
    invertedIndex_term=get_candidate_documents_and_scores(query,index,bodyPath)
    #init all sim (q,d)=0
    dict_docs={}
    for doc, term in invertedIndex_term.keys():
      dict_docs[doc]=[0]*len(query)
    #init query vector tfidf
    query_vector,sumQ=generate_query_tfidf_vector(query,index)
    #init term query index
    unique_terms={}
    i=0
    #dict of term idx in query
    for term in query:
      unique_terms[term]=i
      i+=1
    #init tfidf for doc_id,term
    for doc_id, term in invertedIndex_term.keys():
      indexTerm=unique_terms[term]
      tfidf=invertedIndex_term[(doc_id,term)]
      dict_docs[doc_id][indexTerm]=tfidf
    #create heap for tfidf scores
    top_n_docs=[]
    for doc, doc_tfidf_vector in dict_docs.items():
      cos_sim=cosine_similarity(doc_tfidf_vector,query_vector,doc,sumQ)
      if len(top_n_docs)==N:
        heapq.heapify(top_n_docs)
        mini=heapq.heappop(top_n_docs)
        if cos_sim>mini[0]:
          heapq.heappush(top_n_docs,(cos_sim,doc))
        else:
          heapq.heappush(top_n_docs,mini)
      else:
        heapq.heapify(top_n_docs)
        heapq.heappush(top_n_docs,(cos_sim,doc))
    top_n_docs=list(top_n_docs)
    #sorted scores results
    listi= sorted([(int(tup[1]),float(tup[0])) for tup in top_n_docs], key=lambda x: x[1], reverse=True)[:100]
    return [(tup[0],docs_titles[tup[0]]) for tup in listi if tup[0] in docs_titles]

''''BM25 METHOD'''
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, path, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.path = path
        self.N = len(self.index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_documents_BM25(self,query,path,index):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = []
        avg=0
        for term in np.unique(query):
            if term in index.df:
                pl=InvertedIndex.read_posting_list(self.index,term,path)
                candidates+=pl
                avg+=index.term_total[term]/len(pl)

            else:
                continue
        res=defaultdict(int)
        for doc, tf in candidates:
            res[doc]+=int(tf)
        sstresh=avg*10
        return [doc_id for doc_id,tf in res.items() if tf>sstresh]

    def bm25_search(self, query,path,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        scores = {}
        self.idf = self.calc_idf(query)
        term_tf={}
        for term in query:
            if term in self.index.df:
                # dict: key - term value - dict posting list of term (key - doc_id , value - tf)
                term_tf[term] = dict(InvertedIndex.read_posting_list(self.index,term,path))
        docs = self.get_candidate_documents_BM25(query,path,self.index)
        docList = []
        for doc in docs:
            docList.append((doc,self._score(query, doc,term_tf)))
         #sort docScores by scores
        return sorted(docList,key=lambda x: x[1],reverse=True)[:N]


    def _score(self, query, doc_id,term_tf):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """

        score = 0.0
        if doc_id in self.index.DL:
            doc_len = self.index.DL[doc_id]
            for term in query:
                if term in term_tf and doc_id in term_tf[term]:
                    tfij = term_tf[term][doc_id]
                    numerator = self.idf[term] * tfij * (self.k1 + 1)
                    B = 1 - self.b + self.b * doc_len / self.AVGDL
                    denominator = tfij + self.k1 * B
                    score += (numerator / denominator)
        return score