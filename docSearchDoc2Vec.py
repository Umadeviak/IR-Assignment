#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 23:48:29 2021

@author: Umadevi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

from gensim.models import Doc2Vec
from collections import namedtuple
import gensim.utils
from langdetect import detect
import re
import string

import requests
import json
from bs4 import BeautifulSoup as bs

Details = []
DispDescription = []
DispList = []
Q = []
def mycrawler(seed, maxcount):
    Q = [seed] #this is the queue which initially contains the given seed URL
    count = 0
    Title = ''
    Authors = ''
    Abstract = ''
    KeyWords = ''
    Year = ''
    Month = ''
    DOI = ''
    while(Q!=[] and count < maxcount):
        count +=1
        url = Q.pop(0)
        response = requests.get(url)
        content = bs(response.text,'html.parser')
        links = content.findAll('a', {'class': 'link'})
        
        for i in range(0, len(links)):
            if not links[i]['href'].startswith('#') and not links[i]['href'].startswith('/') and '/organisations/' not in links[i]['href']:
                #print("Main Link: ",links[i]['href'])
                subURL = links[i]['href']
                Q.append(subURL)
                pubURL = subURL + '/publications'
                pubResponse = requests.get(pubURL)
                pubContent = bs(pubResponse.text,'html.parser')
                pubLinks = pubContent.findAll('a')
                for j in range(0, len(pubLinks)):
                    if not pubLinks[j]['href'].startswith('#') and not pubLinks[j]['href'].startswith('/') and '/en/publications/' in pubLinks[j]['href']:
                        reqURL = pubLinks[j]['href']
                        if(reqURL != None and reqURL != '/'):
                            reqURL = reqURL.strip()
                            reqResponse = requests.get(reqURL)
                            reqContent = bs(reqResponse.text,'html.parser')
                            description = reqContent.find('div',{'id' : 'cite-BIBTEX'}).get_text().replace('",','/').replace(',','//').replace('booktitle','book')
                            temp = description.replace('//',',').split('/')
                            #print(temp)
                            for i in range(0, len(temp)):
                                if('title' in temp[i]):
                                    Title = temp[i].split(',')[1]
                                elif(temp[i].strip().startswith('author')):
                                    Authors = temp[i]    
                                elif(temp[i].strip().startswith('abstract')):
                                    Abstract = temp[i]
                                elif(temp[i].strip().startswith('keywords')):
                                    KeyWords = temp[i]
                                elif(temp[i].strip().startswith('year')):
                                    Year = temp[i]
                                elif(temp[i].strip().startswith('month')):
                                    Month = temp[i][0:16]
                                elif(temp[i].strip().startswith('doi')):
                                    DOI = temp[i]
                            Des = Title + Authors + Abstract + KeyWords + Year + Month + DOI
                            Details.append(Des)
                            #DispDetails = 'Author URL: ' + links[i]['href'] + '\n' + '   Publication URL: ' + pubLinks[j]['href'] + '  ' #+ description 
                            DispDetails = {
                                'authorUrl': subURL,
                                'publicationUrl': pubLinks[j]['href'],
                                'Description': Des
                            }
                            #print(DispDetails)
                            DispDescription.append(DispDetails)

        return(Details,DispDescription)
                
urlDetails,DispDescription = mycrawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/persons/',1)



import re

def docClensing(doc):
    CleanedDetails = []
    for d in doc:
        temp = re.sub(r'[^\x00-\x7F]+','',d)
        temp = re.sub(r'@\w+','',temp)
        temp = re.sub(r'\n','',temp)
        temp = re.sub(r'//','',temp)
        temp = re.sub(r'/','',temp)
        temp = temp.replace('{','').replace('}','').replace('=','').replace('"','').replace('\'','')
        temp = temp.replace('author','').replace('title','').replace('month','').replace('year','').replace('day','')
        temp = temp.lower()
        temp = " ".join(temp.split())
        CleanedDetails.append(temp)
    return(CleanedDetails)

CleanedDetails = docClensing(urlDetails)


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

sw = stopwords.words('english')
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def stopWordStemRemoval(doc):
    StopWordStemRemoved = []
    for doc in doc:
        tokens = word_tokenize(doc)
        tmp = ""
        for w in tokens:
            if w not in sw:
                tmp += ps.stem(w) + " "
        StopWordStemRemoved.append(tmp)

    return(StopWordStemRemoved)

StopWordStemRemoved = stopWordStemRemoval(CleanedDetails)
#print(StopWordStemRemoved)

data = np.array(StopWordStemRemoved)

import re
for line_no, line in enumerate(data):
    if (len(line)>150):
        if (detect(line) == 'en') :
            line = re.sub('', line)
            tokens = gensim.utils.to_unicode(line).lower().split()
            words = tokens[0:]
            print(line_no, len(line),len(words))

SentimentDocument = namedtuple('SentimentDocument', 'words tags original_number')
n=0
alldocs = []  # Will hold all docs in original order

regex = re.compile('[%s]' % re.escape(string.punctuation)) #to remove punctuation

for line_no, line in enumerate(data):
    if (len(line)>150):
        if (detect(line) == 'en') :
            line = regex.sub('', line)
            tokens = gensim.utils.to_unicode(line).lower().split()
            words = tokens[0:]
            tags = [n]
            #title = titles[line_no]
            alldocs.append(SentimentDocument(words, tags, line_no))
            n=n+1
            
l = []
for doc in alldocs:
    l.append(len(doc.words))

print('Number of Documents : ', len(alldocs))
print('Mean length of documents : ', np.mean(l))

index = 0
doc = alldocs[index]
print(doc, '\n')
print(data[doc.original_number])

model = Doc2Vec(dm=1, window=10,hs=0,min_count=10,dbow_words=1,sample=1e-5)
model.build_vocab(alldocs)
model.train(alldocs, total_examples=model.corpus_count, epochs=100, start_alpha=0.01, end_alpha=0.01)
model.save("model")


model.wv.most_similar_cosmul(positive = ["tim","aldsworth"])


tokens = "Tim Aldsworth"

new_vector = model.infer_vector(tokens.split() ,alpha=0.001)
tagsim = model.docvecs.most_similar([new_vector])[0]

docsim = alldocs[tagsim[0] ]

print("Document : ", data[docsim.original_number], "\n")
#print("Titre : ", docsim.title)
print("Distance : ", tagsim[1])


tokens ="Tim Aldsworth"

new_vector = model.infer_vector(tokens.split() ,alpha=0.001)
sims = model.docvecs.most_similar([new_vector]) # get *all* similar documents

print(sims)
print("Most : " , data[alldocs[sims[0][0]].original_number], "\n") 
print("Median : " , data[alldocs[sims[2][0]].original_number], "\n")
print("Least : " , data[alldocs[sims[-1][0]].original_number])
            