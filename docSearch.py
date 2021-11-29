#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:51:57 2021

@author: umadevi
"""

import requests
import json
from bs4 import BeautifulSoup as bs
import urllib.request
import urllib.parse 
import pandas as pd
from urllib.request import urlopen
from urllib.request import Request
from urllib.parse import urlparse
def get_page(url):
    response = urllib.request.urlopen(urllib.request.Request(url, 
                                                            headers={'User-Agent': 'Mozilla'} ))
    soup = bs(response, 
                         'html.parser', 
                         from_encoding=response.info().get_param('charset'))
    
    return soup
import pandas as pd
def roboText(robots):
    disallow = []
    lines = str(robots).splitlines()
    for line in lines:

        if line.strip():
            if not line.startswith('#'):
                split = line.split(':', maxsplit=1)
                #data.append([split[0].strip(), split[1].strip()])
                if(split[0].strip() == 'Disallow'):
                    disallow.append(split[1].strip())

    return disallow
robots = get_page("https://www.coventry.ac.uk/robots.txt")
roboTextDisallowList = roboText(robots)
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
            if(roboTextDisallowList.__contains__(links[i])):
                continue
            if not links[i]['href'].startswith('#') and not links[i]['href'].startswith('/') and '/organisations/' not in links[i]['href']:
                #print("Main Link: ",links[i]['href'])
                subURL = links[i]['href']
                Q.append(subURL)
                pubURL = subURL + '/publications'
                pubResponse = requests.get(pubURL)
                pubContent = bs(pubResponse.text,'html.parser')
                pubLinks = pubContent.findAll('a')
                for j in range(0, len(pubLinks)):
                    #if(roboTextDisallowList.__contains__(pubLinks[i])):
                        #continue
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
                
urlDetails,DispDescription = mycrawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/persons/',10)



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


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import json
vectorizerModel = TfidfVectorizer()
X = vectorizerModel.fit_transform(StopWordStemRemoved)
X = X.T.toarray()
df = pd.DataFrame(X, index=vectorizerModel.get_feature_names())
#print(df)

def get_similarity(q):
    #print("Query: ", q)
    CleandQuery = docClensing([q])
    #print(CleandQuery)
    StopWordRemQuery = stopWordStemRemoval(CleandQuery)
    #print(StopWordRemQuery[0])
    DispList = []
    q = [StopWordRemQuery[0]]
    q_vector = vectorizerModel.transform(q).toarray().reshape(df.shape[0],)
    similarity = {}
    for i in range(len(urlDetails)):
        similarity[i] = np.dot(df.loc[:,i].values,q_vector) / np.linalg.norm(df.loc[:,i]) * np.linalg.norm(q_vector)
    #print(similarity)
    similarity_sorted = sorted(similarity.items(),key = lambda X: X[1], reverse = True)
    #print(similarity_sorted)
    for k, v in similarity_sorted:
        if(v != 0.0):
            #print(DispDescription[k])
            DispList.append(DispDescription[k])
            #print("Similariry: ", v)
    #return(json.dumps(DispList, indent = 2, sort_keys = False))
    return(DispList)
    

#SearchResults = get_similarity('Tim Aldsworth and 2002')
#print(SearchResults)
