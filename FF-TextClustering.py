# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 06:41:06 2019

@author: nitis
"""

import nltk
import re

from nltk.tokenize import RegexpTokenizer

from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

import collections
from sklearn.feature_extraction.text import CountVectorizer


allReviews = []    ##
reviewList = []
L_dict = {}
i = 0
tokenizer = RegexpTokenizer(r"\w+")
with open("foods.txt", encoding="ISO-8859-1") as foods:
    for line in foods:
       if(i < 300000):
          i+=1
          colonIndex = line.find(":")
          if(line[:colonIndex] == "review/text"):
              review = line[colonIndex+1 : ].strip()
              allReviews.append(review)
              wordList = tokenizer.tokenize(review)
              reviewList.append(wordList)
              for word in wordList:
                  if (not word in ['br', 't', 've', 'll']) and word.isalpha():
                      if(word.lower() in L_dict.keys()):
                          L_dict[word.lower()] += 1
                      else:
                          L_dict[word.lower()] = 1
foods.close()

L = list(L_dict.keys())

len(L)



LongStopwordList = []
with open("StopWords.txt") as StopWords:
    for line in StopWords:
        LongStopwordList.append(line.strip())
StopWords.close()


# len(LongStopwordList)

# Cleaned list of words
W = [word for word in L if word not in LongStopwordList]

print(len(W))

tempDict = {}
for word in W:
    tempDict[word] = L_dict[word]


mostCommon = collections.Counter(tempDict).most_common(500)

with open("mostCommon500.txt", "w") as output:
    output.write(str(mostCommon))
#files.download('mostCommon500.txt')
output.close()

mostCommonWords = [word[0] for word in mostCommon]

# mostCommonWords

vectorizer = CountVectorizer(vocabulary = mostCommonWords)
X = vectorizer.fit_transform(allReviews)

from sklearn.cluster import KMeans
print("Clustering started");
kmeans = KMeans(n_clusters = 10, max_iter=10, n_jobs=-1).fit(X)
print("CLustering completed")


sortedCenters = kmeans.cluster_centers_.argsort()[:,::-1]
top5 = []
for i in range(len(sortedCenters)):
    temp = []
    for j in list(sortedCenters[i, :5]):
        temp.append(mostCommonWords[j])
    top5.append(temp)
print(top5)