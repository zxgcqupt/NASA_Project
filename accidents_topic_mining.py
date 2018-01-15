"""
Created on Wed Dec 06 12:59:15 2017

@author: zhanx15
"""
#!/usr/bin/python
# encoding=utf8
import csv
import pandas as pd
import sys
import nltk


## read the data from ASRS database
i = 1
data = []
with open('ASRS_DBOnline.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if i == 1:
            row_name = row
        else:
            data.append(row)
        i = i + 1
    f.close()

## construct dataframe
df = pd.DataFrame(data, columns = row_name)

#########################################################
#############   Natural Language Processing   ###########
#########################################################

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora
import gensim
import string

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
stop_words = stopwords.words('english') + list(string.punctuation)

# Create p_stemmer of class PorterStemmer
p_stemmer = SnowballStemmer('english')

texts = []
index = df.columns.get_loc("Synopsis")

for i in range(df.shape[0]):
     # clean and tokenize document string
    raw = df.iloc[i, index].lower()
   # raw = raw.decode('utf8')
    
    tokens = tokenizer.tokenize(raw)
    
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stop_words]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn the tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 400, id2word = dictionary, distributed = False, passes=20)
                                           
topics = ldamodel.print_topics(num_topics = 400, num_words=6)

doc_ID = 6
topics_document= ldamodel.get_document_topics(dictionary.doc2bow(texts[doc_ID]))
print (df.iloc[doc_ID, index].lower())
print (topics_document)

for topicID, probability in topics_document:
    print (topics[topicID], probability)
    
    
## find all the operations/actions taken in the past accidents
index_action = df.columns.get_loc("Result")
