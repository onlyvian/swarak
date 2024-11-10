import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
from xgboost import XGBClassifier
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter("ignore")


import nltk 
nltk.download('all')



def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms+=syn.lemma_names()
    return set(synonyms)

# utlities for pre-processing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')


df_comb = pd.read_csv("dataset\\dis_comb.csv") # Disease combination
df_norm = pd.read_csv("dataset\\dis_norm.csv") # Individual Disease

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

lr = LogisticRegression()
lr = lr.fit(X, Y)
scores = cross_val_score(lr, X, Y, cv=5)

X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]

symptoms = list(X.columns)

user_symptoms = str(input("Please enter symptoms separated by comma(,):\n")).lower().split(',')
# Preprocessing the input symptoms
processed_user_symptoms=[]
for sym in user_symptoms:
    sym=sym.strip()
    sym=sym.replace('-',' ')
    sym=sym.replace("'",'')
    sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
    processed_user_symptoms.append(sym)

"""Pre-processing on symptoms entered by user is done."""

# Taking each user symptom and finding all its synonyms and appending it to the pre-processed symptom string
user_symptoms = []
for user_sym in processed_user_symptoms:
    user_sym = user_sym.split()
    str_sym = set()
    for comb in range(1, len(user_sym)+1):
        for subset in combinations(user_sym, comb):
            subset=' '.join(subset)
            subset = synonyms(subset) 
            str_sym.update(subset)
    str_sym.add(' '.join(user_sym))
    user_symptoms.append(' '.join(str_sym).replace('_',' '))

#  query expansion performed by joining synonyms found for each symptoms initially entered
# print("After query expansion done by using the symptoms entered")
# print(user_symptoms)



found_symptoms = set()
for idx, data_sym in enumerate(symptoms):
    data_sym_split=data_sym.split()
    for user_sym in user_symptoms:
        count=0
        for symp in data_sym_split:
            if symp in user_sym.split():
                count+=1
        if count/len(data_sym_split)>0.5:
            found_symptoms.add(data_sym)
found_symptoms = list(found_symptoms)



# Print all found symptoms
print("Top matching symptoms from your search!")
for idx, symp in enumerate(found_symptoms):
    print(idx,":",symp)
    
# Show the related symptoms found in the dataset and ask user to select among them
select_list = input("\nPlease select the relevant symptoms. Enter indices (separated-space):\n").split()

# Find other relevant symptoms from the dataset based on user symptoms based on the highest co-occurance with the
# ones that is input by the user
dis_list = set()
final_symp = [] 
counter_list = []
for idx in select_list:
    symp=found_symptoms[int(idx)]
    final_symp.append(symp)
    dis_list.update(set(df_norm[df_norm[symp]==1]['label_dis']))
   
for dis in dis_list:
    row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
    row[0].pop(0)
    for idx,val in enumerate(row[0]):
        if val!=0 and symptoms[idx] not in final_symp:
            counter_list.append(symptoms[idx])


# Symptoms that co-occur with the ones selected by user              
dict_symp = dict(Counter(counter_list))
dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)   
#print(dict_symp_tup)



# Iteratively, suggest top co-occuring symptoms to the user and ask to select the ones applicable 
found_symptoms=[]
count=0
for tup in dict_symp_tup:
    count+=1
    found_symptoms.append(tup[0])
    if count%5==0 or count==len(dict_symp_tup):
        print("\nCommon co-occuring symptoms:")
        for idx,ele in enumerate(found_symptoms):
            print(idx,":",ele)
        select_list = input("Do you have have of these symptoms? If Yes, enter the indices (space-separated), 'no' to stop, '-1' to skip:\n").lower().split();
        if select_list[0]=='no':
            break
        if select_list[0]=='-1':
            found_symptoms = [] 
            continue
        for idx in select_list:
            final_symp.append(found_symptoms[int(idx)])
        found_symptoms = []

"""Final Symptom list"""

# Create query vector based on symptoms selected by the user
print("\nFinal list of Symptoms that will be used for prediction:")
sample_x = [0 for x in range(0,len(symptoms))]
for val in final_symp:
    print(val)
    sample_x[symptoms.index(val)]=1

"""Prediction of disease is done"""

# Predict disease
lr = LogisticRegression()
lr = lr.fit(X, Y)
prediction = lr.predict_proba([sample_x])


k = 3
diseases = list(set(Y['label_dis']))
diseases.sort()
topk = prediction[0].argsort()[-k:][::-1]


print(f"\nTop {k} diseases predicted based on symptoms")
topk_dict = {}

for idx,t in  enumerate(topk):
    match_sym=set()
    row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
    row[0].pop(0)

    for idx,val in enumerate(row[0]):
        if val!=0:
            match_sym.add(symptoms[idx])
    prob = (len(match_sym.intersection(set(final_symp)))+1)/(len(set(final_symp))+1)
    prob *= mean(scores)
    topk_dict[t] = prob
j = 0
topk_index_mapping = {}
topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
for key in topk_sorted:
  prob = topk_sorted[key]*100
  print(str(j+1) + " Disease name:",diseases[key], "\tProbability:",str(round(prob, 2))+"%")
  topk_index_mapping[j] = key
  j += 1
