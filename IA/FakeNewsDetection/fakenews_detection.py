import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('punkt')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

#Importing Data sets

df_true = pd.read_csv("/Users/yassineseidou/Desktop/PROJETS TECHNIQUE/PROJET D'IA - DEEP FAKES:FAKE NEWS/FINALS_DOCS/IA/FakeNewsDetection/True.csv") #Data set with True informations
df_fake = pd.read_csv("/Users/yassineseidou/Desktop/PROJETS TECHNIQUE/PROJET D'IA - DEEP FAKES:FAKE NEWS/FINALS_DOCS/IA/FakeNewsDetection/Fake.csv") #Data set with Fake informations

#isfake = 0 then the information is true

df_true ['isfake'] = 0 

#isfake = 1 then the information is False

df_fake ['isfake'] = 1

df = pd.concat([df_true, df_fake]).reset_index(drop = True)

#Let's create a new variable composed by the title of the text and the text itself

df['original'] = df['title'] + ' ' + df['text']

nltk.download("stopwords") 

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use']) 

def preprocess(text):
    result = [] #creating a list where i'll append my tokens 
    for token in gensim.utils.simple_preprocess(text): #iterating in the next conditions on all the token generated 
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words: #verifying if the token is not present in the stopword of Gensim, in the customized stopwords and if the token length is superior to 3
            result.append(token) #If condition verified, append the tokens to the list created 
            
    return result

df['clean'] = df['original'].apply(preprocess) 

list_of_words = [] #creating a list where i'll append the words present in the variable 'clean'
for i in df.clean: #iterating on all the elements of the variable 'clean'
    for j in i: #iterating on all the words present in each element of the variable 'clean'
        list_of_words.append(j)

total_words = len(list(set(list_of_words))) 

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

maxlen = -1 #this variable is initalized to stock the max length in all the documents 
for doc in df.clean_joined: #iteration on all the elements of the variable 'clean_joined'
    tokens = nltk.word_tokenize(doc) #tokenizing the text of all doc and append it to tokens
    if(maxlen<len(tokens)): #verifying if the max length is lower than the tokens length
        maxlen = len(tokens) #if yes append maxlen take automatically the value of the lenght of the tokens
print("The maximum number of words in any document is =", maxlen)

# split data into test and train 

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

# Create a tokenizer to tokenize the words and create sequences of tokenized words


tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results


padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post')


for i,doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)

#Build and train the model


# Sequential Model

model = Sequential()

# embeddidng layer

model.add(Embedding(total_words, output_dim = 128))

# Bi-Directional RNN and LSTM

model.add(Bidirectional(LSTM(128)))

# Dense layers

model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

y_train = np.asarray(y_train)

# train the model


model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

# make prediction

pred = model.predict(padded_test)

# if the predicted value is >0.5 it is real else it is fake

prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

# getting the accuracy

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)

# category dict
category = { 0: 'Fake News', 1 : "Real News"}

def FakeNewsPredictor (model, sentence):
    sentence_tokens = tokenizer.texts_to_sequences([sentence])
    padded_sentence= pad_sequences(sentence_tokens, maxlen= 40, padding= 'post', truncating='post')
    prediction_sentence = model.predict(padded_sentence)
    if prediction_sentence [0][0]>= 0.5 :
        return 'Our model has classified this text as misinformation. Please verify the source of the information. Please enter a new text.'
    else :
        return 'Our model has classified this text as accurate information. Please enter a new text.'


sentence = " "

FakeNewsPredictor (model, sentence)