import json
import string
import random

import keras
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import tensorflow as tf  # A multidimensional array of elements is represented by this symbol.
from keras import Sequential  # Sequential groups a linear stack of layers into a tf.keras.Model
from keras.layers import Dense, Dropout
from Model import model

# nltk.download("punkt")# required package for tokenization
# nltk.download("wordnet")# word database

# Load data
from tensorflow.python.keras.models import load_model

arr_result = []


# 7 Preprocessing the Input
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w.lower():
                bow[idx] = 1
    return num.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(num.array([bow]))[0]  # Extracting probabilities
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)  # Sorting by values of probability in decreasing order
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])  # Contains labels (tags) for highest probability
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry! I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    arr_result.append(result)
    return result


def update_data(intents_json, txt):
    last_qns = arr_result[len(arr_result) - 2]
    list_of_intents = intents_json["intents"]
    num = 0
    for i in list_of_intents:
        index = 0
        for res in i["responses"]:
            if res == last_qns:
                intents_json["intents"][num]["responses"][index] = txt
                with open("../Data/data.json", "w") as jsonFile:
                    json.dump(intents_json, jsonFile)
                break
            index += 1
        num += 1
    return "Information corrected"
