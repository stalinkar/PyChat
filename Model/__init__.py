import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import tensorflow as tf  # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import Sequential  # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout

# nltk.download("punkt")# required package for tokenization
# nltk.download("wordnet")# word database

# Load data
raw_data = open('..\\Data\\data.json', 'r', encoding='utf-8').read()
data = json.loads(raw_data)
# print(data)

# 4 Creating data X and data_Y
words = []  # For Bow model/ vocabulary for patterns
classes = []  # For Bow model/ vocabulary for tags
data_X = []  # For storing each pattern
data_y = []  # For storing tag corresponding to each pattern in data_X

# Iterating over all the intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)  # tokenize each pattern
        words.extend(tokens)  # and append tokens to words
        data_X.append(pattern)  # appending pattern to data_X
        data_y.append(intent["tag"]),  # appending the associated tag to each pattern

    # adding the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

# 5 Text to Numbers
training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated to
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    # add the one hot encoded Bow and associated classes to training
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = num.array(training, dtype=object)
# split the features and target labels
train_X = num.array(list(training[:, 0]))
train_Y = num.array(list(training[:, 1]))

# 6 The Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
# adam = tf.keras.optimizers.Adam (learning_rate=0.01, decay=1e-6)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)
# model.save('modelpf')
