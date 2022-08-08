# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:01:58 2022

@author: Alkios
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# first neural network with keras tutorial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm

os.chdir("C:/Users/Alkios/Downloads/wine/")

df = pd.read_csv (r'C:/Users/Alkios/Downloads/wine/winequality-white.csv', delimiter=';')
print (df)



corr_df = df.corr()
print("The correlation DataFrame is:")
print(corr_df, "\n")
#df.drop(df.columns[[2, 3, 5, 8, 9]], axis = 1, inplace = True)
#df.drop(df.columns[[2, 3, 5]], axis = 1, inplace = True) 

nb_inputs = 11
nb_output = 10
X = df.iloc[:, 0:nb_inputs]
Y = df.iloc[:, [nb_inputs]]

encoded = to_categorical(Y)
Y = pd.DataFrame(encoded)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

#seed = 7
#tf.random.set_seed(seed)


def create_model(): 
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model




model = KerasClassifier(model=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 40, 80]
epochs = [50, 150, 300]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





def create_model(optimizer='adam'): 
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(model=create_model, verbose=0)


optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(model__optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




def create_model():
	# create model
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))
    return model

model = KerasClassifier(model=create_model, loss="categorical_crossentropy", optimizer="SGD", epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.4, 0.8]
param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





def create_model(init_mode='uniform'):
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))
        
    opt = keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# create model
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(model__init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))









def create_model(activation='relu'):
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation=activation))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation=activation))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation=activation))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))
        
    opt = keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(model__activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




def create_model(dropout_rate, weight_constraint):

    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), kernel_initializer='uniform', activation='linear', kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nb_output, activation='softmax'))
        
    opt = keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1.0, 3.0, 5.0]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint)
#param_grid = dict(model__dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



def create_model(neurons):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(nb_inputs,), activation='relu'))
    model.add(Dense(neurons, activation='linear'))
    model.add(Dense(nb_output, activation='softmax'))
        
    opt = keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [32, 64, 128, 256]
param_grid = dict(model__neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))









acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits = 5, shuffle = True)
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    model = Sequential()
    model.add(Dense(64, input_shape=(nb_inputs,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='softmax'))
    
    opt = keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    # Fit data to model
    history = model.fit(inputs[train], targets[train],
              batch_size=16,
              epochs=300,
              verbose = 0)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # Increase fold number
    fold_no = fold_no + 1