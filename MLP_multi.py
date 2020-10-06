#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
dataset = np.loadtxt('./tictac_multi.txt')


# In[46]:


np.random.shuffle(dataset)


# In[47]:


X = dataset[:, : 9 ]
Y = dataset[:, 9 : ]


# In[48]:


print(X.shape)
print(Y.shape)


# In[49]:


from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from numpy import arange
from numpy import argmax
from sklearn.metrics import f1_score
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score


# In[50]:


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# In[51]:


def MLP(train_X, test_X, train_Y, test_Y):
    n_neighbors = 15
    print(test_X.shape)
    print(test_Y.shape)
    parameter_space = {
    'hidden_layer_sizes': [(100,50,25,9)],
    'activation': ['tanh', 'relu', 'sigmoid'],
    'solver': ['adam','lbfgs','sgd'],
    'alpha': [0.0001, 0.05],
    'early_stopping' : [True],
    'learning_rate': ['constant','adaptive','invscaling'],
    }
    mlp = MLPRegressor(random_state=1, max_iter=1000)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    #clf = MLPRegressor(random_state=1, max_iter=3500, solver='lbfgs', early_stopping=True, activation='logistic')
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    res_cm = res
    
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    #print("#### HYPER PARAMETERS ########")
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
    # define thresholds
    thresholds = arange(0, 1, 0.001)
    
    # evaluate each threshold
    scores = [f1_score(test_Y, to_labels(res, t), average = 'weighted') for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    threshold_res = np.where(res < thresholds[ix], 0, 1)
    print("########## Y_actual #########")
    print(test_Y)
    print("######### T_RES ############")
    print(threshold_res)
#     res = np.asarray(res)
#     res = np.reshape(res, (-1, test_Y.shape[0]))
#     accuracy = 0.0
#     for i in range(res.shape[1]):
#         if(res[0][i] == test_Y[i][0]):
#             accuracy += 1
    
    #print(accuracy)
    #print("Accuracy = " + str(accuracy/res.shape[1]))
    accuracy = 0
    mae = metrics.mean_absolute_error(test_Y, threshold_res)
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, threshold_res))
    print('Mean Squared Error:', metrics.mean_squared_error(test_Y, threshold_res))
    mse = np.sqrt(metrics.mean_squared_error(test_Y, threshold_res))
    print('Root Mean Squared Error:', mse)
    
    #calculating accuracy
    vectorAccuracy = np.empty(9)
    
    for i in range(9):
        vectorAccuracy[i] = accuracy_score(test_Y[:, i], threshold_res[:, i], normalize = False)
    
    accuracy = np.sum(vectorAccuracy) / np.shape(test_Y)[0] * 9 
    print("Accuracy: " + str(accuracy))
#     accuracy = 100 - (mae * 100)
#     print("Accuracy:" + str(accuracy))
    
    return accuracy, clf,res_cm


# In[ ]:


from skmultilearn.model_selection import IterativeStratification

accuracy_max = 0.0
accuracy_min = 100.0
accuracy_avg = 0.0

best_clf = 0

k_cross_fold = 10

skf = IterativeStratification(n_splits = k_cross_fold)
for train_index, test_index in skf.split(X, Y):
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    
    accuracy, clf, res_cm = MLP(train_X, test_X, train_Y, test_Y)
    #accuracy *= 100
    
    if(accuracy > accuracy_max):
        accuracy_max = accuracy
        best_clf = clf
    accuracy_min = min(accuracy_min, accuracy)
    accuracy_avg += accuracy
    
    
    print("#### CONFUSION MATRIX ##########")
    rounded_labels = np.argmax(test_Y, axis = 1)
    cm = confusion_matrix(rounded_labels, res_cm.argmax(axis = 1))
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    
    print("#######################")
    
print("Max accuracy = " + str(accuracy_max))
print("Min accuracy = " + str(accuracy_min))
print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))
filename = 'mlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))


# In[ ]:




