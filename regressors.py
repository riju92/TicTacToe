import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score
from numpy import arange
from numpy import argmax
from numpy.linalg import inv
from sklearn.model_selection import GridSearchCV
from numpy import arange
from numpy import argmax
from sklearn import neighbors


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def load_data(dataset_final):
    np.random.shuffle(dataset_final)

    X = dataset_final[:, :9]
    Y = dataset_final[:,9:]
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=42)
    print("######### TRAIN DIM ############")
    print("X:" + str(train_X.shape) + "Y:" + str(train_Y.shape))
    print("######### TEST DIM ############")
    print("X:" + str(test_X.shape) + "Y:" + str(test_Y.shape))
    return train_X, test_X, train_Y, test_Y

def display_test_metrics(test_X, test_Y, clf):
    print("######### TEST METRICS ###############")

    clf.fit(test_X, test_Y)
    pred_Y = clf.predict(test_X)
    res_cm = pred_Y
    accuracy = 0.0
    # define thresholds
    thresholds = arange(0, 1, 0.001)
    
    # evaluate each threshold
    scores = [f1_score(test_Y, to_labels(pred_Y, t), average = 'weighted') for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    threshold_res = np.where(pred_Y < thresholds[ix], 0, 1)
    # print("########## Y_actual #########")
    # print(test_Y)
    # print("######### T_RES ############")
    # print(threshold_res)
    #accuracy and other metrics
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
    
    accuracy = np.sum(vectorAccuracy) / (np.shape(test_Y)[0] * 9) 
    print("Accuracy: " + str(accuracy * 100))

    print("#### CONFUSION MATRIX ##########")
    rounded_labels = np.argmax(test_Y, axis = 1)
    cm = confusion_matrix(rounded_labels, res_cm.argmax(axis = 1))
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

def linearREG(train_X, test_X, train_Y, test_Y):
    print(test_X.shape)
    print(test_Y.shape)
    print(train_X.shape)
    print(train_Y.shape)
    
    #creating the output matrix
    Y_prediction = np.empty((np.shape(test_Y)[0], np.shape(test_Y)[1]))
    
    b = 1
    #train_X = (x + b for x in train_X) # adding bias
    for i in range(9):
        Y = train_Y[:, i]
        theta = np.linalg.pinv(train_X.T @ train_X) @ train_X.T @ Y
        theta = [x + b for x in theta] # adding bias
        #print("theta type:" + str(theta.type))
        #print("test_X:" + str(test_X.shape))
        pred = test_X @ theta
        Y_prediction[:, i] = pred
    res = Y_prediction
#     X = np.c_[np.ones(train_X.shape[0]),train_X]
#     X_transpose = X.T # riju
    
#     #inv_X = np.linalg.pinv(X)
#     #theta = np.dot(inv_X,np.dot(np.transpose(X),train_Y))
#     theta = inv(X_transpose.dot(X)).dot(X_transpose).dot(train_Y) # riju
#     X_t = np.c_[np.ones(test_X.shape[0]),test_X] 
#     Y_prediction = np.dot(X_t,theta)
#     res = Y_prediction
    
    
    #one hot encoding
    res = (res == res.max(axis = 1)[:, None]).astype(int)
        
    res_cm = res
    
    # print("######### Y ############")
    # print(test_Y)
    # print("%%%%%%%%%%% T_RES %%%%%%%%%%%%%%%%%%%")
    # print(res)
    threshold_res = res
  
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, threshold_res))
    print('Mean Squared Error:', metrics.mean_squared_error(test_Y, threshold_res))
    mse = np.sqrt(metrics.mean_squared_error(test_Y, threshold_res))
    print('Root Mean Squared Error:', mse)
    
    #calculating accuracy
    vectorAccuracy = np.empty(9)
    
    for i in range(9):
        vectorAccuracy[i] = accuracy_score(test_Y[:, i], threshold_res[:, i], normalize = False)
    
    accuracy = np.sum(vectorAccuracy) / (np.shape(test_Y)[0] * 9) 
    print("Accuracy: " + str(accuracy * 100))
#     print('Accuracy: {:.2f}'.format(accuracy))
#     print("Accuracy = " + str(accuracy/res.shape[1]))
    
    return accuracy, theta, res_cm


def LR():
    print("########### LINEAR REGRESSOR ##############")
    dataset = np.loadtxt('./datasets-part1/tictac_multi.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = IterativeStratification(n_splits = k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        #print(str(train_index[0]) + " and " + str(train_index[-1])  + " and " + "shape = " + str(train_index.shape))
        #print(str(test_index[0]) + " and " + str(test_index[-1])   + " and " + "shape = " + str(test_index.shape))
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf, res_cm = linearREG(train_X, test_X, train_Y, test_Y)
        accuracy *= 100.0
        #print("A:" + str(accuracy))
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        #print(accuracy_avg)
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    
def KNNRegressor(train_X, test_X, train_Y, test_Y):
    # print(test_X.shape)
    # print(test_Y.shape)
    # print(train_X.shape)
    # print(train_Y.shape)
    
    accuracy = 0
    
    #Hyper Parameters Set
    #params = {'n_neighbors': [5,6,7,8,9,10]}
    params = {'n_neighbors':[5,6,7,8,9,10],
            'leaf_size':[1,2,3,5],
            'weights':['uniform','distance'],
            'algorithm':['auto','ball_tree','kd_tree','brute'],
            'n_jobs':[-1]}
    
    knn = neighbors.KNeighborsRegressor()
    #model = OneVsRestClassifier(knn)
    #Use GridSearch
    clf = GridSearchCV(knn, param_grid=params, n_jobs= -1)    
    #lf = knn
    clf.fit(train_X, train_Y)
    
    #The best hyper parameters set
    print("Best Hyper Parameters:\n",clf.best_params_)
    
    res = clf.predict(test_X)
    res_cm = res
    # define thresholds
    thresholds = arange(0, 1, 0.001)
    
    # evaluate each threshold
    scores = [f1_score(test_Y, to_labels(res, t), average = 'weighted') for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    threshold_res = np.where(res < thresholds[ix], 0, 1)
    # print("########## Y_actual #########")
    # print(test_Y)
    # print("######### T_RES ############")
    # print(threshold_res)
    #accuracy and other metrics
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
    
    accuracy = np.sum(vectorAccuracy) / (np.shape(test_Y)[0] * 9) 
    print("Accuracy: " + str(accuracy * 100))
    
    # print("#### CONFUSION MATRIX ##########")
    # rounded_labels = np.argmax(test_Y, axis = 1)
    # cm = confusion_matrix(rounded_labels, res_cm.argmax(axis = 1))
    # cm = cm / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    
    return accuracy, clf

def KNN():
    print("########### KNN REGRESSOR ##############")
    dataset = np.loadtxt('./datasets-part1/tictac_multi.txt')
    trainX, testX, trainY, testY = load_data(dataset)  
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = IterativeStratification(n_splits = k_cross_fold)

    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = KNNRegressor(train_X, test_X, train_Y, test_Y)
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        #print(accuracy_avg)
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)


def MLP_regressor(train_X, test_X, train_Y, test_Y):
    parameter_space = {
    'hidden_layer_sizes': [(100,50,25)],
    'activation': ['tanh', 'relu', 'sigmoid'],
    'solver': ['adam','lbfgs','sgd'],
    'alpha': [0.0001, 0.05],
    'early_stopping' : [True],
    'learning_rate': ['constant','adaptive','invscaling'],
    }
    mlp = MLPRegressor(random_state=1, max_iter=1000)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
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
    # print("########## Y_actual #########")
    # print(test_Y)
    # print("######### T_RES ############")
    # print(threshold_res)
    #accuracy and other metrics
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
    
    accuracy = np.sum(vectorAccuracy) / (np.shape(test_Y)[0] * 9) 
    print("Accuracy: " + str(accuracy * 100))
#     accuracy = 100 - (mae * 100)
#     print("Accuracy:" + str(accuracy))
    
    return accuracy, clf,res_cm

def MLP():
    print("########### MLP REGRESSOR ##############")
    dataset = np.loadtxt('./datasets-part1/tictac_multi.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = IterativeStratification(n_splits = k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf, res_cm = MLP_regressor(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
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

    display_test_metrics(testX, testY, best_clf)

    filename = 'mlp_model_final.pkl'
    pickle.dump(clf, open(filename, 'wb'))



def main():
    LR()
    MLP()
    KNN()

main()