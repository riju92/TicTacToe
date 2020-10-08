import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score

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

def display_test_metrics(testX, testY, clf):
    print("######### TEST METRICS ###############")

    clf.fit(testX, testY)
    pred_Y = clf.predict(testX)
    res_cm = pred_Y
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(testY, pred_Y, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(testY)[0])
    print("Accuracy: " + str(accuracy * 100))

    print("#### CONFUSION MATRIX ##########")
    cm = confusion_matrix(testY, res_cm)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    print(cm)




def SVM_class1(train_X, test_X, train_Y, test_Y):
    #print("############ Linear SVM ############")
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    res_cm = res
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    
    return accuracy, clf

def SVM1():
    print("########### SVM CLASSIFIER 1 ##############")
    dataset = np.loadtxt('./datasets-part1/tictac_final.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = SVM_class1(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)
    

def SVM_class2(train_X, test_X, train_Y, test_Y):
    params = [{'gamma': [1, 0.1, 0.5],
                         'C': [10, 12]}]
    
    model = SVC(kernel = "linear")
    # set up GridSearchCV()
    clf = GridSearchCV(estimator = model,
                                param_grid = params,
                                scoring = 'accuracy',
                                cv = 10,
                                verbose = 1,
                                return_train_score = True,
                                n_jobs = -1)
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    #print(res)
    res_cm = res
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    return accuracy, clf

def SVM2():
    print("########### SVM CLASSIFIER 2 ##############")
    dataset = np.loadtxt('./datasets-part1/tictac_single.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        #print(str(train_index[0]) + " and " + str(train_index[-1])  + " and " + "shape = " + str(train_index.shape))
        #print(str(test_index[0]) + " and " + str(test_index[-1])   + " and " + "shape = " + str(test_index.shape))
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = SVM_class2(train_X, test_X, train_Y, test_Y)
        
        accuracy *= 100.0
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)

def KNN_class1(train_X, test_X, train_Y, test_Y):
#     n_neighbors = 15
#     print(test_X.shape)
#     print(test_Y.shape)
    
    #Hyper Parameters Set
    #params = {'n_neighbors': [5,6,7,8,9,10]}
    params = {'n_neighbors':[5],
            'leaf_size':[1],
            'weights':['uniform'],
            'algorithm':['auto'],
            'n_jobs':[-1]}
    
    knn = KNeighborsClassifier()
    
    #Use GridSearch
    clf = GridSearchCV(knn, param_grid=params, n_jobs= -1) 
    #clf = KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(train_X, train_Y)
    
    #The best hyper parameters set
    print("Best Hyper Parameters:\n",clf.best_params_)
    
    res = clf.predict(test_X)
    res_cm = res
    #print(res)
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    
    return accuracy, clf

def KNN1():
    print("######## KNN Classifier 1 ############")
    dataset = np.loadtxt('./datasets-part1/tictac_final.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = KNN_class1(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))
    
    display_test_metrics(testX, testY, best_clf)

def KNN_class2(train_X, test_X, train_Y, test_Y):
#     n_neighbors = 30
#     print(test_X.shape)
#     print(test_Y.shape)
#     print(train_X.shape)
#     print(train_Y.shape)
    params = {'n_neighbors':[5,6,7,8,9,10],
            'leaf_size':[1,2,3,5],
            'weights':['uniform', 'distance'],
            'algorithm':['auto', 'ball_tree','kd_tree','brute'],
            'n_jobs':[-1]}
    
    
    knn = KNeighborsClassifier()
    
    #Use GridSearch
    clf = GridSearchCV(knn, param_grid=params, n_jobs= -1) 
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    #The best hyper parameters set
    print("Best Hyper Parameters:\n",clf.best_params_)
    res_cm = res
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    
    return accuracy, clf

def KNN2():
    print("######## KNN Classifier 2 ############")
    dataset = np.loadtxt('./datasets-part1/tictac_single.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = KNN_class2(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)

def MLP_class1(train_X, test_X, train_Y, test_Y):
    parameter_space = {
    'hidden_layer_sizes': [(100,50,25)],
    'activation': ['tanh', 'relu', 'sigmoid'],
    'solver': ['adam','lbfgs','sgd'],
    'alpha': [0.0001, 0.05],
    'early_stopping' : [True],
    'learning_rate': ['constant','adaptive','invscaling'],
    }
    mlp = MLPClassifier(random_state=1, max_iter=1000)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
    clf.fit(train_X, train_Y)
    
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)
    
    res = clf.predict(test_X)
    res_cm = res
    accuracy = 0.0
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    
    return accuracy, clf

def MLP1():
    print("######## MLP Classifier 1 ############")
    dataset = np.loadtxt('./datasets-part1/tictac_final.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = MLP_class1(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)

def MLP_class2(train_X, test_X, train_Y, test_Y):
    # print(train_X.shape)
    # print(test_X.shape)
    # print(train_Y.shape)
    # print(test_Y.shape)
    # #exit(0)
    parameter_space = {
    'hidden_layer_sizes': [(100,50,25)],
    'activation': ['tanh', 'relu', 'sigmoid'],
    'solver': ['adam','lbfgs','sgd'],
    'alpha': [0.0001, 0.05],
    'early_stopping' : [True],
    'learning_rate': ['constant','adaptive','invscaling'],
    }
    mlp = MLPClassifier(random_state=1, max_iter=1000)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    res_cm = res

    accuracy = 0.0
    #print("Shape of res")
    #calculating accuracy
    accuracy = accuracy_score(test_Y, res, normalize = False)
    #print(accuracy)
    accuracy = np.sum(accuracy) / (np.shape(test_Y)[0])
    print("Accuracy: " + str(accuracy * 100))
    
    return accuracy, clf

def MLP2():
    print("######## MLP Classifier 2 ############")
    dataset = np.loadtxt('./datasets-part1/tictac_single.txt')
    trainX, testX, trainY, testY = load_data(dataset)
    accuracy_max = 0.0
    accuracy_min = 100.0
    accuracy_avg = 0.0

    best_clf = 0

    k_cross_fold = 10

    skf = StratifiedKFold(n_splits=k_cross_fold)
    for train_index, test_index in skf.split(trainX, trainY):
        train_X, test_X = trainX[train_index], trainX[test_index]
        train_Y, test_Y = trainY[train_index], trainY[test_index]
        
        accuracy, clf = MLP_class2(train_X, test_X, train_Y, test_Y)
        accuracy *= 100
        
        if(accuracy > accuracy_max):
            accuracy_max = accuracy
            best_clf = clf
        accuracy_min = min(accuracy_min, accuracy)
        accuracy_avg += accuracy
        
        print("#######################")
        
    print("Max accuracy = " + str(accuracy_max))
    print("Min accuracy = " + str(accuracy_min))
    print("Avg accuracy = " + str(accuracy_avg/k_cross_fold))

    display_test_metrics(testX, testY, best_clf)




def main():
    print("Models of the world, UNITE !!")
    print("\n")
    SVM1()
    print("\n")
    SVM2()
    print("\n")
    KNN1()
    print("\n")
    KNN2()
    print("\n")
    MLP1()
    print("\n")
    MLP2()


main()
