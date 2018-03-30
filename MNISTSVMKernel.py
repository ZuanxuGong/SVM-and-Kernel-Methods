# -*- coding: utf-8 -*-
import string
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

file_features = ['trainingData.txt', 'validationData.txt', 'testData.txt']
file_labels = ['trainingLabels.txt', 'validationLabels.txt', 'testLabels.txt']

def generateData(file_feature, file_label):
    labels = [];
    for line in open(file_label):
        labels.append(int(line));
        
    features = [[] for i in range(len(labels))];
    count = 0;
    for line in open(file_feature):
        for x in line.split(','):
            features[count].append(int(x) / 2550.0);
        count += 1;
    return features,labels

def lossCompute(featuresTrain, features, labels, w, lenda, KMatrixT):
    loss = 0
    ones = np.ones((len(features), 6107))
    KMatrix = np.dot(features, featuresTrain.T) + ones
    KMatrix = np.multiply(np.multiply(KMatrix, KMatrix), KMatrix)
    Ka = np.dot(KMatrix, w.T)    

    for i in range(len(features)):
        loss += max(0, 1 - labels[i] * Ka[i])
    
    return (loss / len(labels) + lenda / 2 * np.dot(np.dot(w, KMatrixT), w.T));

def lossCompute1(featuresTrain, features, labels, w, lenda, KMatrixT):
    ones = np.ones((len(features), 6107))
    KMatrix = np.dot(features, featuresTrain.T) + ones
    KMatrix = np.multiply(np.multiply(KMatrix, KMatrix), KMatrix)
    Ka = np.dot(KMatrix, w.T)
    count = 0
    for i in range(len(features)):
        y = labels[i]
        predict = Ka[i];
        if(predict >= 0 and y == 1):
            count += 1;
        elif(predict < 0 and y == -1):
            count += 1;            
    return  1 - count / float(len(labels));

#get training + validation + testing data (features + labels)
[featuresTrain, labelsTrain] = generateData(file_features[0], file_labels[0]);
[featuresValidation, labelsValidation] = generateData(file_features[1], file_labels[1]);
[featuresTest, labelsTest] = generateData(file_features[2], file_labels[2]);

featuresTrain = np.array(featuresTrain)
featuresValidation = np.array(featuresValidation)
featuresTest = np.array(featuresTest)
labelsTrain = np.array(labelsTrain)
labelsValidation = np.array(labelsValidation)
labelsTest = np.array(labelsTest)
ones = np.ones((6107,6107))

mius = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
lendas = [0.000001, 0.000003,0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]

#cross-validation
for miu in mius:
    for lenda in lendas:
        w = np.zeros((1,6107))
        for iter in range(500):
            fwdao = 0
            KMatrix = np.dot(featuresTrain, featuresTrain.T) + ones
            KMatrix = np.multiply(np.multiply(KMatrix, KMatrix), KMatrix)
            Ka = np.dot(KMatrix, w.T)
            for i in range(6107):
                if labelsTrain[i] * Ka[i] < 1:
                    fwdao += -labelsTrain[i] * KMatrix[i]            
            w = w - miu * (fwdao / 6107 + lenda * Ka.T)
            sys.stdout.write(' ' * 10 + '\r')
            sys.stdout.flush()
            sys.stdout.write("Iteration:" + str(iter) + '/' + str(500) +'\r')
            sys.stdout.flush()
        print("learning rate u = %f lenda = %f" % (miu, lenda))
        #check training + validation performance
        lossTrain = lossCompute(featuresTrain, featuresTrain, labelsTrain, w, lenda, KMatrix)
        lossValidation1 = lossCompute1(featuresTrain, featuresValidation, labelsValidation, w, lenda, KMatrix)
        print("Regularization training loss: %f  Misclassification validation loss: %f" % (lossTrain, lossValidation1))

#best miu = 0.003 and lenda = 0.00003
miu = 0.003
lenda = 0.00003
w = np.zeros((1,6107))
lossTrain = []
lossValidation1 = []
for iter in range(500):
    fwdao = 0
    KMatrix = np.dot(featuresTrain, featuresTrain.T) + ones
    KMatrix = np.multiply(np.multiply(KMatrix, KMatrix), KMatrix)
    Ka = np.dot(KMatrix, w.T)
    for i in range(6107):
        if labelsTrain[i] * Ka[i] < 1:
            fwdao += -labelsTrain[i] * KMatrix[i]            
    w = w - miu * (fwdao / 6107 + lenda * Ka.T)
    sys.stdout.write(' ' * 10 + '\r')
    sys.stdout.flush()
    sys.stdout.write("Iteration:" + str(iter) + '/' + str(500) +'\r')
    sys.stdout.flush()
    #check training + validation performance
    lossTrain.append(float(lossCompute(featuresTrain, featuresTrain, labelsTrain, w, lenda, KMatrix)))
    lossValidation1.append(float(lossCompute1(featuresTrain, featuresValidation, labelsValidation, w, lenda, KMatrix)))

iterations = np.linspace(1, 500, 500)
plt.subplot(111)
plt.plot(iterations, np.array(lossTrain), label = "Regularization training risk")
plt.plot(iterations, np.array(lossValidation1), label = "Misclassification validation risk")
plt.axis([1,500,0,max(lossTrain)])
plt.title('empirical risk')
plt.grid(True)
plt.legend()
plt.show()
    
print("best learning rate u = %f lenda = %f" % (miu, lenda))
#check training + validation performance
lossTrainBest = lossCompute(featuresTrain, featuresTrain, labelsTrain, w, lenda, KMatrix)
lossValidationBest1 = lossCompute1(featuresTrain, featuresValidation, labelsValidation, w, lenda, KMatrix)
print("Regularization training error: %f  Misclassification validation error: %f" % (lossTrainBest, lossValidationBest1))
lossTestBest1 = lossCompute1(featuresTrain, featuresTest, labelsTest, w, lenda, KMatrix)
print("Misclassification testing error: %f" % (lossTestBest1))