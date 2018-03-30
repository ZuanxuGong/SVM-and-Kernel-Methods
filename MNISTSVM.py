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
            features[count].append(int(x) / 255.0);
        count += 1;
    return features,labels

def lossCompute(features, labels, w, lenda): #loss with regularization
    count = 0
    loss = 0
    for x in features:
        y = labels[count]
        ywx = (y * np.dot(w,x))
        loss += max(0, 1 - ywx)
        count += 1;
    return (loss / len(labels) + lenda / 2 * np.dot(w,w));

def lossCompute1(features, labels, w, lenda): #misclassification accuracy
    count = 0
    for i in range(len(features)):
        x = features[i]
        y = labels[i]
        predict = np.dot(w,x);
        if(predict >= 0 and y == 1):
            count += 1;
        elif(predict < 0 and y == -1):
            count += 1;
    return  1 - count / float(len(labels));

#initial weight
w0 = []
for i in range(784):
    w0.append(0) 

#get training + validation + testing data (features + labels)
[featuresTrain, labelsTrain] = generateData(file_features[0], file_labels[0]);
[featuresValidation, labelsValidation] = generateData(file_features[1], file_labels[1]);
[featuresTest, labelsTest] = generateData(file_features[2], file_labels[2]);

mius = [0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3]
lendas = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

#cross-validation
for miu in mius:
    for lenda in lendas:
        w = w0      
        for iter in range(500):
            fwdao = 0
            count = 0;
            for x in featuresTrain:
                y = labelsTrain[count]
                ywx = (y * np.dot(w,x))
                if ywx < 1:
                    fwdao += - y * np.array(x)
                count += 1;
            w = w - miu * (fwdao / 6107 + lenda * np.array(w));
            sys.stdout.write(' ' * 10 + '\r')
            sys.stdout.flush()
            sys.stdout.write("Iteration:" + str(iter) + '/' + str(500) +'\r')
            sys.stdout.flush()
        print("learning rate u = %f lenda = %f" % (miu, lenda))
        #check training + validation performance
        lossTrain = lossCompute(featuresTrain, labelsTrain, w, lenda)
        lossValidation1 = lossCompute1(featuresValidation, labelsValidation, w, lenda)
        print("Regularization training loss: %f  Misclassification validation loss: %f" % (lossTrain, lossValidation1))

#best miu = 0.3 and lenda = 0.0001
miu = 0.3
lenda = 0.0001
w = w0
lossTrain = []
lossValidation1 = []
for iter in range(500):
    fwdao = 0
    count = 0;
    for x in featuresTrain:
        y = labelsTrain[count]
        ywx = (y * np.dot(w,x))
        if ywx < 1:
            fwdao += - y * np.array(x)
        count += 1;
    w = w - miu * (fwdao / 6107 + lenda * np.array(w));
    sys.stdout.write(' ' * 10 + '\r')
    sys.stdout.flush()
    sys.stdout.write("Iteration:" + str(iter) + '/' + str(500) +'\r')
    sys.stdout.flush()
    #check training + validation performance
    lossTrain.append(lossCompute(featuresTrain, labelsTrain, w, lenda))
    lossValidation1.append(lossCompute1(featuresValidation, labelsValidation, w, lenda))

iterations = np.linspace(1, 500, 500)
plt.subplot(111)
plt.plot(iterations, np.array(lossTrain), label = "Regularization training risk")
plt.plot(iterations, np.array(lossValidation1), label = "Misclassification validation risk")
plt.axis([1,500,0,1])
plt.title('empirical risk')
plt.grid(True)
plt.legend()
plt.show()
    
print("best learning rate u = %f lenda = %f" % (miu, lenda))
#check training + validation performance
lossTrainBest = lossCompute(featuresTrain, labelsTrain, w, lenda)
lossValidationBest1 = lossCompute1(featuresValidation, labelsValidation, w, lenda)
print("Regularization training error: %f  Misclassification validation error: %f" % (lossTrainBest, lossValidationBest1))
lossTestBest1 = lossCompute1(featuresTest, labelsTest, w, lenda)
print("Misclassification testing error: %f" % (lossTestBest1))
