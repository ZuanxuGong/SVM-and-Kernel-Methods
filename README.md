# SVM-and-Kernel-Methods
Write and train a SVM and a Kernel SVM for Hand-written digit classification (MNIST).

## Data
Data contains 6 files: <br>
1) trainingData.txt <br> 
2) trainingLabels.txt <br> 
3) validationData.txt <br> 
4) validationLabels.txt <br> 
5) testData.txt <br> 
6) testLabels.txt <br>

## Data format
### “*Data.txt” files contain features.
Each row is comma-separated 784 integers which are features of a single sample. <br>
The number of rows (=number of samples) for training, validation, and test data are 6107, 6107 and 2037 respectively. <br>
### “*Labels.txt” files contain +1 or -1 labels. 
Each row corresponds to the corresponding row of the feature file. 
 
## SVM
### Steps
- Implement the Gradient Descent algorithm. 
- Assume initial w is all zeros, the total number of iterations T is 500, and learning rate 𝜂 is a constant.
- Training with different values of 𝜂 = {0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3}. λ = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3].
- Training error uses the loss with regularization and validation error uses misclassification accuracy (1 – accuracy = 1 - # of correctly classified points / # of all points)
- The best value 𝜂 = 0.3 and λ=0.0001.
- The test error is 0.039273.

## SVM - Kernel
### Steps
- Implement the Gradient Descent algorithm. 
- Assume initial w is all zeros, the total number of iterations T is 500, and learning rate 𝜂 is a constant.
- Cross-validation. Training with different values of 𝜂 = {0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01}. λ = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003].
- Training error uses the loss with regularization and validation error uses misclassification accuracy (1 – accuracy = 1 - # of correctly classified points / # of all points);
- The best value 𝜂 = 0.003 and λ=0.00003.
- The test error is 0.059401.

PS: remember to change the path.
