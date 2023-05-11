# Predicting Diabetes using a Neural Network Model Trained on Clinical Data
## An Evaluation of Accuracy, Precision, Recall, and F1 Score

> By Salar Mokhtari Laleh

![Pink and Peach Technology LinkedIn Banner (2)](https://github.com/salarMokhtariL/Diabetes-prediction/assets/75142232/6f8f0b5a-41b8-4518-b8df-eec991df8fce)

# Introduction
This project aims to predict diabetes in individuals using a neural network model. The dataset used for training and testing the model is publicly available and contains information about eight medical predictors (e.g., glucose level, age, and blood pressure) and one target variable indicating whether or not the individual has diabetes.

# Dependencies

This code requires the following dependencies:

* torch
* numpy
* pandas
* scikit-learn


You can install them using pip:

```
pip install torch numpy pandas scikit-learn
```

# Usage
Clone the repository:

```
git clone https://github.com/salarMokhtariL/Diabetes-prediction.git
```
Navigate to the code directory:
```
cd Diabetes-prediction
```

Run the Jupyter Notebook file:

```
jupyter notebook diabetes_prediction.ipynb
```
Follow the instructions in the notebook to execute the code.


# Methods
## Data Preprocessing
The first step in building the model was to load the dataset and preprocess it for training and testing. The dataset was loaded from a remote source and stored in a pandas DataFrame. The input features (predictors) and target variable were then separated into different numpy arrays. To split the dataset into training and testing sets, we used the `train_test_split()` function from scikit-learn, which randomly divides the data into two sets with a specified test size and random state. Finally, the numpy arrays were converted to PyTorch tensors, which are required for training the neural network.

## Neural Network Model
The neural network model used for this task consists of three fully connected layers with 8, 32, and 16 neurons, respectively. Each layer is followed by a batch normalization layer and a ReLU activation function. Additionally, the second and third layers are followed by a dropout layer with a dropout probability of 0.2. The output layer has a single neuron with a sigmoid activation function, which produces a probability value that indicates the likelihood of a patient having diabetes.

The model is trained using the binary cross-entropy loss function and the Adam optimization algorithm.

The neural network model is defined using the following formula:

$y= \sigma(W_3.ReLU(BN_2(W_2.ReLU(BN_1(W_1.x))))$

## Loss Function and Optimization
We used the binary cross-entropy loss function, also known as the log loss, to calculate the error between the predicted and actual values. The Adam optimizer was used to minimize this loss function and update the model parameters during training.

## Training and Evaluation
We trained the neural network model for 1000 epochs on the training set using mini-batch gradient descent. We printed the loss every 100 epochs to monitor the training progress. After training the model, we evaluated its performance on the testing set using four different evaluation metrics: accuracy, precision, recall, and F1 score. These metrics were calculated using scikit-learn's `accuracy_score()`, `precision_score()`, `recall_score()`, and `f1_score()` functions, respectively.

# Results
The trained model achieved an accuracy of 0.7792, a precision of 0.6818, a recall of 0.6571, and an F1 score of 0.6693 on the testing set. These results indicate that the model has a moderate predictive performance for diabetes in individuals.

# Conclusion
In this project, we developed a neural network model to predict diabetes in individuals using eight medical predictors. The model achieved moderate performance on the testing set, indicating that the predictors used in this study have some predictive value for diabetes. Further research could be conducted to explore the use of other predictors or models to improve the predictive performance.


