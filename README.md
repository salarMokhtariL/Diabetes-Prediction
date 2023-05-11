# Predicting Diabetes using a Neural Network Model Trained on Clinical Data
## An Evaluation of Accuracy, Precision, Recall, and F1 Score

> By Salar Mokhtari Laleh

![Pink and Peach Technology LinkedIn Banner (2)](https://github.com/salarMokhtariL/Diabetes-prediction/assets/75142232/6f8f0b5a-41b8-4518-b8df-eec991df8fce)

This code uses a neural network model trained on clinical data to predict diabetes. The model is evaluated using accuracy, precision, recall, and F1 score.
## Dependencies

This code requires the following dependencies:

* torch
* numpy
* pandas
* scikit-learn


You can install them using pip:

```
pip install torch numpy pandas scikit-learn
```

## Usage
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

## Neural Network Model
The neural network model used in this code has the following architecture:

```
Net(
  (fc1): Linear(in_features=8, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=16, bias=True)
  (fc3): Linear(in_features=16, out_features=1, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (bn1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```
