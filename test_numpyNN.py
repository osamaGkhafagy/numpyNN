import numpy as np
import pandas as pd
from numpyNN import Net, Layer, Loss, utility
import matplotlib.pyplot as plt

def categorical_encoding(cat_data):
    """ 1: Iris-setosa
        0: Iris-versicolor
    """
    Y = []
    for i, specie in enumerate(cat_data[0]):
        Y.append(0) if specie=='Iris-setosa' else Y.append(1)
    return np.array(Y)

def prepare_data(file_name):
    df = pd.read_csv(file_name)
    # excluding the 'Iris-virginica' specie for a proper binary classification
    df1 = df[:][0:100]
    df2 = df[:][100:]
    df1 = df1.sample(frac=1)
    df2 = df2.sample(frac=1)
    X = df1[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    X_unique = df2[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    X = np.array(X).T
    X_unique = np.array(X_unique).T
    Y = np.array(df1['species']).reshape(100, 1).T
    Y = categorical_encoding(Y).reshape(1, 100)
    return X, Y, X_unique

def train_test_sets(X, Y):
    train_ratio = 9/10
    m = X.shape[1]
    limit = int(train_ratio * m)
    X_train = X[:, :limit]
    X_test = X[:, limit:]
    Y_train = Y[:, :limit]
    Y_test = Y[:, limit:]
    return X_train, Y_train, X_test, Y_test


X, Y, X_unique = prepare_data('Iris_Data.csv')
# check X and Y's shape
print('X.shape', X.shape)
print('Y.shape', Y.shape)
X_train, Y_train, X_test, Y_test = train_test_sets(X, Y)
# check shapes of train/test sets
print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)
print('Y_train.shape', Y_train.shape)
print('Y_test.shape', Y_test.shape)

m = X_train[1]
nx = X_train.shape[0]
nn = Net([Layer(nx, 8, 'tanh'),
        Layer(8, 16, 'tanh'),
        Layer(16, 8, 'tanh'),
        Layer(8, 1, 'sigmoid')],
        Loss(),
        optimizer="Adam")

nn.train(X_train, Y_train, 0.01, 1000)

train_acc, test_acc = utility.accuracy(nn, X_train, Y_train, X_test, Y_test)
print('train accuracy: {:.2f}%'.format(train_acc))
print('test accuracy: {:.2f}%'.format(test_acc))
