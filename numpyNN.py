import numpy as np
import matplotlib.pyplot as plt
import time

EPSILON = 1e-6

class utility:
    def accuracy(net, X_train, Y_train, X_test, Y_test):
        m_train = X_train.shape[1]
        m_test = X_test.shape[1]
        AL, _ = net.predict(X_train, Y_train)
        print('Some of test predictions:\n')
        print("Actual {}, predicted {}".format(Y_train[:, 2:7], AL[:, 2:7]))
        train_acc = (1 - (1/m_train)*np.sum(np.abs(Y_train - AL), axis=1, keepdims=True)) * 100
        train_acc = np.squeeze(train_acc)
        AL, _ = net.predict(X_test, Y_test)
        test_acc = (1 - (1/m_test)*np.sum(np.abs(Y_test - AL), axis=1, keepdims=True)) * 100
        test_acc = np.squeeze(test_acc)
        return train_acc, test_acc

class Net:
    def __init__(self, layers_list, loss_layer, optimizer="GD"):
        self.layers = layers_list
        self.loss = loss_layer
        self.optimizer = optimizer
        # optimization parameters
        self.opt_params = self.init_opt_params()
        # for debugging
        self.print_opt_params()

    # optimization methods
    def init_opt_params(self):
        opt_params = []
        for i, layer in enumerate(self.layers):
            Vdw = np.zeros((layer.W.shape))
            Vdb = np.zeros((layer.b.shape))
            Sdw = np.zeros((layer.W.shape))
            Sdb = np.zeros((layer.b.shape))
            opt_params.append([Vdw, Vdb, Sdw, Sdb])
        return opt_params

    def print_opt_params(self):
        for i in range(len(self.layers)):
            print("layer ", i+1)
            print('Vdw.shape = ', self.opt_params[i][0].shape)
            print('Vdb.shape = ', self.opt_params[i][1].shape)
            print('Sdw.shape = ', self.opt_params[i][2].shape)
            print('Sdb.shape = ', self.opt_params[i][3].shape)

    def weighted_avgs(self, dW, db, Vdw, Vdb, beta):
        """
        returns the weighted average over dw where beta is the factor that determines the interval of dw over which
        the average is calculated
        """
        Vdw = (1-beta)*dW + beta*Vdw
        Vdb = (1-beta)*db + beta*Vdb
        return Vdw, Vdb

    def RMSprop(self, dW, db, Sdw, Sdb, beta):
        """
        returns the Squared average over sw where beta is the factor that determines the interval of dw over which
        the average is calculated
        """
        Sdw = (1-beta)*dW**2 + beta*Sdw
        Sdb = (1-beta)*db**2 + beta*Sdb
        return Sdw, Sdb

    def Adam(self, dW, db, layer_opt_params, beta1, beta2):
        """"""
        Vdw = layer_opt_params[0]
        Vdb = layer_opt_params[1]
        Sdw = layer_opt_params[2]
        Sdb = layer_opt_params[3]
        Vdw, Vdb = self.weighted_avgs(dW, db, Vdw, Vdb, beta1)
        Sdw, Sdb = self.RMSprop(dW, db, Sdw, Sdb, beta2)
        return Vdw, Vdb, Sdw, Sdb

    def optimize(self, dW, db, layer_opt_params, beta1, beta2):
        """"""
        Vdw = layer_opt_params[0]
        Vdb = layer_opt_params[1]
        Sdw = layer_opt_params[2]
        Sdb = layer_opt_params[3]
        if self.optimizer == 'GD_with_momentum':
            Vdw, Vdb = self.weighted_avgs(dW, db, Vdw, Vdb, beta1)
            W_grad = Vdw
            b_grad = Vdb
        elif self.optimizer == 'RMSprop':
            Sdw, Sdb = self.RMSprop(dW, db, Sdw, Sdb, beta2)
            W_grad = dW/(np.sqrt(Sdw)+EPSILON)
            b_grad = db/(np.sqrt(Sdb)+EPSILON)
        elif self.optimizer == 'Adam':
            Vdw, Vdb, Sdw, Sdb = self.Adam(dW, db, layer_opt_params, beta1, beta2)
            W_grad = Vdw/(np.sqrt(Sdw)+EPSILON)
            b_grad = Vdb/(np.sqrt(Sdb)+EPSILON)
        elif self.optimizer == "GD":
            W_grad = dW
            b_grad = db
        layer_opt_params = [Vdw, Vdb, Sdw, Sdb]

        return W_grad, b_grad, layer_opt_params

    def train(self, X, Y, learning_rate, num_iterations, show_cost=True):
        m = X.shape[1]
        costs = []
        for k in range(num_iterations):
            ## forward propagation
            AL = X
            # forward propagation across layers
            for i, layer in enumerate(self.layers):
                AL = layer.forward_prop(AL)
            # cost calculation
            cost = self.loss.compute_cost(AL, Y)
            costs.append(cost)
            ## backpropagation
            dA_prev = self.loss.compute_loss_grad(AL, Y)
            for j in reversed(range(len(self.layers))):
                layer = self.layers[j]
                dA_prev, dW, db = layer.backprop(dA_prev)
                W_grad, b_grad, layer_opt_params = self.optimize(dW, db, self.opt_params[j], 0.9, 0.999)
                layer.update_params((W_grad, b_grad), learning_rate, m)
                # update optimization params
                self.opt_params[j] = layer_opt_params
            if show_cost:
                if k % 100 == 0:
                    print("Cost at iteration {}: {:.5f}".format(k, cost))
        if show_cost:
            plt.plot(costs)
            plt.show()

    def predict(self, X, Y):
        ## forward propagation
        AL = X
        # forward propagation across layers
        for layer in self.layers:
            AL = layer.forward_prop(AL)
        # cost calculation
        cost = self.loss.compute_cost(AL, Y)
        return AL, cost

class Layer:
    def __init__(self, in_size, out_size, activation):
        self.out_size = out_size
        self.in_size = in_size
        self.activation = activation
        self.W, self.b = self.init_params()

    def init_params(self):
        if self.activation == 'relu':
            W = np.random.rand(self.out_size, self.in_size) * np.sqrt(2/self.in_size)
        else:
            W = np.random.rand(self.out_size, self.in_size) * np.sqrt(1/self.in_size)
        b = np.zeros((self.out_size, 1))

        return W, b

    def relu(self, Z):
        A = [[max(0, i) for i in row] for row in Z]
        g_prime = 1. * (Z > 0)
        return A, g_prime

    def tanh(self, Z):
        A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        g_prime = 1 - A**2
        return A, g_prime

    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        g_prime = A * (1-A)
        return A, g_prime

    def forward_prop(self, input):
        """"""
        self.A_prev = input
        self.m = input.shape[1]
        Z = np.dot(self.W, self.A_prev) + self.b
        if self.activation == 'relu':
            A, g_prime = self.relu(Z)
        elif self.activation == 'tanh':
            A, g_prime = self.tanh(Z)
        elif self.activation == 'sigmoid':
            A, g_prime = self.sigmoid(Z)
        self.A, self.g_prime = np.array(A), np.array(g_prime)
        return self.A

    def backprop(self, dA):
        """"""
        dZ = dA * self.g_prime
        dA_prev = np.dot(self.W.T, dZ)
        dW = (1/self.m) * np.dot(dZ, self.A_prev.T)
        db = (1/self.m) * np.sum(dZ, axis=1, keepdims=True)
        return dA_prev, dW, db

    def update_params(self, params, lr, m, lambd=5):
        dW = params[0]
        db = params[1]
        self.W = self.W - lr*(dW + (lambd/m)*self.W)
        self.b = self.b - lr * db

class Loss:
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        L = -(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        cost = (1/m) * np.sum(L, axis=1, keepdims=True)
        cost = np.squeeze(cost)
        return cost

    def compute_loss_grad(self, AL, Y):
        dJ_by_dA = -(Y/AL) + ((1-Y)/(1-AL))
        return dJ_by_dA
