import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from warnings import simplefilter
simplefilter("ignore")

class SNN: # Simple Neural Network :)

    def __init__(
        self,
        input_layer : int,
        output_layer : int,
        hidden_layer : int = None,
        learning_rate : int = None,
        epochs : int = None,
        random_state : int = None,
        verbose : bool = False,
        error_type : str = None,
        verbose_stepsize : int = None):

        self.inodes = input_layer
        self.onodes = output_layer
        self.hnodes = hidden_layer if hidden_layer else 10
        self.alpha = learning_rate if learning_rate else 1.0
        self.n_epochs = epochs if epochs else 1.0
        self.random_state = random_state if random_state else 0
        self.verbose_mode = verbose
        self.error_type = error_type if error_type else "MSE"
        self.verbose_stepsize = verbose_stepsize if verbose_stepsize else 1
        
        self.__activation = lambda t: 1.0 / (1.0 + np.exp(-t))

        np.random.seed(self.random_state)

        self.__iweights = np.random.rand(self.inodes,self.hnodes)
        self.__oweights = np.random.rand(self.hnodes,self.onodes)

        self.__flag = False
        self.total_error = None

    def __check_arguments(self,x,y):

        try:
            if np.shape(x) == (np.shape(x)[0],self.inodes): pass
            else: x = x.reshape((np.shape(x)[0],self.inodes))
            if np.shape(y) == (np.shape(y)[0],1): pass
            else: y = y.reshape((np.shape(y)[0],1))
        except:
            raise ValueError("'X' and 'y' have incorrect shapes. Reshape according to input/output nodes:\nX: 2darray(items,features), y: 2darray(labels)")

    def __fit_aux(self,X : np.array,y : np.array) -> np.array:

        sum_h = np.dot(self.__iweights.T,X)
        out_h = self.__activation(sum_h)

        sum_o = np.dot(self.__oweights.T,out_h)
        out_o = self.__activation(sum_o)

        error_o = y - out_o
        error_h = np.dot(self.__oweights,error_o)

        activation_prime = lambda t: self.__activation(t) * (1.0 - self.__activation(t))

        delta_h = np.multiply(error_h,activation_prime(out_h))
        delta_o = np.multiply(error_o,activation_prime(out_o))
        self.__iweights += self.alpha * np.dot(delta_h,X.T).T
        self.__oweights += self.alpha * np.dot(delta_o,out_h.T).T

        return error_o

    def fit(self,X_train : np.array,y_train : np.array):

        self.__flag = True

        if not ((type(X_train)=="numpy.ndarray") and (type(y_train)=="numpy.ndarray")):
            X_train = np.array(X_train)
            y_train = np.array(y_train)

        self.__check_arguments(X_train,y_train)

        for progress in range(1,self.n_epochs + 1):

            aggregate_error = np.zeros((np.shape(X_train)[0],self.onodes,1))
            
            for i,x_i in enumerate(X_train):
                x_i = x_i.reshape((np.shape(x_i)[0],1))
                y_i = np.zeros((self.onodes,1))
                y_i[y_train[i]] = 1.0
                error = self.__fit_aux(x_i,y_i)
                aggregate_error[i,:,:] = error

            if self.error_type == "MSE":
                self.total_error = 1.0/np.shape(X_train)[0] * np.sum(aggregate_error **2)
            elif self.error_type == "ESS":
                self.total_error = 1.0/2.0 * np.sum(aggregate_error **2)
            elif self.error_type == "RMS":
                self.total_error = np.sqrt(1.0/np.shape(X_train)[0] * np.sum(aggregate_error **2))

            if self.verbose_mode and (np.mod(progress,self.verbose_stepsize)==0):
                print(f"Epoch {progress}/{self.n_epochs}:\tCalculating...\tError: {self.total_error}")

    def __predict_aux(self,X : np.array) -> np.array:

        sum_h = np.dot(self.__iweights.T,X)
        out_h = self.__activation(sum_h)

        sum_o = np.dot(self.__oweights.T,out_h)
        out_o = self.__activation(sum_o)

        return out_o

    def predict(self,X_test : np.array) -> list:

        predictions = []

        if not type(X_test)=="numpy.ndarray": X_test = np.array(X_test)

        for i,x in enumerate(X_test):
            x = x.reshape((np.shape(x)[0],1))
            y = self.__predict_aux(x)
            predictions.append(np.argmax(y))

        return predictions

    def score(self,X_test : np.array,y_test : np.array) -> float:

        if not ((type(X_test)=="numpy.ndarray") and (type(y_test)=="numpy.ndarray")):
            X_test = np.array(X_test)
            y_test = np.array(y_test)

        s = []

        for i,x in enumerate(X_test):

            x = x.reshape((np.shape(x)[0],1))
            y = self.__predict_aux(x)
            if (np.argmax(y) == y_test[i]): s.append(1)
            else: s.append(0)

        return np.sum(s)/len(s)

    def _diagnostics(self):

        if self.__flag == True: pass
        else: raise SyntaxError("Model has not yet been fitted!")

        plt.figure()
        plt.hist(self.__oweights,alpha=0.8)
        plt.title("Output weights' distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Values")
        plt.show()

        print("-"*40)

        print(f"Epochs performed: {self.n_epochs}")
        print(f"Final error ('{self.error_type}'): {self.total_error}")
        print(f"Input nodes used: {self.inodes}")
        print(f"Hidden nodes used: {self.hnodes}")
        print(f"Output nodes used: {self.onodes}")
        print("Activation function: 'sigmoid'")
        print(f"Algorithm's learning rate: {self.alpha}")

        print("-"*40)

    def _get_input_weights(self):

        return self.__iweights

    def _get_output_weights(self):

        return self.__oweights

    def _get_loss(self):

        return self.total_error

    def _get_learning_rate(self):

        return self.alpha

    def _get_layers_input(self):

        return {"I":self.inodes,"H":self.hnodes,"O":self.onodes}