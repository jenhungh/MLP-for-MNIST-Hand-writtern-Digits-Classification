# Do not import any additional 3rd party external libraries
import numpy as np
import os
import matplotlib.pyplot as plt


class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example (do not change)

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # hint: save the useful data for back propagation
        self.state = x
        return 1 / (1 + np.exp(-x))

    def derivative(self):
        sigmoid = self.forward(self.state)
        return sigmoid * (1 - sigmoid) 


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = x
        return np.tanh(x)

    def derivative(self):
        return 1 - np.power(self.forward(self.state), 2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        return np.maximum(x, np.zeros_like(x))

    def derivative(self):
        return (self.state > 0) * 1.0


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        # you can add variables if needed

    # Softmax function 
    def softmax(self, x):
        size = x.shape[0]
        softmax = np.exp(x) / (np.sum(np.exp(x), axis = 1).reshape(size,1))
        return softmax    
    
    # Cross-entropy loss
    def forward(self, x, y):
        loss = -np.sum(y * np.log(self.softmax(x)), axis = 1)
        return np.mean(loss)

    # Derivative of SoftmaxCrossEntropy 
    def derivative(self, x, y):
        size = y.shape[0]
        gradient = self.softmax(x) - y
        return gradient / size


# randomly intialize the weight matrix with dimension d0 x d1 via Normal distribution
def random_normal_weight_init(d0, d1):
    w0 = np.random.normal(loc = 0.0, scale = 1.0, size = (d0, d1))
    return w0


# initialize a d-dimensional bias vector with all zeros
def zeros_bias_init(d):
    b0 = np.zeros((d, 1)) 
    return b0


class MLP(object):

    """
    A simple multilayer perceptron
    (feel free to add class functions if needed)
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr):

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes
        self.nn_dim = [input_size] + hiddens + [output_size]
        # list containing Weight matrices of each layer, each should be a np.array
        self.W = [weight_init_fn(self.nn_dim[i], self.nn_dim[i+1]) for i in range(self.nlayers)]
        # list containing derivative of Weight matrices of each layer, each should be a np.array
        self.dW = [np.zeros_like(weight) for weight in self.W]
        # list containing bias vector of each layer, each should be a np.array
        self.b = [bias_init_fn(self.nn_dim[i+1]) for i in range(self.nlayers)]
        # list containing derivative of bias vector of each layer, each should be a np.array
        self.db = [np.zeros_like(bias) for bias in self.b]

        # You can add more variables if needed
        # list containing output of each layer, each should be a np.array
        self.layer_output = []


    def forward(self, x):
        self.x = x
        inputs = x.T
        for layer in range(0, self.nlayers-1):
            # f = Wx + b
            f = np.dot(self.W[layer].T, inputs) + self.b[layer]
            # Activation function
            act = self.activations[layer](f)
            # Store the layer_output_history
            self.layer_output.append(act.T) 
            inputs = act
        # Output layer
        result = np.dot(self.W[self.nlayers-1].T, inputs) + self.b[self.nlayers-1]
        self.outputs = result.T     # Dimension: 8 * 10 
        return NotImplementedError

    def zero_grads(self):
        # set dW and db to be zero
        self.dW = [np.zeros_like(weight) for weight in self.W]
        self.db = [np.zeros_like(bias) for bias in self.b]
        return NotImplementedError

    def step(self):     
        # update the W and b of each layer
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.dW[i]
            self.b[i] -= self.lr * self.db[i]
        return NotImplementedError

    def backward(self, labels):
        # Calculate dW and db only under training mode
        if self.train_mode:
            # Zero the gradient
            self.zero_grads()
            
            # Calculate dW and db
            # Compute the gradient of SoftmaxCrossEntropy dE/df3
            dEdf = self.criterion.derivative(self.outputs, labels)
            # df3/dW3
            dfdWout = self.layer_output[self.nlayers-2]
            # dE/dW3 = dE/df3 * df3/dW3
            self.dW[self.nlayers-1] = np.dot(dEdf.T, dfdWout).T
            # dE/db3 = dE/df3 * df3/db3 = dE/df3 * 1 
            self.db[self.nlayers-1] = (np.sum(dEdf, axis = 0)/dEdf.shape[0]).reshape(dEdf.shape[1], 1)

            # Back-propagation on the hidden layers
            for layer in range(self.nlayers-2, 0, -1):
                # df3/db
                dfdact = self.W[layer+1]
                # db/df2
                dactdf = self.activations[layer].derivative()
                # Update dEdf : dE/df2 = dE/df3 * df3/db * db/df2  
                dEdf = (np.dot(dEdf, dfdact.T).T * dactdf).T
                # df2/dW2
                dfdW = self.layer_output[layer-1]
                # dE/dW2 = dE/df2 * df2/dW2
                self.dW[layer] = np.dot(dEdf.T, dfdW).T
                # dE/db2 = dE/df2 * df2/db2 = dE/df2 * 1
                self.db[layer] = (np.sum(dEdf, axis = 0)/dEdf.shape[0]).reshape(dEdf.shape[1], 1)

            # df2/da
            dfdact = self.W[1]
            # da/df1
            dactdf = self.activations[0].derivative()
            # Update dEdf : dE/df1 = dE/df2 * df2/da * da/df1  
            dEdf = (np.dot(dEdf, dfdact.T).T * dactdf).T
            # df1/dW1 = x
            dfdWin = self.x
            # dE/dW1 = dE/df1 * df1/dW1
            self.dW[0] = np.dot(dEdf.T, dfdWin).T
            # dE/db1 = dE/df1 * df1/db1 = dE/df1 * 1
            self.db[0] = (np.sum(dEdf, axis = 0)/dEdf.shape[0]).reshape(dEdf.shape[1], 1)
        return NotImplementedError 

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        # training mode
        self.train_mode = True

    def eval(self):
        # evaluation mode
        self.train_mode = False

    def get_loss(self, labels):
        # Return the current loss value given labels
        # Calculate SoftmaxCrossEntropy
        loss = 0
        loss = self.criterion(self.outputs, labels)
        return loss

    def get_error(self, labels):
        # Return the number of incorrect preidctions gievn labels
        error = 0
        result =  self.criterion.softmax(self.outputs)
        pred_labels = np.argmax(result, axis = 1)
        true_labels = labels.argmax(axis = 1)
        for i in range(len(true_labels)):
            if (pred_labels[i] != true_labels[i]):
                error += 1
        return error

    def save_model(self, path='model.npz'):
        # save the parameters of MLP (do not change)
        np.savez(path, self.W, self.b)


# Don't change this function
def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    for e in range(nepochs):
        print("epoch: ", e)
        train_loss = 0
        train_error = 0
        val_loss = 0
        val_error = 0
        num_train = len(trainx)
        num_val = len(valx)

        for b in range(0, num_train, batch_size):
            mlp.train()
            mlp(trainx[b:b+batch_size])
            mlp.backward(trainy[b:b+batch_size])
            mlp.step()
            train_loss += mlp.get_loss(trainy[b:b+batch_size])
            train_error += mlp.get_error(trainy[b:b+batch_size])
        training_losses += [train_loss/num_train]
        training_errors += [train_error/num_train]
        print("training loss: ", train_loss/num_train)
        print("training error: ", train_error/num_train)

        for b in range(0, num_val, batch_size):
            mlp.eval()
            mlp(valx[b:b+batch_size])
            val_loss += mlp.get_loss(valy[b:b+batch_size])
            val_error += mlp.get_error(valy[b:b+batch_size])
        validation_losses += [val_loss/num_val]
        validation_errors += [val_error/num_val]
        print("validation loss: ", val_loss/num_val)
        print("validation error: ", val_error/num_val)

    test_loss = 0
    test_error = 0
    num_test = len(testx)
    for b in range(0, num_test, batch_size):
        mlp.eval()
        mlp(testx[b:b+batch_size])
        test_loss += mlp.get_loss(testy[b:b+batch_size])
        test_error += mlp.get_error(testy[b:b+batch_size])
    test_loss /= num_test
    test_error /= num_test
    print("test loss: ", test_loss)
    print("test error: ", test_error)

    return (training_losses, training_errors, validation_losses, validation_errors)


# get one hot key encoding of the label (no need to change this function)
def get_one_hot(in_array, one_hot_dim):
    dim = in_array.shape[0]
    out_array = np.zeros((dim, one_hot_dim))
    for i in range(dim):
        idx = int(in_array[i])
        out_array[i, idx] = 1
    return out_array


def main():
    # load the mnist dataset from csv files
    image_size = 28 # width and length of mnist image
    num_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    train_data = np.loadtxt("./mnist/mnist_train.csv", delimiter=",")
    test_data = np.loadtxt("./mnist/mnist_test.csv", delimiter=",") 

    # rescale image from 0-255 to 0-1
    fac = 1.0 / 255
    train_imgs = np.asfarray(train_data[:50000, 1:]) * fac
    val_imgs = np.asfarray(train_data[50000:, 1:]) * fac
    test_imgs = np.asfarray(test_data[:, 1:]) * fac
    train_labels = np.asfarray(train_data[:50000, :1])
    val_labels = np.asfarray(train_data[50000:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # convert labels to one-hot-key encoding
    train_labels = get_one_hot(train_labels, num_labels)
    val_labels = get_one_hot(val_labels, num_labels)
    test_labels = get_one_hot(test_labels, num_labels)

    print(train_imgs.shape)
    print(val_imgs.shape)
    print(test_imgs.shape)
    print(train_labels.shape)
    print(val_labels.shape)
    print(test_labels.shape)

    dataset = [
        [train_imgs, train_labels],
        [val_imgs, val_labels],
        [test_imgs, test_labels]
    ]

    # These are only examples of parameters you can start with
    # you can tune these parameters to improve the performance of your MLP
    # this is the only part you need to change in main() function
    hiddens = [128, 64]
    activations = [Sigmoid(), Sigmoid()]
    lr = 0.01
    num_epochs = 100
    batch_size = 8

    # build your MLP model
    mlp = MLP(
        input_size=image_pixels, 
        output_size=num_labels, 
        hiddens=hiddens, 
        activations=activations, 
        weight_init_fn=random_normal_weight_init, 
        bias_init_fn=zeros_bias_init, 
        criterion=SoftmaxCrossEntropy(), 
        lr=lr
    )

    # train the neural network
    losses = get_training_stats(mlp, dataset, num_epochs, batch_size)

    # save the parameters
    mlp.save_model()

    # visualize the training and validation loss with epochs
    training_losses, training_errors, validation_losses, validation_errors = losses

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(training_losses, color='blue', label="training")
    ax1.plot(validation_losses, color='red', label='validation')
    ax1.set_title('Loss during training')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(training_errors, color='blue', label="training")
    ax2.plot(validation_errors, color='red', label="validation")
    ax2.set_title('Error during training')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('error')
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()