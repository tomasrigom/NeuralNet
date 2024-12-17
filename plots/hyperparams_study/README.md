
In this folder the results of a study on the hypermarameters of the neural network is shown, namely:

    - The learning rate 'eta'
    - The regularization parameter 'lambda'
    - The momentum co-efficient 'mu'
    - The number of examples per mini batch (in the context of stochastic gradient descent), 'batch'
    - The regularization type 'reg'
    - The network architecture 'net'

In the plots within this folder, all in '.png' format, the cost and accuracy of the model on the evaluation data as a function of epoch for different values of the aforementioned parameters is shown. To this end, each plot is named as the corresponding parameter under study, and includes text indicating the best accuracy obtained, the value of the studied parameter for which this value was achieved, and the corresponding epoch. A couple important notes:

The 'etas' plot shows that both constant and variable values (i.e. functions) have been used for the training of the model. Three different functions have been used:

    - f_1(acc): This is a function of the accuracy (acc) of the model and is defined as:

        f_1(acc) = 0.1 if acc >= 0.1 else 0.5 * acc

    - f_2(acc): This is a function of the accuracy as well, defined as:

        f_2(acc) = 0.1 if acc >= 0.1 else 0.2 * acc

    - f_3(epoch) is instead a function of the epoch and is defined as:

        f_3(epoch) = 0.1 if epoch <= 60 else 0.1 * np.exp(1 - x / 60)

I thought about these basic functions while experimenting with the code. It turns out that the function dependent on the epoch works best, but this might be attributed to maintaining a large learning rate until an advanced stage of the training.

A second important thing to note is the structure of the 'nets' variable. This is a list showing each of the neurons per layer, e.g. [784, 60, 10] indicates a neural network with an input layer consisting of 784 neurons (corresponding to a flattened version of the MNIST images), a hidden layer consisting of 60 neurons, and an output layer consisting of 10 neurons (10 possible digits).

When modifying some hyperparameters, the others are set to their default values, as found in neuralnet.py. The training stops when 15 epochs have passed and the accuracy has not improved by 0.01.