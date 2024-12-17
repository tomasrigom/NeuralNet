# Training of three different neural networks using the optimal parameters found in ../hyperparams_study

For this study, i train and test the neural network using the same architecture and hyperparameters, but for three different cases:

    - Sigmoid hidden and output layers, and cross-entropy cost function

    - Sigmoid hidden layer, softmax output layer, and log-likely cost

    - ReLU hidden layer, softmax output layer, and log-likely cost

The network architecture is three layers with number of neurons [784, 480, 10], and the hyperparameters chosen are:

    - Learning rate scheduled decay of f(epoch) = 0.1 if epoch <= 60 else 0.1*np.exp(1-x/60)
    - L2 regularization parameter of 0.1
    - Momentum co-efficient (for momentum-based gradient descent) of 0.3
    - Batch size (for stochastic gradient descent) of 160

Furthermore, i manually changed the learning rate schedule for the ReLU-softmax-loglikely model, as it finished training due to early stopping before reaching 60 epochs (i.e. before starting to decay). The schedule was therefore changed to 

    f(epoch) = 0.1 if epoch <= 30 else 0.1*np.exp(1-x/30)

hence starting the decay of the learning rate earlier so the model does not stop too early.

It is important to note that the regularization parameter lambda was not adjusted to the change in batch size from the default case, since in the regularization expression i am dividing it by the total number of training examples, not the mini-batch size.

# Results

The three models were trained in 7.1 hours, 5.9 hours and 2.9 hours respectively, and yielded final accuracies on the test data of 97.64%, 97.28% and 97.97%, proving therefore that a combination of rectified linear neurons and a softmax output trains in the smallest time and yields the highest accuracy.