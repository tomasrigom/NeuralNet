# NeuralNet
 Repository for development of a trainable vanilla neural network, inspired by Michael Nielsen's book on neural networks (available online at http://neuralnetworksanddeeplearning.com/index.html), **extended with additional functionalities and vectorized for improved efficiency**.

 This repository also includes an optimizaton study of this basic neural network on the **MNIST dataset**. The files are:
 - **`neuralnet.py`**: Main file that contains the definition of the neural network object and its functionalities. Full description is found within the file. There are also defined the classes for the different activation functions for the neurons and the cost (loss) functions, which include derivative-related functionalities for the backpropagation algorithm.
 - **`load_mnist.py`**: File containing the definition of the dataloader objects for the mnist dataset. Note this were written by me, not using pytorch or any python packages. The purpose of this repository is to use as few packages as possible.
 - **`basic_hyperparameter_search.py`**: File for performing a basic optimization study of the effect of different hyperparameters on the performace of the network. The results can be found in the **plots** directory.
  - **`train_best_model.py`**: File for performing an optimization study of the different activation and cost functions on the network using the resulting best parameters from the previous study. These results can also be found in the **plots** directory.
 - **`autoencoder.py`**: Testing of the autoencoder functionality of the Network object found in **`neuralnet.py***.
 - **plots**: Plotted results obtained from **`basic_hyperparameter_search.py`** and **`train_best_model.py`**.
