
# Optimization study of the neural network using the MNIST dataset. 

The subfolders here contain:

    - hyperparams_study: Study of the effect of different hyperparameters on the cost and accuracy of the neural network neuralnet.py, and a README.md file containing relevant information.

    - best_model: The information from hyperparams_study is used to train 3 different models (varying in the type of neuron, output and cost function) with the best hyperparameter choices. Here the cost and accuracy results of running such a model on the evaluation data during training are shown, as well as a README.md file with the results on the testing dataset.

Of course, the best_model could be further calibrated, as the choice of certain hyperparameters might affect others. These refinements, however, go beyond the scope of this exercise.