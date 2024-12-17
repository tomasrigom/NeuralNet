
# Optimization Study

This study explores the optimization of our vanilla neural network's hyperparameters for its training on the **MNIST dataset**. The directory contains the following subfolders:

### 1. **`hyperparams_study`**
This folder includes a study of the effect of different hyperparameters on the cost and accuracy of the neural network implemented in `neuralnet.py`, and a `README.md` file with relevant information summarizing the hyperparameter tuning process and findings.

### 2. **`best_model`**
This folder contains the results of training three different models using the optimal hyperparameters found in `hyperparams_study` while varying the type of neuron, output and cost function, and a `README.md` file detailing the study and results of running the models on the test dataset.

---

While the experiments in **best_model** use optimal hyperparameters derived from `hyperparams_study`, further calibration could be performed as some hyperparameters might have a dependence on others. This is why the approach of varying one parameter at a time might not be the best. These precise calibrations, however, are beyond the scope of this exercise.