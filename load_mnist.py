import os
import gzip
import numpy as np

class MNISTLoader():

    def __init__(self, trainingdata_path, traininglabels_path, testingdata_path, testinglabels_path):
        self.trainingdata = trainingdata_path
        self.traininglabels = traininglabels_path
        self.testingdata = testingdata_path
        self.testinglabels = testinglabels_path

    @staticmethod
    def load_mnist_images(filename):
        '''
        Load MNIST images
        '''
        with gzip.open(filename, 'rb') as f:
            # Read header information
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_images = int.from_bytes(f.read(4), byteorder='big')
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')

            # Read the image data
            buffer = f.read(num_images * num_rows * num_cols)
            images = np.frombuffer(buffer, dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
            return images / 255.0  # Normalize to [0, 1]

    @staticmethod
    def load_mnist_labels(filename):
        '''
        Load MNIST labels
        '''
        with gzip.open(filename, 'rb') as f:
            # Read header information
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_labels = int.from_bytes(f.read(4), byteorder='big')

            # Read the label data
            buffer = f.read(num_labels)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return labels
    
    @staticmethod
    def convert(images, labels):
        '''
        Converts data format to that used by our neural network, i.e., a list of tuples (image, label)
        '''
        return [(image, label) for image, label in zip(images, labels)]

    def process_all(self):
        '''
        Processes all the data and turns it into the format desired by our neural network
        '''
        training = MNISTLoader.convert(MNISTLoader.load_mnist_images(self.trainingdata), MNISTLoader.load_mnist_labels(self.traininglabels))
        testing = MNISTLoader.convert(MNISTLoader.load_mnist_images(self.testingdata), MNISTLoader.load_mnist_labels(self.testinglabels))

        return training, testing