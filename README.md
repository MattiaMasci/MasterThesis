Python project implemented to experiment the effects of the various freezing schemes considered in the thesis "Fast Learning for Deep Neural Network through Casual Freezing".
For training it is possible to choose between two types of neural network: an LSTM with four LSTM layers and three fully-connected layers, a convolutional network that can be created with a number of layers chosen between 11-13-16-19 (VGG).
As is specified in more detail in the thesis, the LSTM network is used as a reference model to do Human Activity Recognition with the UCI-HAR dataset, while VGG11 (and similar) were used to do image classification with the CIFAR-10 dataset.
Below is an example command to train the VGG11 network using the SEQUENTIAL-FREEZING-SCHEMA with freezing period equal to 2 and freezing period fraction equal to 0.5:
python3 __mainVGG__.py --training_method SFS --freezing_period 2 --freezing_span_fraction 0.5 --init init1
