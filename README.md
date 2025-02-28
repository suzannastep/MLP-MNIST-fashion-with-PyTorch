
# MLP MNIST-fashion with PyTorch
MLP implementation in Python with PyTorch for the MNIST-fashion dataset (90+ on test).

1. [General](#General)
    - [Background](#background)
    - [Models Structure](https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch/blob/main/README.md#models-structure)
    - [Running Instructions](https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)
4. [Footnote](#footnote)

## General

#### Contributes
* Roei Gida
* Tomer Shay

### Background
Implementation of a neural network on the MNIST dataset, which takes as an input a ``28*28`` grayscale image (``784`` floating point values of pixels between ``0-255``).

### Models Structure
The program contains about seven models of different networks, implemented through ``pytorch``. The last layer size of all the networks is ``10 neurons`` with the ``Softmax`` activation function.

During learning, the network verifies its accuracy on an independent set of data on which learning is not performed. This group is called a ``validation set``. After all the ``epochs``, the network saves its best state, the weights that resulted the maximum accuracy on the validation set, to prevent overfitting.

Finally, the network exports a graph of the accuracy on the training, validation, and the testing sets, by the number of epochs, and prints the final accuracy on the testing set.

The seven models architecture:
1. ``Model A``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.12
	* No batch normalization, no dropout

1. ``Model A middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 100, 100, 100, 50, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.12
	* No batch normalization, no dropout

1. ``Model B``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* No batch normalization, no dropout

1. ``Model B middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 100, 100, 100, 50, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* No batch normalization, no dropout

1. ``Model C``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* **dropout:** 20% on the 3rd layer (size of 50)
	* No batch normalization

1. ``Model C middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 100, 100, 100, 50, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* **dropout:** 20% on the penultimate layer (size of 50)
	* No batch normalization

1. ``Model D``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.01
	* **batch normalization:** before the activation function (ReLU) on each of the hidden layers.
	* No dropout

1. ``Model D middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 100, 100, 100, 50, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.01
	* **batch normalization:** before the activation function (ReLU) on each of the hidden layers.
	* No dropout

1. ``Model E``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [ReLU, ReLU, ReLU, ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.1
	* No batch normalization, no dropout

1. ``Model E middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.1
	* No batch normalization, no dropout

1. ``Model F``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [Sigmoid, Sigmoid, Sigmoid, Sigmoid, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.001
	* No batch normalization, no dropout

1. ``Model F middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [Sigmoid, Bias-Free Linear, Bias-Free Linear, Sigmoid, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.001
	* No batch normalization, no dropout

1. ``Model G`` (best model):
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 512, 256, 128, 64, 10]
	* **activation function:** [Leaky ReLU, Leaky ReLU, Leaky ReLU, Leaky ReLU, Softmax]
	* **optimizer:** starts with ADAM, then to SGD
	* **learning rate:** 0.001
	* **batch normalization:** before the activation function (Leaky ReLU) on each of the hidden layers.
	* **dropout:** 10% on the input layer (size of 784), 3rd layer (size of 256), and 5th layer (size of 64).

1. ``Model G middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 512, 256, 128, 64, 10]
	* **activation function:** [Leaky ReLU, Bias-Free Linear, Bias-Free Linear, Leaky ReLU, Softmax]
	* **optimizer:** starts with ADAM, then to SGD
	* **learning rate:** 0.001
	* **batch normalization:** before the activation function (Leaky ReLU) on each of the hidden layers.
	* **dropout:** 10% on the input layer (size of 784), 3rd layer (size of 256), and 5th layer (size of 64).

1. ``Model H``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 1000, 1000, 1000, 1000, 10]
	* **activation function:** [ReLU, ReLU, ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.001
	* **batch normalization:** None
	* **dropout:** 10% on the input layer, 3rd layer, and 5th layer.

1. ``Model H middle linear``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 1000, 1000, 1000, 1000, 10]
	* **activation function:** [ReLU, Bias-Free Linear, Bias-Free Linear, Bias-Free ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.001
	* **batch normalization:** None
	* **dropout:** 10% on the input layer, 3rd layer, and 5th layer.

To get the best percentages on the testing set (90+), our experiments showed that the ``Model G`` should be run for about ``30`` epochs, with ``batch size = 64`` and ``validation percentage = 10%``.

### Model Chart
| Model | # Hidden Layers | Model Sketch | Optimizer | Learning Rate | Batch Normalization | Dropout |
| - | - | - | - | - | - | - |
| A     | 2      | 100 ReLU 50 ReLU 10 Softmax                       | SGD  | 0.12 |  |  | 
| A (ML)| 4      | 100 ReLU 100 Bias-Free Linear 100 Bias-Free Linear 50 Bias-Free ReLU 10 Softmax | SGD  | 0.12 |  |  | 
| B     | 2      | 100 ReLU 50 ReLU 10 Softmax | Adam | 1e-4 |  |  | 
| B (ML)| 4      | 100 ReLU 100 Bias-Free Linear 100 Bias-Free Linear 50 Bias-Free ReLU 10 Softmax | Adam | 1e-4 |  |  | 
| C     | 2      | 100 ReLU 50 ReLU 10 Softmax | Adam | 1e-4 |  | 20% on penultimate layer | 
| C (ML)| 4      | 100 ReLU 100 Bias-Free Linear 100 Bias-Free Linear 50 Bias-Free ReLU 10 Softmax | Adam | 1e-4 |  | 20% on penultimate layer | 
| D     | 2      | 100 ReLU 50 ReLU 10 Softmax | Adam | 1e-2 | Before each ReLU |  | 
| D (ML)| 4      | 100 ReLU 100 Bias-Free Linear 100 Bias-Free Linear 50 Bias-Free ReLU 10 Softmax | Adam | 1e-2 | Before each ReLU |  | 
| E     | 4      | 128 ReLU 64 ReLU 10 ReLU 10 ReLU 10 Softmax | SGD | 1e-1 |  |  | 
| E (ML)| 4      | 128 ReLU 64 Bias-Free Linear 10 Bias-Free Linear 10 Bias-Free ReLU 10 Softmax | SGD | 1e-1 |  |  | 
| F     | 4      | 128 Sigmoid 64 Sigmoid 10 Sigmoid 10 Sigmoid 10 Softmax | Adam | 1e-3 |  |  | 
| F (ML)| 4      | 128 Sigmoid 64 Bias-Free Linear 10 Bias-Free Linear 10 Bias-Free Sigmoid 10 Softmax | Adam | 1e-3 |  |  | 
| G     | 4      | 512 LeakyReLU 256 LeakyReLU 128 LeakyReLU 64 LeakyReLU 10 Softmax | 14 epochs of Adam, then SGD | 1e-3 | Before each LeakyReLU | 10% on the input layer (size of 784), 3rd layer (size of 256), and 5th layer (size of 64) | 
| G (ML)| 4      | 512 LeakyReLU 256 Bias-Free Linear 128 Bias-Free Linear 64 Bias-Free LeakyReLU 10 Softmax | 14 epochs of Adam, then SGD | 1e-3 | Before each LeakyReLU | 10% on the input layer (size of 784), 3rd layer (size of 256), and 5th layer (size of 64) | 
| H     | 4      | 1000 ReLU 1000 ReLU 1000 ReLU 1000 ReLU 10 Softmax | Adam | 1e-3 |  | 10% on the input layer, 3rd layer, and 5th layer | 
| H     | 4      | 1000 ReLU 1000 Bias-Free Linear 1000 Bias-Free Linear 1000 Bias-Free ReLU 10 Softmax | Adam | 1e-3 |  | 10% on the input layer, 3rd layer, and 5th layer | 


- (ML) denotes the "middle linear" version of the model
- Trained on negative log likelihood loss with a scheduler that decays the learning by a factor of 0.2 every 7 epochs. 
- After each epoch, the validation error of the model is recorded; the final trained model is the model at the epoch with the best validation error.
- If the validation accuracy ever surpasses 90% at the end of an epoch, then the learning rate is set to 1e-4.
- Trained for 30 epochs with a batch size of 64

### Running Instructions

The program gets several arguments, and this can be seen with the ``-h`` or with ``-help`` flags when running. A total of about ten arguments can be sent:
* **flag ```-train_x STRING```:** A ``String`` for the training images file path (file that contains 784 values in each row). *NOTE: this flag will be used only if ``-local True`` was enterd.*
* **flag ```-train_y STRING```:** A ``String`` for the training labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file). *NOTE: this flag will be used only if ``-local True`` was enterd.*
* **flag ```-test_x STRING```:** A ``String`` for the testing images file path (file that contains 784 values in each row). *NOTE: this flag will be used only if ``-local True`` was enterd.*
* **flag ```-test_y STRING```:** A ``String`` for the testing labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file). *NOTE: this flag will be used only if ``-local True`` was enterd.*
* **flag ```-e INT```:** An ``Integer`` for the number of epochs (``default value = 10``).
* **flag ```-batch_size INT```:** An ``Integer`` for the batch size (``default value = 64``).
* **flag ```-validate INT```:** An ``Integer`` for the percentage of the training set that should be allocated to the validation set (``default value = 10``).
* **flag ```-model STRING```:** A ``String`` that says with which model to work in the program run. You can send ``'A'`` - ``'F'`` or ``'ModelG'`` (``default value = ModelG``).
* **flag ```-local BOOLEAN```:** ``True`` to load the dataset locally (according to the paths entered), or ``False`` to load the original MNIST-fashion dataset (``default value = False``).
* **flag  `-plot BOOLEAN`:** ``True`` to export a graph of the percentage of accuracy and loss value in each epoch (`default value = True`).

running example:
```
	$ python3 mian.py -train_x train_x -train_y train_y -test_x test_x -test_y test_y -local True
```

Note that for using the dataset given in this repo, you need to unzip the dataset.zip folder (using 7-zip for example)
## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* Git
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Argparse](https://pypi.org/project/argparse/)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [torchvision](https://pypi.org/project/torchvision/)

## Installation

1. Open the terminal.
2. Clone the project by:
	```
	$ git clone https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch.git
	```	
3. Enter the project folder:
	```
	$ cd MLP-MNIST-fashion-with-PyTorch
	```
4. You can unzip the ``dataset.zip`` for local running:
	```
	$ unzip dataset.zip
	```
5. Run the ```main.py``` file with your favorite parameters:
	```
	$ python3 main.py -e 30 -validate 10 -model ModelG -batch_size 64
	 ```

 
## Footnote:
As you can see, there are several additional files. The files contain different graphs of different experiments of the program with some changes of the hyperparameters (e.g. learning rate, batch size, etc.).
