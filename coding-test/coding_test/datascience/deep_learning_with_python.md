# Deep Learning with Python

## Kernel Methods and SVM
Kernel methods are a group of classification algorithms, the best known of which is the **support vector machine**.

SVM aim at solving classification problems by finding good **decision boundaries** between two sets of points belonging to two different categories. A decision boundary can be thought of as a line or surface separating your training data into two spaces corresponding to two categories.

SVM proceed to find theses boundaries in two steps:
1. the data is mapped to a new high-dimensional representation where the decision boundary can be expressed as a hyperplane.
2. A good decision boundary is computed by trying to maximize the distance between the hyperplane and the closest data points from each class, a step called **maximizing the margin**. This allows the boundary to generalize well to new samples outside of the training dataset.

To find a good decision hyperplanes in the new representation space, you don't have to explicitly compute the coordinates of every point in the new space, you just need to compute the distance between pairs of points in that space, which can be done efficiently using a **kernel function**.

A kernel function is a computationally tractable operation that maps any two points in your initial space to the distance between these points in your target representation space, completely bypassing the explicit computation of the new representation. Kernel functions are typically crafted by hand rather than learned from data. In case of an SVM, only the seperation hyperplane is learned.

## Decision Trees, Random Forests, Gradient Boosting Machines

Decision trees learned from data begin to be preferred to kernel method in the 2010's. In particular, the **Random Forest** algorithm introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their ouputs. **Random Forest** are applicable to a wide variety of problems. A **gradient boosting machine**, much like a random forest, is a machine-learning technique based on ensembling weak prediction models, generally decision trees. It uses **gradient boosting**, a way to improve any machine-learning model by iteratively training new models that specialize in addressing the weak points of the previous models. It may be one of the best, if not the best algorithm for dealing with nonperceptual data today.

Gradient boosting is used for problems where structured data is available.

In 2016 and 2017, Kaggle was dominated by gradient boosting machines and deep learning.

**XGBoost library**

## Deep learning
Deep learning is used for preceptual problems such as image classification.

There are two essentials characteristics of how deep learning learns from data:
* the incremental, layer-by-layer way in which increasingly complex representations are developed
* these intermediate incremental representations are learned jointly, each layer being updated to follow both the representational needs of the layer above and the needs of the layer below.

## The Math of Deep Learning

Learning means finding a combination of model parameters that minimizes a loss function for a given set of training data samples and their correspondiong targets.

Learning happens by drawing random batchs of data samples and their targets, and computing the gradient of the network parameters with respect to the loss on the batch. The network parameters are then moved a bit (the magnitude of the move is defined by the learning rate) in the opposite direction from the gradient.

The entire learning process is made possible by the fact that neural networks are chains of differentiable tensor operations, and thus it's possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value.

The loss is the quantity to minimize during training, so it should represent a measure of success for the task you're trying to solve.


The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum ...

```python
from keras import models, layers

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
network.add(layers.Dense(10, activation="softmax"))

network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
...

network.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Layer extract **representations** out of the data fed into them, hopefully representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive **data distillation**. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined **data filters**- the layers.

The 10-way softmax layer returns an array of 10 probability summing to 1. Each score is the probability that the current digit image belongs to one of the 10 digit classes.

The test-set accuracy turns out to be lower than the training set accuracy. This gap between training accuracy and test accuracy is an example of overfitting. The machine learning model tends to perform worse on new data than on their training data.

Key attributes of tensors:
* number of axes (rank): for instance 3D tensor has three axes and a matrix has two axes. This is also called the tensor's `ndim` in python libraries such as numpy
* shape: how many dimensions the tensor has along each axis
  * scalar: ()
  * vector: (n,)
  * matrix: (n,m)
  * tensor: (b, n, m)

Real world example of data tensors:
* vector data, 2D tensors of shape (samples, features)
* timeseries data or sequence data, 3D tensors of shape (samples, timesteps, features)
* images, 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
* video, 5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

```python
keras.layers.Dense(512, activation="relu")
# equivalent to: output = relu(dot(W, input) + b)
```

`relu` operation is $relu(x)$ is $max(x, 0)$. Relu and addition operations are **element-wise** operations: operations that are applied independently to each entry in the tensors being considered.
BLAS (Basic Linear ALgebra Subprograms) are low-lvel, highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.

What happens with addition when the shapes of the two tensors being added differ? When possible, and if there's no ambiguity, the smaller tensor will be broadcasted to match the shape of the larger tensor.
1. Broadcast axes are added to the smaller tensor to match the `ndim` of the larger tensor.
2. the smaller tensor is repeated alongside these new axes to match the full shape of the largest tensor


The training loop works as follow:
1. Draw batch of training samples x and corresponding targets y
2. Forward pass, run the network on x to obtain predictions y_pred
3. Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y: $loss_value = loss(y_pred, y)$
4. Backard pass: compute the gradient of the loss with regard to the network's parameters: the tensor $gradient(f)(W_0)$ is the gradient of the function $f(W) = loss_value$ in $W_0$
5. Move the parameters a little in the opposite direction from the gradient. Update all weights of the network in a way that slightly reduces the loss on this batch. For example: $W -= step * gradient$. Step is a scaling factor, 

Taking advantage of the fact that all operations used in the network are differentiable, compute the gradient of the loss with regard to the network's coefficients. You can then move the coefficients in the opposite direction from the gradient, thus decreasing the loss.

What is described is called **mini-batch stochastic gradient descent**, mini batch SGD, The term stochastic refers to the fact that each batch of data is drawn at random.

Momentum is implemented by moving the ball at each step based not only on the curret slope (current acceleration), but also on the current velocity (resulting from the past acceleration)

A neural network function consists of many tensor operations chain togetehr. For instance:

$$
f(W_1, W_2, W_3) = a(W_1, b(W_2, c(W_3)))
$$


Calculus tells us that such a chain of functions can be derived usig the chain rule identity $f(g(x)) = f'(g(x)) * g'(x)$

Backpropagation starts with the final loss value and works backward form the top layers to the bottom layers, applying the chain rule to compute the contribution that each parameter had in the loss value.

## Models: networks of layers
A deep-learning model is a directed acycic graph of layers. The topology of a network defines a hypothesis space. By choosing a network topology (linear stack of layers, two-branch networks, multihead networks, inception networks), you constrain your **space of possibilities** (hypothesis space) to a specific series of tensor operations, mapping input data to output data.

## Loss and optimizers
A neural network that has multiple outputs may have multiple loss functions, one per output. But he gradient-descent process must be based on a single-scalar loss value; so for multiloss networks, all losses are combined, via averaging, into a single scalar quantity.

Choosing the right objective function for the right problem is extremely important. When it comes to common problems there are simple guidelines:
* binary cross entropy for two-class classification
* categorical cross entropy for multi-class classification
* mean-squared error for regression
* connectionist temporal classification CTC, for sequence-learning problem
* ...

Without an activation function like `relu`, also called non-linearity, the Dense layer would consist of two linear operations - a dot product and an addition: $output = dot(W, input) + b$
So the layer could only learn **linear transformations**, affine transformations of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16 dimensional space. In order to get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity, or activation function.

Crossentropy is usually the best choice when you're dealing with models that output probabilities. Crossentropy is a quantity from the field of information theory that measures the distance between probability distributions or in this case, between the ground-truth distribution and your predictions.

In binary classification problem, your network should end with a Dense layer with one unit and a sigmoid activvation. the output of your network should be a scalar between 0 and 1, encoding probability. With such a scalar sigmoid ouput on a binary classificaion problem, the loss function should be binary_crossentropy. The rmsprop optimizer is generally a good enough choice, whatever your problem.

In a single-label, multiclass classification problem, the network should en with a softmax activation so that it will ouput a probability distribution over the N ouput classes. The loss function used is generally the categorical crossentropy. It minimizes the distance between the probability distributions output by the network and the true distribution of the targets.
Two ways to handle labels in multiclass classification:
* encoding the labels via categorical encoding, also known as one-hot encoding, and use categorical_crossentropy
* encoding the labels as integers and using the sparce_categorical_crossentropy loss function

> Don't confuse regression and the algorithm **logistic regression**. Confusingly, logistic regression isn't a regression algorithm, it's a classification algorithm

Regression is done using **Mean Squared Error** MSE loss function. A common metric for regression is **Mean Absolute Error** MAE. When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step. When there is little data available, using K-fold validation is a great way to reliably evaluate the model.

## Evaluation
Split the data into three sets:
* training
* validation
* test

Train on the training data and evaluate the model on the validation data. Once the model is ready for prime time, test it one final time on the test data.

* simple hold-out validation: set apart some fraction of your data as your test set. Train on the remaining data, and evaluate on the test set. In order to prevent information leaks, you shouldn't tune your model based on the test set, and therefore you should also reserve a validation set.
* k-fold validation: split the data into K partitions of equal size. For each partition i, train a model on the remaining K-1 partitions, and evaluate it on partition i. The final score is then the averages of the K scores obtained. This method is helpful when the performance of the model shows significant variance based on the train-test split. Like hold-out validation, you also have to use a distinct validation set for model calibration.
* iterated k-fold validation with shuffling: for situation in which you have relatively little data available and you need to evaluate your model as precisely as possible. It consists of applying k-fold validation multiple times, shuffling the data every time before splitting it K ways. The final score is the average of the scores obtained at each run of K-fold validation. You end up training and evaluating P x K models (where P is the number of iterations), which can be very expensive.

Keep an eye on:
* data representativeness: both train and test need to be representative of the data at hand
* the array of time: prevent temporal leak
* redundancy in the data: make sure no duplicate are present

## Data preprocessing, feature engineering and feature learning
All inputs and targets in a neural network must be tensors of floating-point data, or in specific cases, tensors of integers.

In general it isn't safe to feed into a neural network data that takes relatively large values or data that is heterogeneous. Doing so can trigger large gradient updates that will prevent the network from converging. To make learning easier for the network:
* take small values: most values should be in the 0-1 range
* be homogeneous: feature should take values in roughly the same range, normalize each feature indepedently to have a mean of 0, normalize each feature independently to have a standard deviation of 1

Good features still allow to solve problems more elegantly while using fewer resources. Good features let you solve a problem with far less data. The ability of deep-learning models to learn features on their own relies on having lots of training data available; if you have only a few samples, the the information value in their features becomes critical.

## Underfitting and overfitting
The fundamental issue in machine learning is the tension between **optimization** and **generalization**. Optimization refers to the process of adjusting a model to get the best performance possible on training data, whereas generalization refers to how well the trained model performs on data it has never seen before.

To prevent a model from learning misleading or irrelevant patterns found in the training data, the best solution is to get more training data. A model trained on more data will naturally generalize better. The next best solution is to modulate the quantity of information that the model is allowed to store or to add constraints on what information it's allowed to store.

The process of fighting overfitting this way is called regularization:
* reducing the network's size
* adding weight regularization
  * L1 regularization: the cost added is proportional to the absolute value of the weight coefficient, the L1 norm of the weights
  * L2 regularization: the cost added is proportional to the square of the value of the weight coefficient, the L2 norm of the weights. Also called weight decay in the context of neural networks.
* adding dropout: applied to a layer, randomly drop out (setting to zero) a number of output features of the layer during training. the core idea i that introducing noise in the output values of a layer can break up happenstance patterns that aren't significant, which the network will start memorizing if no noise is present

In keras, weight regularization is added by passing weight regularizer instances to layers as keyword arguments.
```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation="relu", input_shape=(10000,)))
...
```

## Deep Learning for Computer Vision
The fundamental difference between a densely connected layer and a convolution layer is that Dense layers learn global patterns in their input feature space, whereas convolution layers learn local pattern: in the case of images, pattern found in small 2D windows of the inputs.

Convnets have two intersting properties:
* the patterns they learn are translation invariant: after learning a certain pattern in the lower-right corner of a picture, a convnet can recognize it anywhere. Therefore they need fewer training samples
* they can learn spatial hierarchies of patterns. a first convolution layer learns small local patterns such as edges, a second convolution layer learns larger patterns mode of features of the first layers... This allows convnets to efficiently learn increasingly complex and abstract visual concepts.

The convolution operation extracts patches, production an **ouptut feature map**. This output feature map is still a 3D tensor. It has a width and a height, its depth is an arbitrary parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather they stand for **filters**. Filter encode specific aspects of the input data, at a high level, a single filter could encode the concept "presence of a face in the input" for instance.

In the MNIST example, the first convolution layer takes a feature map of size (28, 28, 1), and outputs a feature map of size (26, 26, 32): it computes 32 filters over its input. Each of these 32 output channels contains a 26x26 grid of values, which is a **response map** of the filter over the input.

Max pooling consists of extracting window from the input feature maps and outputting the max value of each channel. It's conceptually similar to convolution, except that instead of transforming local patch via a learned linear transformation (the convolution kernel), theyre transformed via a hardcoded `max` tensor operation.

The reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows. Max pooling isn't the only way to achieve such downsampling. One can also uses strides in the prior convolution layer. You can also use average pooling instead of max pooling.

### Using a pretrained convnet: feature extraction
Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.
We reuse the convolutional base because representations learned are likely to be more generic and therefore more reusable. But the representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained.
If the new dataset differs a lot from the dataset on which the original model was trained, you may better off using only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.


### Using a pretrained convnet: fine tuning

It's only possible to fine-tune the top layers of the convolutional base once the classifier on top has already been trained. If the classifier isn't already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed.
Steps for fine-tuning:
* add your custom network on top of an already trained base network
* freeze the base network
* train the part you added
* unfreeze some layers in the base network
* jointly train both these layers and the part you added

Earlier layers in the convolutional base encode more generic, reusable features, whereas layers higher up encode more specialized features. It's more useful to fine-tune the more specialized features, because these are the ones that need to be repurposed on your new problem. There would be fast decreasing returns in fine tuning lower layers.

The more parameters you're training, the more you're at risk of overfitting. The convolutional base has 15 million parameters, so it would be risky to attempt to train it on a small dataset.


How could accuracy stay stable or improve if the loss isn't decreasing? what you display is an average of pointwise loss values; but what matters for accuracy is the distribution of the loss values, not their average, because accuracy is the result of binary thresholding of the class probability predicted by the model. The model may still be improving even if this isn't reflected in the average loss.

## Visualizing what convnets learn
The representations learned by convnets are highly amenable to visualization, in large part because they're **representations of visual concepts**.

* Visualizing intermediate convnet outputs (intermediate activations): useful for understanding how successive convnets layers transform their input, ad for getting a first idea of the meaning of individual convnet filters
* Visualizing convnets filters: useful for understanding precisely what visual pattern or concept each filter in a convnet is receptive to
* Visualizing heatmaps of class activation in an image: useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images.

The features extracted by a layer become increasingly abstract with the depth of the layer. The activations of higher layers carry less and less information about the specific input being seen, and more and more information about the target. A deep neural network effectively acts as an **information distillation pipeline**.

We can inspect the filters learned by convnet by displaying the visual pattern that each filter is meant to respond to. This can be done with **gradient ascent in input space**: applying gradient descent to the value of the input image of a convnet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

## Recurrent Neural Networks
To use dropout with recurrent networks, you should use a time constant dropout mask ad recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the dropout and recurrent_dropout arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They're also much more expensive and thus not always worth it. Although they offer clear gains on complex problems such as machine translation, they may not always be relevant to smaller, simpler problems.

Bidirectional RNNs, which look at a sequence both ways, are useful on natural language processing problems. But they aren't strong performers on sequence data where the recent past is much more informative than the beginning of the sequence.

You can use RNNs for timeseries regression (predicting the future), timeseries classification, anomaly detection in timeseries, and sequence labelingn, such as identifying names or dates in sequences.

You can use 1D convnets for machine translation, sequence to sequence conv models, like SliceNet, document classification and spelling correction.

If **global order matters** in the sequence data, then it's prefereable to use a recurrent network to process it. This is typically the case for timeseries, where the recent past is likely to be more informative than the distant past.

If **global ordering isn't fundamentally meaningful**, then 1D convent will turn out to work at least as well and are cheaper. This is often the case for text data, where a keywork found at the beginning of a sequence is just as meaningul as the keywork found at the end.