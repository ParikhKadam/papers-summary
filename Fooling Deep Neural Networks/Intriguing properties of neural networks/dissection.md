# Intriguing properties of neural networks
https://arxiv.org/pdf/1312.6199.pdf

## Related Works
1. Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. arXiv preprint arXiv:1311.2524, 2013.
    - IDK (Need to read it)
2. Ian Goodfellow, Quoc Le, Andrew Saxe, Honglak Lee, and Andrew Y Ng. Measuring invariances in deep networks. Advances in neural information processing systems, 22:646–654, 2009.
    - IDK (Need to read it)
3. Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional neural networks. arXiv preprint arXiv:1311.2901, 2013.
    - IDK (Need to read it)
4. Thomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.
    - IDK (Need to read it)
5. Dumitru Erhan, Yoshua Bengio, Aaron Courville, and Pascal Vincent. Visualizing higher-layer features of a deep network. Technical Report 1341, University of Montreal, June 2009. Also presented at the ICML 2009 Workshop on Learning Feature Hierarchies, Montr ́eal, Canada.
    - IDK (Need to read it)
6. 

## Introduction
This paper concludes two different properties of neural networks:
1. Semantic meaning of individual units.
2. Stability of neural networks with respect to small perturbations to their inputs.

At first, let us look at the first property. The previous works [1, 2, 3] have analyzed the semantic meaning of various units (neurons) by **finding the set of inputs that maximally activate a given unit**. The inspection of individual units makes the **implicit assumption that the units of the last feature layer form a distinguished basis** which is particularly useful for extracting semantic information. Generally, it seems that **it is the entire space of activations, rather than the individual units**, that contains the bulk of the semantic information.

A similar, but even stronger conclusion was reached recently by Mikolov et al. [4] for word representations, where the various directions in the vector space representing the words are shown to give rise to a surprisingly rich semantic encoding of relations and analogies. At the same time, the vector representations are stable up to a rotation of the space, so the individual units of the vector representations are unlikely to contain semantic information.

Now, let's get to second one. Consider a state-of-the-art deep neural network that generalizes well on an object recognition task. We expect such network to be robust to small perturbations of its input, because small perturbation cannot change the object category of an image. However, we find that **applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction** (see figure 5). These perturbations are found by optimizing the input to maximize the prediction error (i.e using gradient ascent). Usually, we train a model by minimizing error and adjusting the weights. But here, we perturb an image by maximizing the error and adjusting the pixels of that image. We term the so perturbed examples “adversarial examples”.

If we think of using gradient ascent this way, then one might expect that the adversarial examples generated this way might be unique to a particular neural network, because these examples are produced by maximizing the loss of that particular network. But it is not the case. The authors found that adversarial examples are relatively robust, and are **shared by neural networks with varied number of layers, activations or trained on different subsets of the training data**. That is,if we use one neural net to generate a set of adversarial examples, we find that these examples are still statistically hard for another neural network even when it was trained with different hyperparameters or, most surprisingly, when it was trained on a different set of examples.

These results suggest that the deep neural networks that are learned by backpropagation have non-intuitive characteristics and intrinsic blind spots, whose structure is connected to the data distribution in a non-obvious way.

Though, my views are a little bit different:
1. It is an intuitive characteristic. Neural networks don't always learn a continuous function (apart from a few generative models, so I think generative models should be robust to adversarial examples). That is, if `f(x)` is a function learned by a network, the values of `x` (domain of `f(x)`) for which it learned it perfectly is discrete. One can relate this to the variability in training set.
2. Also, one can think about the discreteness of weights learned by the network. The weights of a network aren't continuous. A minor change in weight of a single unit **can** (and not **will**) bring a large change in network's output. If the weights are learned by the network in a continuous fashion, I believe that the networks will be robust to such adversarial examples. 
3. Also, one other point is, the networks now-a-days are built too large for just a minor improvement in it's accuracy. I recently [experimented a tiny network for MNIST classification which was able to achieve 96% accuracy and loss of ~0.13](https://github.com/ParikhKadam/mnist-experiments/blob/master/experiment1.ipynb). Now, people build [a large model to get an improvement of 3-4% over this](https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer). Here, I think that as the size of network increases, the number of blind spots (the combinations of weights of the network which are left to learn by the network i.e. for `f(x;theta)`, the various `theta` values that the network forgot to take into account) in the network increase.
    - While, the authors think that blind spots exist in input, I think they exist as well in weights.
    - That is, they think that blind spots are the images unseen by the model. While, I think it's because a number of combination of weights of different units is left unseen by the model.
    - In order to solve this, one must provide too many input images during training so that the model can explore different combination of weights for different images.
    - Hence, the solution for both remains the same. But there is one other solution for my understanding of blind spots. And the solution is **smaller networks**.
    - Will try adversarial attacks on a tiny network to confirm this hypothesis. Till then, it's just an intuitive theory.

## Work

### Formulation
Traditional computer vision systems rely on **feature extraction**. Often a single feature is easily interpretable. This allows one to inspect the individual coordinates of the **feature space**, and link them back to meaningful variations in the input domain. This means, change a few values in feature vector and propagate the changes back to input. In an intuitive way, it is nothing but changing a few values in an image feature vector and propagating the changes back into network to generate an image from this feature vector. The generated image is said to have the features represented by this feature vector.

Similar reasoning was used in previous work that attempted to analyze neural networks that were applied to computer vision problems. These works interpret an **activation of a hidden unit as a meaningful feature** (features extracted by intermediate layer(s)/unit(s)). They look for input images which maximize the activation value of this single feature [1, 2, 3, 5].

The aforementioned technique can be formally stated as visual inspection of images `x′`, which satisfy (or are close to maximum attainable value):
`x′= argmax x∈I〈φ(x),ei〉`. Here, 〈x,y〉 represents the inner product of `x` and `y`, `I` is a held-out set of images from the data distribution that the network was not trained on and `ei` is the natural basis vector associated with the i-th hidden unit, `φ(x)` is the activation values of some layer (not a single unit).

Our experiments show that any random direction `v∈Rn` gives rise to similarly interpretable semantic properties. More formally, we find that images `x′` are semantically related to each other, for many `x′` such that `x′= argmax x∈I〈φ(x),v〉`. This suggests that the natural basis is not better (also not worse) than a random basis for inspecting the properties of `φ(x)`.

The above two paragraphs may no tbe clear. So, let's understand them first. The authors denote an entire layer of activations via `ϕ(x)`, for any input image `I`. So imagine we have a layer with three neurons, whose activations we denote via: `(ϕ1(x),ϕ2(x),ϕ3(x))`. The claim that some earlier papers made is that **certain neurons respond to certain features**. As an example, you could figure out which images maximally activate the first neuron, by computing the maximum value attained by that neuron over a set of test images: `argmax x∈I ϕ1(x)`. Then you could find images which achieve similar activation strength to the maximum, and make claims about similarities between those images (e.g. diagonal strokes). To be clear, the above claim is empirically correct: **certain neurons are more sensitive to particular features**. Before proceeding, note that `ϕ1(x) = <ϕ(x),e1>`, e.g. the right side is just a dot product between the full vector of activations and the first unit vector, which pops out the first neuron's activation.

The linked paper argues however, that **it is incorrect to conclude that neural networks disentangle images into features across individual neurons**. More precisely, neural networks **may not** disentangle image features across individual neurons. Let's understand this first. Yes, neural networks learn image features. But it not right to believe that a single neuron learns a single feature. That is, it might be a case that a group of neurons together learn a unique feature. More concisely, if we take out a neuron from this group of neurons, and examine it, we may find that the remaining part of group has learned nothing. That is, we need all the neurons together in order to understand the feature it (the group) has learned.

Because, if certain neurons respond more toward a given feature, one might think that the network tries to be efficient and not do the same encoding work twice across multiple neurons. Note that part of the reason for this is using **dropout**, which would **make it more difficult for single neurons to focus on a specific feature**.

To prove this, the authors instead consider optimizing `argmax x∈I <ϕ(x),v> = argmax x∈I <ϕ1(x)v1+ϕ2(x)v2+ϕ3(x)v3>`, where v is a random vector. So now instead of focusing on a particular neuron, we're focusing on an arbitrary linear combination. They show that images that give high activations for this are also semantically related.

We used the MNIST test set for `I`. Figure 1 shows images that maximize the activations in the natural basis, and Figure 2 shows images that maximize the activation in random directions. In both cases the resulting images share many high-level similarities. Next, we repeated our experiment on an AlexNet, where we used the validation set as `I`. Figures 3 and 4 compare the natural basis to the random basis on the trained network. The rows appear to be semantically meaningful for both the single unit and the combination of units.

|![Figure 1](/Fooling%20Deep%20Neural%20Networks/Intriguing%20properties%20of%20neural%20networks/images/fig1.png)|
|:--:|
|Figure 1|

|![Figure 2](/Fooling%20Deep%20Neural%20Networks/Intriguing%20properties%20of%20neural%20networks/images/fig2.png)|
|:--:|
|Figure 2|

|![Figure 3](/Fooling%20Deep%20Neural%20Networks/Intriguing%20properties%20of%20neural%20networks/images/fig3.png)|
|:--:|
|Figure 3|

|![Figure 4](/Fooling%20Deep%20Neural%20Networks/Intriguing%20properties%20of%20neural%20networks/images/fig4.png)|
|:--:|
|Figure 4|

This suggests that **the natural basis is not better than a random basis** for inspecting the properties of `φ(x)`. 

### Non-local generalization
It means that the algorithm should be able to provide good generalizations even for inputs that are far from those it has seen during training. It should be able to generalize to new combinations of the underlying concepts that explain the data. Nearest-neighbor methods and related ones like kernel SVMs and decision trees can only generalize in some neighborhood around the training examples, in a way that is simple (like linear interpolation or linear extrapolation). Because the number of possible configurations of the underlying concepts that explain the data is exponentially large, this kind of generalization is good but not sufficient at all. **Non-local generalization refers to the ability to generalize to a huge space of possible configurations of the underlying causes of the data, potentially very far from the observed data**, going beyond linear combinations of training examples that have been seen in the neighborhood of the given input.

### Blind Spots in ANNs
Generally speaking, the output layer unit of a neural network is a highly nonlinear function of its input. When it is trained with the cross-entropy loss (using the Softmax activation function), it represents a conditional distribution of the label given the input (and the training set presented so far).

**It has been argued [6] that the deep stack of non-linear layers in between the input and the output unit of a neural network are a way for the model to encode anon-local generalization prior over the input space**. In other words, it is assumed that is possible for the output unit to assign non-significant (and, presumably, non-epsilon) probabilities to **regions of the input space that contain no training examples in their vicinity**. Such regions can represent, for instance, the same objects from different view points, which are relatively far (in pixel space), but which share nonetheless both the label and the statistical structure of the original inputs.

Hence, if we consider local generalization here, it is implicit that the model will assign high probability to the images in the vicinity of a particular image. And this high probability will be assigned to the class to which that particular image (whose neighborhood is being examined) belongs. It can be written mathematically as, for a small enough radius `ε > 0` in the vicinity of a given training input `x`, an `x+r` satisfying `||r|| < ε` will get assigned a high probability of the correct class by the model.

In general, imperceptibly tiny perturbations of a given image do not normally change the underlying class. Why? Because of local generalization.

Our main result is that for deep neural networks, the above statement does not hold. Specifically, we show that by using a simple optimization procedure,we are able to find adversarial examples, which are obtained by imperceptibly small perturbations to a correctly classified input image, so that it is no longer classified correctly.

## Unsolved queries (may or may not be a part of this paper but are obviously related to it)
1. The tiny network I trained on MNIST, achieves high accuracy (~96%) when weights of the network are initialized with a random seed of 42. But yields a low value (~76%) when a seed isn't provided. Also, observations without providing seed are also interesting. Creating and training the model for the first time achieves a low accuracy, but **recreating it from scratch** and training the same architecture again, yields an accuracy score of ~95%. I haven't dedicated much time for looking into this (but will do it for sure). But for now, I just have a query in mind. Why is the training **highly** dependent on weights initialization? **Are our optimizers efficient?**

## For more information
1. The Limitations of Adversarial Training and the Blind-Spot Attack - https://arxiv.org/pdf/1901.04684.pdf

## Terminology
1. Feature Extraction - Feature vectors extracted from the model. For example, latent vectors in autoencoder, face feature vectors in CNNs using triplet loss function, or word/sentence vectors in RNNs/Transformers.
2. Feature Space - Space consisting of all (not only the ones learned/generated by the model) feature vectors.

## Referennces
1. Intriguing properties of neural networks - https://stats.stackexchange.com/questions/371564/intriguing-properties-of-neural-networks
2. 

## English Vocabulary
1. perturbation - change/modification (in mathematics)
2. vicinity - neighborhood