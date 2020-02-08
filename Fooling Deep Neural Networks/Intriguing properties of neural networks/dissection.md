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
6. Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1106–1114, 2012.
    - IDK (Need to read it)
7. 

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

### Capturing Semantics - Single unit vs. Combination of units
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

### Blind Spots in ANNs

#### Non-local generalization
It means that the algorithm should be able to provide good generalizations even for inputs that are far from those it has seen during training. It should be able to generalize to new combinations of the underlying concepts that explain the data. Nearest-neighbor methods and related ones like kernel SVMs and decision trees can only generalize in some neighborhood around the training examples, in a way that is simple (like linear interpolation or linear extrapolation). Because the number of possible configurations of the underlying concepts that explain the data is exponentially large, this kind of generalization is good but not sufficient at all. **Non-local generalization refers to the ability to generalize to a huge space of possible configurations of the underlying causes of the data, potentially very far from the observed data**, going beyond linear combinations of training examples that have been seen in the neighborhood of the given input.

___

Generally speaking, the output layer unit of a neural network is a highly nonlinear function of its input. When it is trained with the cross-entropy loss (using the Softmax activation function), it represents a conditional distribution of the label given the input (and the training set presented so far).

**It has been argued [6] that the deep stack of non-linear layers in between the input and the output unit of a neural network are a way for the model to encode a non-local generalization prior over the input space**. In other words, it is assumed that is possible for the output unit to assign non-significant (and, presumably, non-epsilon) probabilities to **regions of the input space that contain no training examples in their vicinity**. Such regions can represent, for instance, the same objects from different view points, which are relatively far (in pixel space), but which share nonetheless both the label and the statistical structure of the original inputs.

Hence, if we consider local generalization here, it is implicit that the model will assign high probability to the images in the vicinity of a particular image. And this high probability will be assigned to the class to which that particular image (whose neighborhood is being examined) belongs. It can be written mathematically as, for a small enough radius `ε > 0` in the vicinity of a given training input `x`, an `x+r` satisfying `||r|| < ε` will get assigned a high probability of the correct class by the model.

In general, imperceptibly tiny perturbations of a given image do not normally change the underlying class. Why? Because of local generalization.

**Our main result is that for deep neural networks, the above statement does not hold.** Specifically, we show that by using a simple optimization procedure,we are able to find adversarial examples, which are obtained by imperceptibly small perturbations to a correctly classified input image, so that it is no longer classified correctly.

In some sense, what we describe is a way to traverse the manifold represented by the network in an efficient way (by optimization) and finding adversarial examples in the input space. The adversarial examples represent low-probability (high-dimensional) "pockets" in the manifold, which are hard to efficiently find by simply randomly sampling the input around a given example. Already, a variety of recent state of the art computer vision models employ input deformations during training for increasing the robustness and convergence speed of the models [6, 3]. These deformations are, however, statistically inefficient, for a given example: they are highly correlated and are drawn from the same distribution throughout the entire training of the model. We propose a scheme to make this process adaptive in a way that exploits the model and its deficiencies in modeling the local space around the training data.

#### Comparison with Hard-Negative Mining
Hard-negative mining, in computer vision, consists of identifying training set examples (or portions thereof) which are given low probabilities by the model, but which should be high probability instead, cf. [5]. The training set distribution is then changed to emphasize such hard negatives and a further round of model training is performed. As shall be described, the optimization problem proposed in this work can also be used in a constructive way, similar to the hard-negative mining principle. That is, first identify such adversarial examples and retrain neural networks on the these.

## Diving Deeper into Blind Spots

### Observations
1. For all the networks we studied (MNIST, QuocNet, AlexNet), for each sample, we have always managed to generate very close, visually hard to distinguish, adversarial examples that are misclassified by the original network.
2. **Cross model generalization**: a relatively large fraction of examples will be misclassified by networks trained from scratch with different hyper-parameters (number of layers, regularization or initial weights).
3. **Cross training-set generalization**: a relatively large fraction of examples will be misclassified by networks trained from scratch on a disjoint training set.

### Spectral Analysis of Instability
This section describes a **proof for the 1st observation (only) as well as method to control such adversaries**.

Mathematically, if `φ(x)` denotes the output of a network of `K` layers corresponding to input `x` and trained parameters `W`, we write `φ(x) = φk(φk−1(...φ1(x;W1);W2)...;WK)`, where `φk` denotes the operator mapping layer `k−1` to layer `k`. The unstability of `φ(x)` can be explained by inspecting the upper Lipschitz constant of each layer `k=  1...K`, defined as the constantL `k>0` such that `∀x,r ‖φk(x;Wk) − φk(x+r;Wk)‖ <= Lk * ‖r‖`

The resulting network thus satisfies `‖φ(x) − φ(x+r)‖ ≤ L * ‖r‖, with L = ∏ k=1toK Lk`.

To understand it further, we will need to understand the Lipschitz constant.

#### Lipschitz constant
In mathematical analysis, Lipschitz continuity, named after Rudolf Lipschitz, is a strong form of uniform continuity for functions. Intuitively, a **Lipschitz continuous function is limited in how fast it can change**: **there exists a real number such that, for every pair of points on the graph of this function, the absolute value of the slope of the line connecting them is not greater than this real number**; the smallest such bound is called the Lipschitz constant of the function (or modulus of uniform continuity).

Also, a continuous differentiable function is always Lipschitz continuous.

Given two metric spaces `(X, dX)` and `(Y, dY)`, where `dX` denotes the metric (i.e. distance function) on the set `X` and `dY` is the metric on set `Y`, a function `f : X → Y` is called Lipschitz continuous if there exists a real constant `K ≥ 0` such that, for all `x1` and `x2` in `X`,

`dY(f(x1), f(x2)) ≤ K dX(x1, x2)` where, `dY(f(x1), f(x2))` represents the distance between `f(x1)` and `f(x2)` and `dX(x1, x2)` represents the distance between `x1` and `x2`.

Any such `K` is referred to as a Lipschitz constant for the function `f`. The smallest constant is sometimes called the (best) Lipschitz constant; however, in most cases, the latter notion is less relevant. If `K = 1` the function is called a **short map**, and if `0 ≤ K < 1` and f maps a metric space to itself, the function is called a `contraction`.

In particular, a real-valued function `f : R → R` is called Lipschitz continuous if there exists a positive real constant `K` such that, for all real `x1` and `x2`,

`|f(x1) − f(x2)| ≤ K |x1 − x2|`

In this case, `Y` is the set of real numbers `R` with the standard metric `dY(y1, y2) = |y1 − y2|`, and `X` is a subset of `R`.

In general, the inequality is (**trivially**) satisfied if `x1 = x2`. Otherwise, one can equivalently define a function to be Lipschitz continuous if and only if there exists a constant `K ≥ 0` such that, `for all x1 ≠ x2`,

`dY(f(x1) , f(x2)) / dX(x1 , x2) ≤ K`

___

#### Lipschitz constant for nested functions
Final result: `‖φ(x) − φ(x+r)‖ ≤ L * ‖r‖, with L = ∏ k=1toK Lk`.

The above result can be derived for nested Lipschitz function calls.

Take two functions `f : X → Y` and `g : Y → Z`. Now, suppose that,

`dY(f(x1) , f(x2)) / dX(x1 , x2) ≤ L1`

and

`dZ(g(y1) , g(y2)) / dY(y1 , y2) ≤ L2`,

then

```
dZ(g(y1) , g(y2)) / dX(x1 , x2)
= ( dZ(g(y1) , g(y2)) / dX(x1 , x2) ) * ( dY(y1 , y2) / dY(y1 , y2) )
= ( dZ(g(y1) , g(y2)) / dY(y1 , y2) ) * ( dY(y1 , y2) / dX(x1 , x2) )
≤ L2 * ( dY(y1 , y2) / dX(x1 , x2) )
≤ L2 * ( dY(f(x1) , f(x2)) / dX(x1 , x2) )
≤ L2 * L1
```
___

Remember that, we assumed `φ(x)` as the output of a network of `K` layers corresponding to input `x` and trained parameters `W`, i.e. `φ(x) = φk(φk−1(...φ1(x;W1);W2)...;WK)`. Hence, using the above proof, we get `‖φ(x) − φ(x+r)‖ ≤ L * ‖r‖, with L = ∏ k=1toK Lk`. Here, `∏` denotes product.

#### Contractive mapping (functions)
In mathematics, a contraction mapping, or contraction or contractor, on a metric space `(M,d)` is a **function f from M to itself**, with the property that there is some nonnegative real number `0 ≤ k < 1` such that for all `x` and `y` in `M`,

`d(f(x) , f(y)) ≤ k * d(x , y)

The thing to note is the range of k i.e. **0 ≤ k < 1**.

___

#### Short Map / Non-expansive mapping
If `k = 1` in a Lipschitz mapping, then it is called a non-expansive mapping. Note that for contractive mapping, **k < 1**.

___

#### Fixed Point
In mathematics, a fixed point (sometimes shortened to fixpoint, also known as an invariant point) of a function is an element of the function's domain that is mapped to itself by the function. That is to say, **c is a fixed point of the function f if f(c) = c**. This means `f(f(...f(c)...)) = f n(c) = c`, an important terminating consideration when recursively computing f. A set of fixed points is sometimes called a fixed set. 

For example, if f is defined on the real numbers by

`f(x) = x^2 − 3x + 4`

then 2 is a fixed point of f, because f(2) = 2. 

**Not all functions have fixed points**: for example, if f is a function defined on the real numbers as f(x) = x + 1, then it has no fixed points, since x is never equal to x + 1 for any real number. In graphical terms, a fixed point x means the **point (x, f(x)) is on the line y = x**, or in other words the graph of f has a point in common with that line. 

**Points that come back to the same value after a finite number of iterations of the function are called periodic points**. A fixed point is a periodic point with period equal to one. In projective geometry, a fixed point of a projectivity has been called a **double point**.

___

#### Attractive fixed points
An attractive fixed point of a function f is a fixed point x0 of f such that **for any value of x** in the domain that is close enough to x0, the iterated function sequence

`x, f(x) , f(f(x)), f(f(f(x))), ...`

**converges to x0**. An expression of prerequisites and proof of the existence of such solution is given by the [Banach fixed-point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem).

___

So, why did we study all these things? We will look at it's use in this paper in a small time. But the reason we studied fixed points is here..

#### Property of Contractive Mapping
**A contraction mapping has at most one fixed point**. Moreover, the [Banach fixed-point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) states that every contraction mapping on a nonempty complete metric space has a unique fixed point, and that for any x in M the iterated function sequence x, f(x), f(f(x)), f(f(f(x))), ... converges to the fixed point. 

___

Now, getting back to the result and understanding it. `‖φ(x) − φ(x+r)‖ ≤ L * ‖r‖, with L = ∏ k=1toK Lk`

**The formulation above can be interpreted as, for an input image `x` and another input image `x+r`, the difference in final activations of a stack of layers, normalized by the distance between these two images in the image (input) space, is bounded by `L`, where `L` is the product of Lipschitz constants of the activation functions of each layer in the stack. **

Hence, **if Lk > 1 for k=1 to K, then as the depth of network increases, the instability increases too**, because the product of Lk for k=1 to K is going to increase, and hence, the normalized difference between activations **gets a chance** to increase too.

There's a difference between, getting a chance to increase and, increase.

#### Analysis of relu
A half-rectified activation (for both convolutional or fully connected) is defined by the mapping `φk(x;Wk,bk) = max(0,Wkx+bk)`. Let `‖W‖` denote the operator norm of `W` (i.e., **its largest singular value**). Since the **non-linearity ρ(x) = max(0,x) is contractive**, i.e. satisfies `‖ρ(x)−ρ(x+r)‖ ≤ ‖r‖` for all x,r; it follows that

`‖φk(x;Wk) − φk(x+r;Wk)‖ = ‖max(0,Wkx+bk) − max(0,Wk(x+r)+bk)‖ ≤ ‖Wk*r‖ ≤ ‖Wk‖ * ‖r‖`

, and hence `Lk ≤ ‖Wk‖`.

I will make some minor modifications to the author's statements, analyse the relu activation and then understand what this concludes.

Let's look at the plot of relu.

|![Rectified Linear Unit](/Fooling&#32;Deep&#32;Neural&#32;Networks/Intriguing&#32;properties&#32;of&#32;neural&#32;networks/images/relu1.png)|
|:--:|
|Rectified Linear Unit|

Note that for any two points x1 and x2, such that `x1, x2 >= 0`, `(||relu(x1) - relu(x2)||) / (||x1 - x2||) = 1`. Also, in case when `x1 < 0 and x2 >= 0`, `(||relu(x1) - relu(x2)||) / (||x1 - x2||) < 1`.

Hence, it means relu is **either contractive or non-expansive** based on the values of input. 

Now that we know that Lipschitz constant for relu is <=1 (i.e. **relu won't cause any instability in the model, but will try to reduce it in cases where L < 1**), let's go further.

But, relu is applied on either convolution or dense layer. So, let's calculate the Lipschitz constant for these. Let's `w` represent the weights of convolution and fully connected layer. So, we generalized the problem to any layer with weights `w`. Let `L` represent the Lipschitz constant for the function `relu(f(x))`. We have already seen that, if Lipschitz constant of `relu(y)` is `L1` and Lipschitz constant of `f(x)` is `L2`, then `L = L1 * L2`.

The authors said that the value of L2 here is `||w||` but I don't know why. To know how it is proved, read the paper. Hence, `||relu(f(x1)) - relu(f(x2)) <= ||w|| * 1 * ||x1 - x2||` i.e. `||relu(f(x1)) - relu(f(x2)) <= ||w|| * ||r||`.

___

#### Analysis of maxpool
A max-pooling layer `φk` is contractive: `∀x, r, ‖φk(x) − φk(x+r)‖ ≤ ‖r‖`, since its Jacobian is a projection onto a subset of the input coordinates and hence does not expand the gradients.

So, how is Jacobian related to contraction? I am copying the explanation from a stackoverflow post (link given in references section).

At a very basic level, the eigenvectors of a matrix are the directions in which the action of the associated transformation is a simple scaling operation. The associated eigenvalues are the scale factors. By linearity, the overall action of the transformation can be described by a superposition of these “eigen-actions.” If we want the transformation to contract in every direction, then all of its eigenvalues must have absolute value less than unity.

The Jacobian of a map at a point is the best linear approximation to the action of the map near that point and so, per the above explanation, **for the map to be a contraction, the eigenvalues of the Jacobian must have absolute values less than unity**.

___

#### Analysis of contrast normalization
Skipping this part as contrast normalization isn't used anymore in CNNs.

___

|![Table 5](/Fooling&#32;Deep&#32;Neural&#32;Networks/Intriguing&#32;properties&#32;of&#32;neural&#32;networks/images/table5.png)|
|:--:|
|Table 5|

These results are consistent with the existence of blind spots constructed in the previous section, but they **don’t attempt to explain why these examples generalize across different hyperparameters or training sets**. We emphasize that we compute upper bounds: large bounds do not automatically translate into existence of adversarial examples; however, small bounds guarantee that no such examples can appear. This suggests **a simple regularization of the parameters, consisting in penalizing each upper Lipschitz bound, which might help improve the generalisation error of the networks**.

## Unsolved queries (may or may not be a part of this paper but are obviously related to it)
1. The tiny network I trained on MNIST, achieves high accuracy (~96%) when weights of the network are initialized with a random seed of 42. But yields a low value (~76%) when a seed isn't provided. Also, observations without providing seed are also interesting. Creating and training the model for the first time achieves a low accuracy, but **recreating it from scratch** and training the same architecture again, yields an accuracy score of ~95%. I haven't dedicated much time for looking into this (but will do it for sure). But for now, I just have a query in mind. Why is the training **highly** dependent on weights initialization? **Are our optimizers efficient?**
2. What are singular values of a matrix? How is it different from eigen values?
3. Deeper networks perform better in practice, but according to this paper and the proof provided, the instability if network increases as they become deeper. So, maybe they learn more but also get confused more?

## For more information
1. The Limitations of Adversarial Training and the Blind-Spot Attack - https://arxiv.org/pdf/1901.04684.pdf

## Terminology
1. Feature Extraction - Feature vectors extracted from the model. For example, latent vectors in autoencoder, face feature vectors in CNNs using triplet loss function, or word/sentence vectors in RNNs/Transformers.
2. Feature Space - Space consisting of all (not only the ones learned/generated by the model) feature vectors.

## Referennces
1. Intriguing properties of neural networks - https://stats.stackexchange.com/questions/371564/intriguing-properties-of-neural-networks
2. Lipschitz constant - https://en.wikipedia.org/wiki/Lipschitz_continuity
3. Metric - https://en.wikipedia.org/wiki/Metric_(mathematics)
4. Contraction Mapping - https://en.wikipedia.org/wiki/Contraction_mapping
5. Fixed point - https://en.wikipedia.org/wiki/Fixed_point_(mathematics)
6. How to know if a mapping is contractive - https://math.stackexchange.com/questions/2420107/what-is-the-link-between-contraction-mappings-and-the-eigenvalues-of-their-jacob 

## English Vocabulary
1. intriguing - arousing one's curiosity or interest; fascinating.
2. perturbation - change/modification (in mathematics)
3. vicinity - neighborhood