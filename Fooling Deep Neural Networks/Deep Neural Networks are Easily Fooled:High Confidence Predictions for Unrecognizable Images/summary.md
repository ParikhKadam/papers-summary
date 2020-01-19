## Related Works
1. C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.
   - Changing an image, originally correctly classified (e.g. as a lion), in a way imperceptible to human eyes, can cause a DNN to label the image as something else entirely (e.g. mislabeling a lion as a library).
   - For example, Changing a few pixels in an image of lion can mislead the DNN to classify it as a library. Though the altered image seems like a lion to human, the DNN recognises it as a library.
2. D. Floreano and C. Mattiussi. Bio-inspired artificial intelligence:  theories,  methods,  and  technologies.   MIT  press,2008.
   - IDK (Need to read it)
3. A. Cully, J. Clune, and J.-B. Mouret. Robots that can adapt like natural animals. arXiv preprint arXiv:1407.3501, 2014.
   - IDK (Need to read it)
4. K. Deb.Multi-objective optimization using evolutionary algorithms, volume 16. John Wiley & Sons, 2001.
   - IDK (Need to read it)
5. H. Lipson. Principles of modularity, regularity, and hierarchy for scalable systems.Journal of Biological Physics andChemistry, 7(4):125, 2007.
   - IDK (Need to read it)
6. K. O. Stanley.  Compositional pattern producing networks: A novel abstraction of development. Genetic programming and evolvable machines, 8(2):131–162, 2007
   - IDK (Need to read it)
7. . Secretan, N. Beato, D. B. D Ambrosio, A. Rodriguez,A. Campbell, and K. O. Stanley. Picbreeder: evolving pictures collaboratively online. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages1759–1768. ACM, 2008.
   - IDK (Need to read it)
8. J. E. Auerbach.Automated evolution of interesting im-ages.  In Artificial Life 13, number EPFL-CONF-191282.MIT Press, 2012.
   - IDK (Need to read it)
9.  

## Introduction
It is easy to produce images that are completely unrecognizable to humans (Fig. 1), but that state-of-the-art DNNs believe to be recognizable objects with over 99% confidence (e.g. labeling with certainty that TV static (refers to black and white dots screen on TV) is a motorcycle).

![Figure 1](images/fig1.jpg)

Figure 1

<details open>
  <summary>Comparision with [1]</summary>

    This differs from [1] in a sense that, in [1] they modified the pixels of a lion image (the image contained a legit object) and the network misclassified it as library. Whereas, in this paper, the authors take a **garbage image** i.e. image with black and white dots (no legit object), and the **model classifies it as a motorcycle**.
</details>


We use **evolutionary algorithms** or **gradient ascent** to generate images.

We also find that, for MNIST DNNs, **it is not easy to prevent the DNNs from being fooled** by retraining them with fooling images labeled as such. While retrained DNNs learn to classify the negative examples as fooling images, anew batch of fooling images can be produced that fool these new networks, even after many retraining iterations.

Here, the authors experimented with MNIST dataset. They added a new class 10 for garbage images. Prepared a dataset of such garbage images to train the network on it, with an expectation that these are garbage images and stop misclassifying them. As expected, the network learned patterns (after all that's what the ML models do) and stopped misclassifying such images as legit target classes. But now, running the algorithm (mentioned above) with this newly trained model, it generated new garbage images which the model will misclassify. Repeating these steps, it showed no good results.

### Topics discussed
1. Comparison between human vision and DNN-based computer vision
2. Performance of DNNs, in general, across different types of images than the ones they have been trained and traditionally tested on.

## Work

### Pre-trained models used
1. ImageNet DNN - AlexNet + ImageNet
2. MNIST DNN - LeNet + MNIST

### Generating images using evolutionary algorithm
EAs are optimization algorithms inspired by Darwinian evolution (the theory of evolution of humans -- from monkey to human). They contain a population of "organisms" (here, images) that alternately face **selection** (keeping the best) and then **random perturbation** (mutation and/or crossover). Which organisms are selected depends on the **fitness function**, which in these experiments is the highest prediction value a DNN makes for that image belonging to a class (Fig. 2).

![Figure 2](images/fig2.png)

Figure 2

The EA mentioned in [2] optimize solutions to perform well on one objective or on a small set of objectives (e.g. evolving images to match a single ImageNet class).

So, we use a new algorithm called the multi-dimensional archive of phenotypic elites MAP-Elites [3], which enables us to simultaneously evolve a population that contains individuals that score well on many classes (e.g. all 1000 ImageNet classes).

**Fitness** is determined by showing the image to the DNN; if the image generates a higher prediction score (probability) **for any class** than has been seen before, the newly generated individual (image) becomes the champion in the archive **for that class**.

<details open>
  <summary>Representation of Images as genome (Image Encoding)</summary>

    Will update from here after going through the references.
</details>
